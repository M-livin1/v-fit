import cv2
import numpy as np
import os

print("Shirt preprocessing started")

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT  = os.path.join(ROOT, "data",    "shirt.png")
OUTPUT = os.path.join(ROOT, "results", "cloth_clean.png")

img = cv2.imread(INPUT)
if img is None:
    raise Exception("Shirt image not found: " + INPUT)

h, w = img.shape[:2]

# ── STEP 1: Multi-method background detection ─────────────────────────────────
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# White/near-white background (typical product shots)
_, mask_bright = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

# Otsu's auto-threshold
_, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

mask_combined = cv2.bitwise_or(mask_bright, mask_otsu)

# ── STEP 2: GrabCut refinement ────────────────────────────────────────────────
gc_mask = np.where(mask_combined > 128, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)

border = 5
gc_mask[:border, :]  = cv2.GC_BGD
gc_mask[-border:, :] = cv2.GC_BGD
gc_mask[:, :border]  = cv2.GC_BGD
gc_mask[:, -border:] = cv2.GC_BGD

eroded = cv2.erode(mask_combined, np.ones((15, 15), np.uint8), iterations=2)
gc_mask[eroded > 128] = cv2.GC_FGD

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

try:
    cv2.grabCut(img, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    mask_gc = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
except cv2.error:
    print("GrabCut failed, using threshold mask only")
    mask_gc = mask_combined

# ── STEP 3: Morphological cleanup (aggressive hole filling) ───────────────────
kernel = np.ones((7, 7), np.uint8)
mask_gc = cv2.morphologyEx(mask_gc, cv2.MORPH_CLOSE, kernel, iterations=5)
mask_gc = cv2.morphologyEx(mask_gc, cv2.MORPH_OPEN,  kernel, iterations=1)

# Fill all internal holes via largest contour fill
contours, _ = cv2.findContours(mask_gc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_filled = np.zeros_like(mask_gc)
if contours:
    # Keep the largest contour (the shirt itself)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask_filled, [largest], -1, 255, thickness=cv2.FILLED)

# Flood-fill any remaining interior holes
flood = mask_filled.copy()
flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(flood, flood_fill_mask, (0, 0), 255)
flood_inv = cv2.bitwise_not(flood)
mask_filled = cv2.bitwise_or(mask_filled, flood_inv)

# ── STEP 4: Create a HARD (binary) alpha — no semi-transparency ───────────────
# We use a very small feather ONLY at the very edge for anti-aliasing
feather_radius = max(3, int(min(h, w) * 0.004))
if feather_radius % 2 == 0:
    feather_radius += 1

# Erode slightly before feathering so the feathered edge stays inside the shirt
alpha_core = cv2.erode(mask_filled, np.ones((3, 3), np.uint8), iterations=1)
alpha = cv2.GaussianBlur(alpha_core, (feather_radius, feather_radius), 0)

# ── STEP 5: Create BGRA output ────────────────────────────────────────────────
b, g, r = cv2.split(img)
rgba = cv2.merge([b, g, r, alpha])

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
cv2.imwrite(OUTPUT, rgba)

print("Shirt background removed — solid alpha, feathered edges only")
print("Saved:", OUTPUT)