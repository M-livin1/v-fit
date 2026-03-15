"""
overlay.py – Simple shirt display

New behavior (per your request):
1. Do NOT render the user at all.
2. Take the cleaned shirt image produced by preprocessing (`cloth_clean.png`).
3. Composite it onto a plain white background and save as `final_overlay.jpg`.
"""

import cv2
import numpy as np
import os

print("Simple shirt overlay started")

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SHIRT_CLEAN  = os.path.join(BASE_DIR, "results", "cloth_clean.png")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "results", "final_overlay.jpg")

# Load the cleaned shirt with alpha channel
shirt_rgba = cv2.imread(SHIRT_CLEAN, cv2.IMREAD_UNCHANGED)
if shirt_rgba is None:
    raise FileNotFoundError("cloth_clean.png missing")

h, w = shirt_rgba.shape[:2]

# Split into color and alpha
if shirt_rgba.shape[2] == 4:
    bgr = shirt_rgba[:, :, :3]
    alpha = shirt_rgba[:, :, 3].astype(np.float32) / 255.0
else:
    # Fallback in case the image has no alpha (treat as fully opaque)
    bgr = shirt_rgba
    alpha = np.ones((h, w), dtype=np.float32)

# Create a white background
background = np.full((h, w, 3), 255, dtype=np.uint8)

# Alpha composite shirt onto white
alpha_3 = cv2.merge([alpha, alpha, alpha])
final_f = bgr.astype(np.float32) * alpha_3 + background.astype(np.float32) * (1.0 - alpha_3)
final = np.clip(final_f, 0, 255).astype(np.uint8)

cv2.imwrite(OUTPUT_IMAGE, final)
print(f"Simple shirt overlay saved to {OUTPUT_IMAGE}")
