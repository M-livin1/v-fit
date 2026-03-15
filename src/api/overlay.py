"""
overlay.py – Virtual Try-On: THE SUPREME "CORRECT FIT" (Fixed)

Features:
1. Professional foreground extraction (GrabCut).
2. Multi-stage erasure (Torso + Collar) for a clean base.
3. Anatomical scaling (2.75x span) for full coverage.
4. Seamless neck blending and alignment.
"""

import cv2
import json
import numpy as np
import os

print("Supreme Correct Fit Overlay started")

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USER_IMAGE    = os.path.join(BASE_DIR, "data", "user.jpg")
SHIRT_PNG     = os.path.join(BASE_DIR, "data", "shirt.png")
KEYPOINT_FILE = os.path.join(BASE_DIR, "results", "user_pose_keypoints.json")
OUTPUT_IMAGE  = os.path.join(BASE_DIR, "results", "final_overlay.jpg")

user = cv2.imread(USER_IMAGE)
h_img, w_img = user.shape[:2]

shirt = cv2.imread(SHIRT_PNG)
if shirt is None: raise FileNotFoundError("shirt.png missing")

# ── Step 1: Alpha (GrabCut) ──────────────────────────────────────────────────
mask = np.zeros(shirt.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(shirt, mask, (10, 10, shirt.shape[1]-20, shirt.shape[0]-20), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
alpha = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
shirt_rgba = cv2.merge([shirt, alpha])
nz = cv2.findNonZero(alpha)
bx, by, bw, bh = cv2.boundingRect(nz)
shirt_c = shirt_rgba[by:by+bh, bx:bx+bw]
sh, sw = shirt_c.shape[:2]

with open(KEYPOINT_FILE) as f:
    kp = json.load(f)
def get_kp(name): return (kp[name]["x"], kp[name]["y"])
ls, rs, neck, nose = get_kp("left_shoulder"), get_kp("right_shoulder"), get_kp("neck"), get_kp("nose")
lh, rh = get_kp("left_hip"), get_kp("right_hip")
span = abs(ls[0] - rs[0])

# ── Step 2: Supreme Erasure ──────────────────────────────────────────────────
# Torso Poly
torso_poly = np.array([
    [rs[0]-50, rs[1]-40], # Very high shoulders
    [ls[0]+50, ls[1]-40],
    [ls[0]+80, lh[1]+100],
    [rs[0]-80, rh[1]+100]
], np.int32)
# Neck Oval (to cover navy collar)
neck_h = neck[1] - nose[1]
neck_oval_center = (neck[0], neck[1] - int(neck_h * 0.1))
neck_oval_axes = (int(span * 0.45), int(neck_h * 0.5))

erasure_mask = np.zeros((h_img, w_img), dtype=np.uint8)
cv2.fillPoly(erasure_mask, [torso_poly], 255)
cv2.ellipse(erasure_mask, neck_oval_center, neck_oval_axes, 0, 0, 360, 255, -1)

# Face Protection
face_y = nose[1] + int((neck[1]-nose[1])*0.5)
erasure_mask[:face_y, :] = 0

user_cleaned = user.copy()
user_cleaned[erasure_mask == 255] = [255, 255, 255]

# ── Step 3: Precise Scaling & Positioning ────────────────────────────────────
# Slightly reduce overall width so the shirt hugs the shoulders better
target_w = int(span * 2.4)
scale = target_w / float(sw)
target_h = int(sh * scale)
resized = cv2.resize(shirt_c, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

center_x = (ls[0] + rs[0]) // 2
paste_x = center_x - (target_w // 2)

# Raise the shirt so the collar opening sits around the neck keypoint
# Using a larger negative offset pulls the shirt upward compared to before
paste_y = neck[1] - int(target_h * 0.18)

# ── Step 4: Composite ────────────────────────────────────────────────────────
canvas_bgr = np.zeros((h_img, w_img, 3), dtype=np.uint8)
canvas_alpha = np.zeros((h_img, w_img), dtype=np.float32)

y1, y2 = max(0, paste_y), min(h_img, paste_y + target_h)
x1, x2 = max(0, paste_x), min(w_img, paste_x + target_w)

if y2 > y1 and x2 > x1:
    sy1, sx1 = y1 - paste_y, x1 - paste_x
    sy2, sx2 = sy1 + (y2 - y1), sx1 + (x2 - x1)
    canvas_bgr[y1:y2, x1:x2] = resized[sy1:sy2, sx1:sx2, :3]
    canvas_alpha[y1:y2, x1:x2] = resized[sy1:sy2, sx1:sx2, 3].astype(np.float32) / 255.0

# Sharp face cutoff
canvas_alpha[:face_y, :] = 0
# Smooth top transition
for y in range(face_y, face_y + 10):
    if y < h_img:
        canvas_alpha[y, :] *= (y - face_y) / 10.0

a3 = cv2.merge([canvas_alpha]*3)
result_f = canvas_bgr.astype(np.float32) * a3 + user_cleaned.astype(np.float32) * (1.0 - a3)
result = np.clip(result_f, 0, 255).astype(np.uint8)

cv2.imwrite(OUTPUT_IMAGE, result)
print(f"Supreme Final Fit saved to {OUTPUT_IMAGE}")
