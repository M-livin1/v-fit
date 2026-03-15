"""
overlay.py – Virtual Try-On: THE SUPREME "WARPED FIT" (Advanced)

Features:
1. Affine Warping: Uses 3-point transform to match torso pose exactly.
2. Dynamic Anchors: Maps shoulder and hip keypoints to shirt landmarks.
3. Clean Masking: Uses user_mask.png for body-locked placement.
4. Seamless Blending: Improved neck/skin transitions.
"""

import cv2
import json
import numpy as np
import os

print("Supreme Warped Fit Overlay (Advanced) started")

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
USER_IMAGE    = os.path.join(BASE_DIR, "data", "user.jpg")
SHIRT_CLEAN   = os.path.join(BASE_DIR, "results", "cloth_clean.png")
USER_MASK     = os.path.join(BASE_DIR, "results", "user_mask.png")
KEYPOINT_FILE = os.path.join(BASE_DIR, "results", "user_pose_keypoints.json")
OUTPUT_IMAGE  = os.path.join(BASE_DIR, "results", "final_overlay.jpg")

# 1. Load Images & Data
user = cv2.imread(USER_IMAGE)
if user is None: raise FileNotFoundError(f"user.jpg missing")
h_img, w_img = user.shape[:2]

shirt_rgba = cv2.imread(SHIRT_CLEAN, cv2.IMREAD_UNCHANGED)
if shirt_rgba is None: raise FileNotFoundError("cloth_clean.png missing")

user_mask = cv2.imread(USER_MASK, cv2.IMREAD_GRAYSCALE)
if user_mask is None: user_mask = np.zeros((h_img, w_img), dtype=np.uint8)

with open(KEYPOINT_FILE) as f:
    kp = json.load(f)

def pt(name): return np.array([kp[name]["x"], kp[name]["y"]], dtype=np.float32)

# Keypoints
ls, rs = pt("left_shoulder"), pt("right_shoulder")
lh, rh = pt("left_hip"), pt("right_hip")
nk, ns = pt("neck"), pt("nose")
tc = pt("torso_center")

# 2. Extract Shirt Content
alpha_s = shirt_rgba[:, :, 3]
nz = cv2.findNonZero(alpha_s)
bx, by, bw, bh = cv2.boundingRect(nz)
shirt_c = shirt_rgba[by:by+bh, bx:bx+bw]
sh, sw = shirt_c.shape[:2]

# 3. Define Affine Transform (3-Point Mapping)
# Source points on shirt (landmarks in relative coordinates)
# We assume the shirt is centered. 
# Shoulders: roughly 20/80% width, 25% depth. Hip center: center, 95% depth.
src_pts = np.array([
    [0.25 * sw, 0.25 * sh], # Left shoulder
    [0.75 * sw, 0.25 * sh], # Right shoulder
    [0.50 * sw, 0.95 * sh]  # Torso center (bottom)
], dtype=np.float32)

# Destination points on User (match the pose)
# We can adjust these slightly to ensure full coverage
span = np.linalg.norm(ls - rs)
ls_dst = ls + (ls - rs) * 0.15 # Expand slightly for better shoulder coverage
rs_dst = rs + (rs - ls) * 0.15
tc_dst = tc + np.array([0, 20], dtype=np.float32) # Push bottom down slightly

dst_pts = np.array([ls_dst, rs_dst, tc_dst], dtype=np.float32)

# Perform Warping
M = cv2.getAffineTransform(src_pts, dst_pts)
warped_shirt = cv2.warpAffine(shirt_c, M, (w_img, h_img), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

"""
STEP 4: Aggressively remove the ORIGINAL shirt from the user image,
but keep the rest of the body (face, arms, background).
This ensures only the INPUT shirt (cloth_clean.png) is visible.
"""

# Start from the pose‑based torso polygon (slightly expanded)
torso_mask = np.zeros((h_img, w_img), dtype=np.uint8)
poly = np.array([
    ls_dst + [0, -60],   # a bit above shoulders
    rs_dst + [0, -60],
    rh + [60, 80],       # extend below hips
    lh + [-60, 80]
], np.int32)
cv2.fillPoly(torso_mask, [poly], 255)

# Protect the face region
face_y = int(ns[1] + (nk[1] - ns[1]) * 0.5)
torso_mask[:face_y, :] = 0

# Use the body segmentation mask (if available) to limit removal
if user_mask is not None:
    # Intersect torso area with detected body to avoid wiping background
    erasure_mask = cv2.bitwise_and(torso_mask, user_mask)
else:
    erasure_mask = torso_mask

# Feather edges so transition into the new shirt is smooth
erasure_mask = cv2.GaussianBlur(erasure_mask, (21, 21), 0)
erasure_alpha = (erasure_mask.astype(np.float32) / 255.0)[..., None]

# Remove original shirt by fading torso pixels to black
user_cleaned = user.astype(np.float32)
user_cleaned = user_cleaned * (1.0 - erasure_alpha)
user_cleaned = np.clip(user_cleaned, 0, 255).astype(np.uint8)

"""
STEP 5: Composite & blend the WARPED INPUT SHIRT on top of the cleaned user.
Only the uploaded shirt (cloth_clean.png) is visible in the torso area.
"""
out_bgr = warped_shirt[:, :, :3]
out_alpha = warped_shirt[:, :, 3].astype(np.float32) / 255.0

# Smooth neck transition (gradient at the top of shirt)
for y in range(face_y, min(face_y + 15, h_img)):
    weight = (y - face_y) / 15.0
    out_alpha[y, :] *= weight

out_alpha[:face_y, :] = 0

a3 = cv2.merge([out_alpha] * 3)
final_f = out_bgr.astype(np.float32) * a3 + user_cleaned.astype(np.float32) * (1.0 - a3)
final = np.clip(final_f, 0, 255).astype(np.uint8)

cv2.imwrite(OUTPUT_IMAGE, final)
print(f"Supreme Warped Fit saved to {OUTPUT_IMAGE}")
