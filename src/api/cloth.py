# src/api/cloth.py
import cv2
import numpy as np
from PIL import Image

def prep_cloth(cloth_path):
    """
    Returns cloth_rgba (H,W,4) and mask (H,W) where mask is 0-255.
    Works if cloth PNG with alpha, else attempts simple white-bg removal.
    """
    img = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(cloth_path)

    # If image has alpha channel already
    if img.shape[2] == 4:
        cloth_rgba = img
        mask = img[:, :, 3]
    else:
        # simple threshold to remove white background
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        b, g, r = cv2.split(img)
        cloth_rgba = cv2.merge([b, g, r, mask])
    return cloth_rgba, mask

if __name__ == "__main__":
    path = "data/cloth.png"
    try:
        cloth_rgba, mask = prep_cloth(path)
        cv2.imwrite("data/cloth_rgba.png", cloth_rgba)
        cv2.imwrite("data/cloth_mask.png", mask)
        print("Saved cloth_rgba and cloth_mask in data/")
    except Exception as e:
        print("Error:", e)
