import cv2
import os

# Absolute paths
USER_IMAGE = r"c:\Users\livin\Desktop\v-fit\data\user.jpg"
SHIRT_PNG = r"c:\Users\livin\Desktop\v-fit\data\shirt.png"

user = cv2.imread(USER_IMAGE)
shirt = cv2.imread(SHIRT_PNG)

if user is None:
    print(f"Failed to load user: {USER_IMAGE}")
else:
    print(f"user.jpg shape: {user.shape}")

if shirt is None:
    print(f"Failed to load shirt: {SHIRT_PNG}")
else:
    print(f"shirt.png shape: {shirt.shape}")
