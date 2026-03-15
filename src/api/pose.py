import cv2
import json
import os
import mediapipe as mp
import numpy as np

print(" Pose detection started")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

INPUT_IMAGE = os.path.join(BASE_DIR, "data", "user.jpg")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "results", "user_pose.jpg")
OUTPUT_JSON = os.path.join(BASE_DIR, "results", "user_pose_keypoints.json")

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

print(" Input image:", INPUT_IMAGE)

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError(" User image not found")

h, w, _ = img.shape
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

results = pose.process(img_rgb)

if not results.pose_landmarks:
    raise RuntimeError(" No human pose detected")

# Save the segmentation mask if available
if results.segmentation_mask is not None:
    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
    # Resize mask to original image dimensions
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_path = os.path.join(BASE_DIR, "results", "user_mask.png")
    cv2.imwrite(mask_path, mask)
    print(" Body mask saved:", mask_path)

landmarks = results.pose_landmarks.landmark

def get_point(name):
    lm = landmarks[getattr(mp_pose.PoseLandmark, name).value]
    return {
        "x": int(lm.x * w),
        "y": int(lm.y * h)
    }

keypoints = {
    "nose": get_point("NOSE"),

    "left_shoulder": get_point("LEFT_SHOULDER"),
    "right_shoulder": get_point("RIGHT_SHOULDER"),

    "left_elbow": get_point("LEFT_ELBOW"),
    "right_elbow": get_point("RIGHT_ELBOW"),

    "left_hip": get_point("LEFT_HIP"),
    "right_hip": get_point("RIGHT_HIP")
}

keypoints["neck"] = {
    "x": int((keypoints["left_shoulder"]["x"] + keypoints["right_shoulder"]["x"]) / 2),
    "y": int((keypoints["left_shoulder"]["y"] + keypoints["right_shoulder"]["y"]) / 2)
}


keypoints["torso_center"] = {
    "x": int((keypoints["left_hip"]["x"] + keypoints["right_hip"]["x"]) / 2),
    "y": int((keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"]) / 2)
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(keypoints, f, indent=2)

print(" Keypoints saved:", OUTPUT_JSON)

debug_img = img.copy()

for name, pt in keypoints.items():
    cv2.circle(debug_img, (pt["x"], pt["y"]), 6, (0, 0, 255), -1)
    cv2.putText(
        debug_img,
        name,
        (pt["x"] + 5, pt["y"] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )

mp_draw.draw_landmarks(
    debug_img,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS
)

cv2.imwrite(OUTPUT_IMAGE, debug_img)

print(" Annotated image saved:", OUTPUT_IMAGE)
print(" Pose detection finished successfully")
