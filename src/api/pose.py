import cv2
import mediapipe as mp
import json
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_keypoints(image_path):
    """
    Returns dict with pixel coords for left_shoulder, right_shoulder, neck (approx via nose).
    Returns None if no pose detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    def to_px(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    left_shoulder = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = to_px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    neck = to_px(mp_pose.PoseLandmark.NOSE.value)

    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "neck": neck
    }

def draw_landmarks_and_save(image_path, out_image_path):
    """
    Debug helper: draws pose landmarks, highlights shoulders & neck,
    saves annotated image and a keypoints JSON file.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return None
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res = pose.process(rgb)

    if not res.pose_landmarks:
        print("No pose detected.")
        return None

    annotated = img.copy()
    mp_drawing.draw_landmarks(annotated, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    lm = res.pose_landmarks.landmark
    def to_px(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)

    ls = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    rs = to_px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    neck = to_px(mp_pose.PoseLandmark.NOSE.value)

    cv2.circle(annotated, ls, 8, (0,255,0), -1)
    cv2.circle(annotated, rs, 8, (0,0,255), -1)
    cv2.circle(annotated, neck, 6, (255,0,0), -1)

    os.makedirs(os.path.dirname(out_image_path) or '.', exist_ok=True)
    cv2.imwrite(out_image_path, annotated)

    keypoints = {
        "left_shoulder": ls,
        "right_shoulder": rs,
        "neck": neck
    }
    json_path = os.path.splitext(out_image_path)[0] + "_keypoints.json"
    with open(json_path, "w") as f:
        json.dump(keypoints, f)

    print("Saved:", out_image_path)
    print("Saved:", json_path)
    return keypoints

if __name__ == "__main__":
    # debug-run (used when running the file locally)
    inp = "data/user.jpg"
    out = "results/user_pose.jpg"
    kp = draw_landmarks_and_save(inp, out)
    print("Keypoints:", kp)
