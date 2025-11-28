# src/api/pose.py
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def get_keypoints(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(rgb)
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    def to_px(idx):
        return int(lm[idx].x * w), int(lm[idx].y * h)
    left_shoulder = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = to_px(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    nose = to_px(mp_pose.PoseLandmark.NOSE.value)  # approximate neck
    return {"left_shoulder": left_shoulder, "right_shoulder": right_shoulder, "neck": nose}

if __name__ == "__main__":
    # simple test: place a sample image at data/user.jpg
    path = "data/user.jpg"
    try:
        kp = get_keypoints(path)
        print("Keypoints:", kp)
    except Exception as e:
        print("Error:", e)
