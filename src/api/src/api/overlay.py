# src/api/overlay.py
import cv2
import numpy as np
import math

def apply_cloth(user_bgr, cloth_rgba, keypoints, person_mask=None):
    """
    user_bgr: numpy array BGR
    cloth_rgba: numpy array (H,W,4)
    keypoints: dict with 'left_shoulder','right_shoulder','neck'
    Returns: result_bgr image (numpy)
    """
    ls = keypoints['left_shoulder']
    rs = keypoints['right_shoulder']

    dx = rs[0] - ls[0]
    dy = rs[1] - ls[1]
    shoulder_w = int(math.hypot(dx, dy))
    if shoulder_w <= 0:
        return user_bgr

    scale = 1.2
    h_c, w_c = cloth_rgba.shape[:2]
    new_w = max(10, int(shoulder_w * scale))
    new_h = max(10, int(h_c * (new_w / w_c)))
    cloth_resized = cv2.resize(cloth_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    neck = keypoints['neck']
    x_center = int((ls[0] + rs[0]) / 2)
    x0 = x_center - new_w // 2
    y0 = max(0, neck[1] - int(new_h * 0.2))  # position slightly above neck

    result = user_bgr.copy()
    H, W = result.shape[:2]

    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(W, x0 + new_w)
    y2 = min(H, y0 + new_h)

    cloth_x1 = x1 - x0
    cloth_y1 = y1 - y0
    cloth_x2 = cloth_x1 + (x2 - x1)
    cloth_y2 = cloth_y1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return result

    cloth_roi = cloth_resized[cloth_y1:cloth_y2, cloth_x1:cloth_x2]
    if cloth_roi.shape[0] == 0 or cloth_roi.shape[1] == 0:
        return result

    alpha = cloth_roi[:, :, 3] / 255.0
    for c in range(3):
        result[y1:y2, x1:x2, c] = (alpha * cloth_roi[:, :, c] +
                                   (1 - alpha) * result[y1:y2, x1:x2, c]).astype(np.uint8)
    return result

if __name__ == "__main__":
    # small demo using files in data/
    import os
    from api.pose import get_keypoints
    from api.cloth import prep_cloth

    user_path = "data/user.jpg"
    cloth_path = "data/cloth.png"
    user = cv2.imread(user_path)
    if user is None:
        print("Put a sample user image at data/user.jpg")
    else:
        kp = get_keypoints(user_path)
        if kp is None:
            print("Pose not detected in user image")
        else:
            cloth_rgba, mask = prep_cloth(cloth_path)
            out = apply_cloth(user, cloth_rgba, kp)
            cv2.imwrite("data/result_demo.jpg", out)
            print("Saved data/result_demo.jpg")
