from flask import Flask, render_template, request
import cv2
import os
from api.pose import get_keypoints
from api.cloth import prep_cloth
from api.overlay import apply_cloth

app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = "data"
RESULT_PATH = os.path.join("static", "result.jpg")
os.makedirs("data", exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/tryon', methods=['POST'])
def tryon():
    user_file = request.files.get('user')
    cloth_file = request.files.get('cloth')
    if not user_file or not cloth_file:
        return "Please upload both user and cloth images", 400
    user_path = os.path.join(UPLOAD_FOLDER, "user.jpg")
    cloth_path = os.path.join(UPLOAD_FOLDER, "cloth.png")
    user_file.save(user_path)
    cloth_file.save(cloth_path)

    kp = get_keypoints(user_path)
    if kp is None:
        return "No person detected in the image. Use front-facing clear photo.", 400

    cloth_rgba, mask = prep_cloth(cloth_path)
    user_img = cv2.imread(user_path)
    result = apply_cloth(user_img, cloth_rgba, kp)
    cv2.imwrite(RESULT_PATH, result)
    return render_template("result.html", result_url="/static/result.jpg")

if __name__ == "__main__":
    app.run(debug=True)

