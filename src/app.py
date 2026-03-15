import os
import sys
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")

POSE_SCRIPT = os.path.join(PROJECT_ROOT, "src", "api", "pose.py")
PREPROCESS_SCRIPT = os.path.join(PROJECT_ROOT, "src", "api", "preprocessing.py")
OVERLAY_SCRIPT = os.path.join(PROJECT_ROOT, "src", "api", "overlay.py")

FINAL_IMAGE = os.path.join(RESULTS_DIR, "final_overlay.jpg")
USER_NO_SHIRT = os.path.join(RESULTS_DIR, "user_no_shirt.jpg")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        
        user_file = request.files.get("user_image")
        cloth_file = request.files.get("cloth_image")

        if not user_file or not cloth_file:
            return " Please upload both images."

        user_path = os.path.join(DATA_DIR, "user.jpg")
        cloth_path = os.path.join(DATA_DIR, "shirt.png")

        user_file.save(user_path)
        cloth_file.save(cloth_path)

        subprocess.run([sys.executable, PREPROCESS_SCRIPT], check=True, cwd=PROJECT_ROOT)
        subprocess.run([sys.executable, POSE_SCRIPT], check=True, cwd=PROJECT_ROOT)
        subprocess.run([sys.executable, OVERLAY_SCRIPT], check=True, cwd=PROJECT_ROOT)

       
        if not os.path.exists(FINAL_IMAGE):
            return " Final image not created. Check overlay.py"

        return redirect(url_for("result"))

    except subprocess.CalledProcessError as e:
        return f" Error running AI modules:<br>{e}"

@app.route("/result")
def result():
    import time
    if not os.path.exists(FINAL_IMAGE):
        return " No result image found."

    return render_template("result.html", ts=int(time.time()))

@app.route("/final")
def final_image():
    if os.path.exists(FINAL_IMAGE):
        return send_file(FINAL_IMAGE, mimetype="image/jpeg")
    return " Image not found."


@app.route("/no_shirt")
def no_shirt_image():
    if os.path.exists(USER_NO_SHIRT):
        return send_file(USER_NO_SHIRT, mimetype="image/jpeg")
    return " Image not found."



if __name__ == "__main__":
    app.run(debug=True)
