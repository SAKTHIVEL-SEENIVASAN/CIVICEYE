from flask import Flask, render_template, request
import os
from detector import detect_image, detect_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    lat = request.form.get("lat", "NA")
    lon = request.form.get("lon", "NA")

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(path)

    ext = os.path.splitext(file.filename)[1].lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        result = detect_image(path, lat, lon)

    elif ext in [".mp4", ".avi", ".mov"]:
        result = detect_video(path, lat, lon)

    else:
        return "Unsupported file type"

    return render_template("result.html", data=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)