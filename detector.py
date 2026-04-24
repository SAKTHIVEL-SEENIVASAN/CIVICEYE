import cv2
import numpy as np
import math
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage

# ==============================
# EMAIL CONFIG
# ==============================

SENDER_EMAIL = "s.m.sakthivelofficial@gmail.com"
APP_PASSWORD = "hbdqbsmucseplcve"
RECEIVER_EMAIL = "sakthinaidu@gmail.com"

# ==============================
# MODEL
# ==============================

MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

# ==============================
# COST MODEL
# ==============================

BASE_COST = 5000
MANPOWER_FACTOR = 6

# ==============================
# EMAIL FUNCTION
# ==============================

def send_email(potholes, severity, priority, cost, manpower, lat, lon, image_path):

    try:
        email_body = f"""
Respected Sir/Madam,

CivicEye system detected a pothole.

Detection Summary
-----------------
Potholes Detected : {potholes}
Severity Score    : {severity:.2f}
Priority Level    : {priority}

Estimated Repair
----------------
Approx Cost       : ₹{cost}
Manpower Required : {manpower}

Location
--------
Latitude  : {lat}
Longitude : {lon}

Department : Roads Department

Regards
CivicEye AI System
"""

        msg = EmailMessage()
        msg["Subject"] = "[CivicEye Alert] Pothole Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        msg.set_content(email_body)

        with open(image_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="jpeg",
                filename="evidence.jpg"
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)

        print("✅ Email sent")

    except Exception as e:
        print("❌ Email failed:", e)


# ==============================
# IMAGE DETECTION
# ==============================

def detect_image(path, lat="NA", lon="NA"):

    frame = cv2.imread(path)

    if frame is None:
        return {"error": "Invalid image"}

    h, w, _ = frame.shape
    image_area = h * w

    results = model.predict(source=frame, imgsz=640, conf=0.3)[0]

    total_area = 0
    potholes = 0

    # ==============================
    # Pothole Detection Logic
    # ==============================

    if results.masks is not None:
        for mask in results.masks.data.cpu().numpy():

            binary = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                potholes += 1
                total_area += cv2.contourArea(contours[0])

    # ==============================
    # Severity Calculation
    # ==============================

    severity = min((total_area / image_area) * 5, 1.0)

    # ==============================
    # Priority
    # ==============================

    if severity > 0.6:
        priority = "HIGH"
    elif severity > 0.3:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # ==============================
    # Cost & Manpower
    # ==============================

    cost = int(BASE_COST * (1 + severity))
    manpower = max(1, math.ceil(MANPOWER_FACTOR * severity))

    # ==============================
    # Save Output Image
    # ==============================

    annotated = results.plot(boxes=False)
    output_path = "static/result.jpg"
    cv2.imwrite(output_path, annotated)

    # ==============================
    # SEND EMAIL
    # ==============================

    if potholes > 0:
        send_email(potholes, severity, priority, cost, manpower, lat, lon, output_path)

    # ==============================
    # RETURN RESULT
    # ==============================

    return {
        "potholes": potholes,
        "severity": round(severity, 2),
        "priority": priority,
        "cost": cost,
        "manpower": manpower,
        "lat": lat,
        "lon": lon,
        "image": output_path
    }


# ==============================
# VIDEO DETECTION
# ==============================

def detect_video(path, lat="NA", lon="NA"):

    cap = cv2.VideoCapture(path)

    potholes = 0
    total_area = 0
    frame_count = 0
    image_area = 1
    saved_frame = None

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # process every 30th frame
        if frame_count % 30 != 0:
            continue

        h, w, _ = frame.shape
        image_area = h * w

        results = model.predict(source=frame, imgsz=640, conf=0.3)[0]

        if results.masks is not None:
            for mask in results.masks.data.cpu().numpy():

                binary = (mask > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    potholes += 1
                    total_area += cv2.contourArea(contours[0])
                    saved_frame = results.plot(boxes=False)

    cap.release()

    severity = min((total_area / image_area) * 5, 1.0)

    if severity > 0.6:
        priority = "HIGH"
    elif severity > 0.3:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    cost = int(BASE_COST * (1 + severity))
    manpower = max(1, math.ceil(MANPOWER_FACTOR * severity))

    output_path = "static/result.jpg"

    if saved_frame is not None:
        cv2.imwrite(output_path, saved_frame)

    if potholes > 0 and saved_frame is not None:
        send_email(potholes, severity, priority, cost, manpower, lat, lon, output_path)

    return {
        "potholes": potholes,
        "severity": round(severity, 2),
        "priority": priority,
        "cost": cost,
        "manpower": manpower,
        "lat": lat,
        "lon": lon,
        "image": output_path
    }