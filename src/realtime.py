import os
import sys
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BEST_MODEL_PATH, CLASS_NAMES, DISPOSAL_RECOMMENDATIONS,
    IMG_SIZE, MEAN, STD,
    CONFIDENCE_THRESHOLD, HIGH_CONF_COLOR, LOW_CONF_COLOR,
    WARNING_COLOR, CAMERA_INDEX, ENABLE_VOICE, NUM_CLASSES
)
from model import load_model

voice_engine = None
if ENABLE_VOICE:
    try:
        import pyttsx3
        voice_engine = pyttsx3.init()
        voice_engine.setProperty("rate", 150)
        print("[Voice] pyttsx3 voice engine initialized.")
    except ImportError:
        print("[Voice] pyttsx3 not installed. Voice disabled.")

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    tensor    = inference_transform(pil_img).unsqueeze(0)  
    return tensor


def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)
        conf, idx = probs.max(dim=1)
    class_name = CLASS_NAMES[idx.item()]
    confidence = conf.item()
    return class_name, confidence


def speak(text):
    if voice_engine:
        voice_engine.say(text)
        voice_engine.runAndWait()


def draw_overlay(frame, class_name, confidence, recommendation, is_warning):
    h, w = frame.shape[:2]

    if is_warning:
        text_color = WARNING_COLOR
        status_text = "  LOW CONFIDENCE — UNSURE"
    else:
        text_color = HIGH_CONF_COLOR
        status_text = "  HIGH CONFIDENCE"

    overlay = frame.copy()
    panel_x1, panel_y1 = 10, 10
    panel_x2, panel_y2 = w - 10, 185
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "SMART WASTE SEGREGATION",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Class: {class_name.upper()}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                text_color, 2, cv2.LINE_AA)

    bar_x, bar_y, bar_h = 20, 95, 18
    bar_max_w = w - 40
    bar_fill  = int(bar_max_w * confidence)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h),
                  text_color, -1)
    cv2.putText(frame, f"{confidence*100:.1f}%",
                (bar_x + bar_max_w + 5, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(frame, recommendation,
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (220, 220, 100), 1, cv2.LINE_AA)

    cv2.putText(frame, status_text,
                (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                text_color, 1, cv2.LINE_AA)

    cv2.putText(frame, "Press 'q' to quit | 's' to save screenshot",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (150, 150, 150), 1, cv2.LINE_AA)

    return frame


def run_realtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RealTime] Using device: {device}")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[RealTime] ERROR: No model at {BEST_MODEL_PATH}")
        print("  → Run python src/train.py first.")
        return

    print("[RealTime] Loading model...")
    model = load_model(BEST_MODEL_PATH, num_classes=NUM_CLASSES, device=device)
    print("[RealTime] Model ready.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[RealTime] ERROR: Cannot open webcam (index {CAMERA_INDEX})")
        return

    print("[RealTime] Webcam opened. Starting real-time inference...")
    print("  Press 'q' to quit | 's' to save a screenshot")

    last_prediction = ""
    last_speak_time = 0
    speak_interval  = 3.0  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[RealTime] Failed to read frame. Exiting.")
            break

        tensor     = preprocess_frame(frame)
        class_name, confidence = predict(model, tensor, device)
        is_warning = confidence < CONFIDENCE_THRESHOLD
        recommendation = DISPOSAL_RECOMMENDATIONS.get(class_name, "Unknown bin")

        now = time.time()
        if (ENABLE_VOICE
                and not is_warning
                and class_name != last_prediction
                and now - last_speak_time > speak_interval):
            speak(f"This is {class_name}. {recommendation}.")
            last_prediction = class_name
            last_speak_time = now

        frame = draw_overlay(frame, class_name, confidence,
                             recommendation, is_warning)

        cv2.imshow("Smart Waste Segregation Assistant", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[RealTime] Quit key pressed. Exiting.")
            break
        elif key == ord("s"):
            screenshot_path = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"[RealTime] Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[RealTime] Done.")


if __name__ == "__main__":
    run_realtime()