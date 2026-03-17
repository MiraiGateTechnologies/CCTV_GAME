"""
================================================
  WEB SERVER — Browser pe live video stream
  Flask MJPEG server — http://localhost:5000
================================================
"""

# ┌─────────────────────────────────────────────────────────────────────┐
# │  FILE   : web_server.py                                             │
# │  PURPOSE: Browser mein live video dikhana (MJPEG streaming)         │
# │           Flask server jo main.py / scheduler.py dono use karte     │
# │           Ye file STANDALONE hai — kisi aur file pe depend nahi     │
# ├─────────────────────────────────────────────────────────────────────┤
# │  CHANGE GUIDE — Agar kuch change karna ho:                          │
# │                                                                     │
# │  ► Default port change karna ho (5000 → kuch aur)                  │
# │    → start_server(host="0.0.0.0", port=5000) — port value badlo    │
# │      Ya run karte waqt: python main.py --web-port 8080              │
# │                                                                     │
# │  ► JPEG stream quality change karni ho                              │
# │    → generate_frames() ke andar cv2.imencode(".jpg", output_frame)  │
# │      Badal ke: cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) │
# │                                                                     │
# │  ► HTML page (browser UI) change karni ho                           │
# │    → templates/index.html file edit karo                            │
# │                                                                     │
# │  ► Naya API endpoint add karna ho (e.g., /stats, /count)           │
# │    → @app.route("/your_route") decorator se naya function banao     │
# └─────────────────────────────────────────────────────────────────────┘

import cv2
import threading
from flask import Flask, render_template, Response

app = Flask(__name__)

# Global variable to hold the latest frame
# We use a Lock to ensure thread-safety when updating/reading the frame
output_frame = None
lock = threading.Lock()

def update_frame(frame):
    """Updates the global frame with the latest processed frame."""
    global output_frame
    with lock:
        output_frame = frame.copy()

def generate_frames():
    """Generator function that yields MJPEG chunks."""
    while True:
        with lock:
            if output_frame is None:
                continue
            # Encode frame to JPEG
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        # Yield the output frame in the byte format required for MJPEG
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_server(port=5000, host="0.0.0.0"):
    """Starts the Flask server in a daemon thread."""
    t = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
    t.start()
    print(f"[WEB] Server started at http://localhost:{port}")
