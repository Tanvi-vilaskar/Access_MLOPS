import os
import time
import cv2
import av
from streamlit_webrtc import VideoTransformerBase
from config import VIDEO_DIR

# Ensure directory exists
os.makedirs(VIDEO_DIR, exist_ok=True)

# Load Haar Cascade
MODEL_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Face detection model not found at {MODEL_PATH}")

detector = cv2.CascadeClassifier(MODEL_PATH)


class FaceDetector(VideoTransformerBase):
    """Real-time face detection for MFA using webcam/streamlit-webrtc."""

    def __init__(self):
        self.face_detected = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            self.face_detected = True
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(
                img,
                "No face detected",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def capture_video(username, duration=5):
    """
    Capture a short video of the user for face registration.
    """
    file_path = os.path.join(VIDEO_DIR, f"user_{username}.avi")

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except PermissionError:
            print(f"Cannot delete {file_path}, file in use")
            return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam")
        return None

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    start_time = time.time()
    face_detected = False

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face_detected = True

        out.write(frame)

        if time.time() - start_time > duration:
            break

    cap.release()
    out.release()

    if face_detected:
        return file_path

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except PermissionError:
            print(f"Cannot delete {file_path}, file in use")

    return None


def verify_video(user_id):
    """Check if registered video exists."""
    file_path = os.path.join(VIDEO_DIR, f"user_{user_id}.avi")
    return os.path.exists(file_path)
