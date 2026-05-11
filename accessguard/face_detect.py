import cv2
import dlib
import os

# Path to the pretrained CNN face detector
# Download if not available: http://dlib.net/files/mmod_human_face_detector.dat.bz2
# Extract .dat file and place it in the same directory
MODEL_PATH = "mmod_human_face_detector.dat"

# Load CNN face detector
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Download mmod_human_face_detector.dat from dlib website and place in script folder."
    )

cnn_face_detector = dlib.cnn_face_detection_model_v1(MODEL_PATH)

# Path to input image
image_path = "f6bd9232-856f-44dc-a95e-b240fcab899b.png"

# Load the image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Image not found or invalid path.")

# Convert to RGB (dlib expects RGB)
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
detections = cnn_face_detector(rgb_img, 1)

if len(detections) == 0:
    print("No face detected")
else:
    print(f"Detected {len(detections)} face(s).")

    # Draw bounding boxes
    for i, det in enumerate(detections):
        x1 = det.rect.left()
        y1 = det.rect.top()
        x2 = det.rect.right()
        y2 = det.rect.bottom()

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Face {i+1}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Show output
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
