import cv2
import numpy as np
import streamlit as st


def check_face_in_image(image_data):
    """
    Checks if a face is present in the provided image data (bytes from st.camera_input).

    Args:
        image_data: File buffer object containing the image data.

    Returns:
        True if a face is detected, False otherwise.
    """
    if image_data is None:
        return False

    # Convert the file buffer to a numpy array for OpenCV
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode image data.")
        return False

    # Load the pre-trained cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Optionally draw a rectangle on the detected face for confirmation
        for x, y, w, h in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image with the detection box (convert BGR to RGB for Streamlit)
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Face Detected (MFA Verified)",
            use_container_width=True,
        )
        return True
    else:
        # Display the image without detection
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="No Face Detected",
            use_container_width=True,
        )
        return False


# The previous start_mfa_verification() is removed as it caused blocking issues.
