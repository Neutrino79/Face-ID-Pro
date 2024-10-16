import face_recognition
import cv2
import numpy as np


def detect_and_encode_face(image, use_cnn=False):
    """
    Detects a face in the image and returns the face encoding.
    If `use_cnn` is True, it uses the CNN model for detection.
    """
    # Resize image to 1/4 size for faster face detection
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (OpenCV default) to RGB color
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    # Use CNN-based model for more accurate detection, otherwise use HOG
    model = "cnn" if use_cnn else "hog"
    face_locations = face_recognition.face_locations(rgb_small_image, model=model)

    if face_locations:
        # Compute the face encoding for the first face detected
        face_encoding = face_recognition.face_encodings(rgb_small_image, face_locations)[0]
        return True, face_encoding
    else:
        return False, None


def is_blurry(image, threshold=100.0):
    """
    Detect if an image is blurry using the Laplacian variance method.
    A lower variance indicates a blurrier image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var < threshold


def align_face(image):
    """
    Aligns the face in the image using facial landmarks. Helps improve recognition accuracy.
    """
    # Detect the face landmarks
    face_landmarks_list = face_recognition.face_landmarks(image)

    if face_landmarks_list:
        # Extract the chin and eye landmarks to calculate the face angle
        chin = face_landmarks_list[0]['chin']
        left_eye = face_landmarks_list[0]['left_eye']
        right_eye = face_landmarks_list[0]['right_eye']

        # Align based on the position of the eyes
        # Implement your own logic here to rotate or align the face
        # You can calculate the eye center and align the face horizontally
        pass

    # Return the aligned face (for now returning the same image)
    return image


def compare_faces(known_face_encodings, face_encoding):
    """
    Compares a face encoding against a list of known face encodings.
    """
    return face_recognition.compare_faces(known_face_encodings, face_encoding)
