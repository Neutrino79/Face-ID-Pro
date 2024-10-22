import face_recognition
import cv2
import numpy as np
from PIL import Image


def detect_and_encode_face(image, use_cnn=True):
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
    Aligns the face in the image using facial landmarks.
    """
    face_landmarks_list = face_recognition.face_landmarks(image)

    if face_landmarks_list:
        landmarks = face_landmarks_list[0]
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)

        # Calculate angle to align eyes horizontally
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Get the center of the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        # Rotate the image to align the eyes
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return aligned_image

    return image  # Return original image if no face landmarks found


def compare_faces(known_face_encodings, face_encoding, tolerance=0.6):
    """
    Compares a face encoding against a list of known face encodings.
    Returns a list of boolean values indicating which known faces match the given face.
    """
    return face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)


def face_distance(known_face_encodings, face_encoding):
    """
    Computes the face distance between the target face encoding and a list of known face encodings.
    """
    return face_recognition.face_distance(known_face_encodings, face_encoding)
