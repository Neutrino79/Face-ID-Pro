import face_recognition
import cv2


def detect_and_encode_face(image):
    # Resize image to 1/4 size for faster face detection
    small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

    # Find all the faces in the image using the HOG-based model
    face_locations = face_recognition.face_locations(rgb_small_image, model="hog")

    if face_locations:
        # Compute the face encoding for the face
        face_encoding = face_recognition.face_encodings(rgb_small_image, face_locations)[0]
        return True, face_encoding
    else:
        return False, None


def compare_faces(known_face_encodings, face_encoding):
    return face_recognition.compare_faces(known_face_encodings, face_encoding)
