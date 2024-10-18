import logging
from django.shortcuts import render
from django.http import JsonResponse
from .models import FaceProfile
from face_app.utils import detect_and_encode_face, compare_faces, is_blurry, align_face, face_distance
import base64
import numpy as np
import cv2
import face_recognition
import math

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

THRESHOLD = 0.6
REQUIRED_SAMPLES = 5
MAX_ATTEMPTS = 3  # Maximum number of attempts for each pose

POSES = [
    {'instruction': 'Look straight at the camera', 'validation': 'validate_front_face'},
    {'instruction': 'Turn your head slightly to the left', 'validation': 'validate_left_face'},
    {'instruction': 'Turn your head slightly to the right', 'validation': 'validate_right_face'},
    {'instruction': 'Tilt your head up slightly', 'validation': 'validate_up_face'},
    {'instruction': 'Tilt your head down slightly', 'validation': 'validate_down_face'}
]


def home(request):
    return render(request, 'home.html')

def get_next_pose(request):
    sample_count = int(request.GET.get('sample_count', 0))
    attempt_count = int(request.GET.get('attempt_count', 0))

    if attempt_count >= MAX_ATTEMPTS:
        return JsonResponse({'success': False, 'error': 'Maximum attempts reached. Please start over.', 'reset': True})

    if sample_count < len(POSES):
        return JsonResponse({'instruction': POSES[sample_count]['instruction'], 'attempt_count': attempt_count})
    else:
        return JsonResponse({'complete': True})

def validate_face_angle(image, landmarks, validation_func):
    result = globals()[validation_func](landmarks)
    logger.debug(f"Validation result for {validation_func}: {result}")
    return result

def validate_front_face(landmarks):
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    eye_angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    logger.debug(f"Front face eye angle: {eye_angle}")
    return abs(eye_angle) < 15  # Increased tolerance

def calculate_face_ratio(landmarks):
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    nose_tip = landmarks['nose_tip'][0]
    left_ratio = (nose_tip[0] - left_eye[0]) / (right_eye[0] - left_eye[0])
    right_ratio = (right_eye[0] - nose_tip[0]) / (right_eye[0] - left_eye[0])
    logger.debug(f"Left eye: {left_eye}, Right eye: {right_eye}, Nose tip: {nose_tip}")
    logger.debug(f"Left ratio: {left_ratio}, Right ratio: {right_ratio}")
    return left_ratio, right_ratio

def validate_left_face(landmarks):
    left_ratio, right_ratio = calculate_face_ratio(landmarks)
    is_valid = left_ratio > 0.30 and right_ratio < 0.70  # Adjusted thresholds
    logger.debug(f"Left face validation result: {is_valid}")
    return is_valid

def validate_right_face(landmarks):
    left_ratio, right_ratio = calculate_face_ratio(landmarks)
    is_valid = left_ratio < 0.30 and right_ratio > 0.70  # Adjusted thresholds
    logger.debug(f"Right face validation result: {is_valid}")
    return is_valid

def validate_up_face(landmarks):
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    mouth = np.mean(landmarks['top_lip'], axis=0)
    eye_mouth_distance = np.linalg.norm(np.mean([left_eye, right_eye], axis=0) - mouth)
    logger.debug(f"Up face eye-mouth distance: {eye_mouth_distance}")
    return eye_mouth_distance < 55  # Increased threshold

def validate_down_face(landmarks):
    left_eye = np.mean(landmarks['left_eye'], axis=0)
    right_eye = np.mean(landmarks['right_eye'], axis=0)
    mouth = np.mean(landmarks['top_lip'], axis=0)
    eye_mouth_distance = np.linalg.norm(np.mean([left_eye, right_eye], axis=0) - mouth)
    logger.debug(f"Down face eye-mouth distance: {eye_mouth_distance}")
    return eye_mouth_distance > 50 and eye_mouth_distance < 62  # Adjusted range

def register_face(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        sample_count = int(request.POST.get('sample_count', 0))
        attempt_count = int(request.POST.get('attempt_count', 0))

        if not image_data:
            return JsonResponse({'success': False, 'error': 'Image is required.'})

        try:
            # Decode the base64 image
            format, imgstr = image_data.split(';base64,')
            image = base64.b64decode(imgstr)
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

            # Pre-check if the image is too blurry
            if is_blurry(image):
                return JsonResponse({'success': False, 'error': 'Image is too blurry. Please try again.', 'attempt_count': attempt_count + 1})

            # Detect face landmarks
            face_landmarks_list = face_recognition.face_landmarks(image)
            if not face_landmarks_list:
                return JsonResponse({'success': False, 'error': 'No face detected in the image. Please try again.', 'attempt_count': attempt_count + 1})

            # Validate face angle
            validation_result = validate_face_angle(image, face_landmarks_list[0], POSES[sample_count]['validation'])
            if not validation_result:
                instruction = POSES[sample_count]["instruction"].lower()
                error_message = f'Face not in correct position. Please {instruction}.'
                return JsonResponse({'success': False, 'error': error_message, 'attempt_count': attempt_count + 1})

            # Align the face for more accurate encoding
            image = align_face(image)

            # Detect and encode face using CNN model
            success, face_encoding = detect_and_encode_face(image, use_cnn=True)
            if success:
                # Store the encoding in the session
                request.session.setdefault('face_encodings', []).append(face_encoding.tolist())

                sample_count += 1
                if sample_count >= REQUIRED_SAMPLES:
                    return JsonResponse({'success': True, 'message': 'Face samples collected. Please enter your name.', 'complete': True})
                else:
                    return JsonResponse({'success': True, 'message': f'Sample {sample_count} of {REQUIRED_SAMPLES} captured.', 'complete': False, 'sample_count': sample_count, 'attempt_count': 0})
            else:
                return JsonResponse({'success': False, 'error': 'Failed to encode face. Please try again.', 'attempt_count': attempt_count + 1})

        except Exception as e:
            logger.exception("An error occurred during face registration")
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}', 'attempt_count': attempt_count + 1})

    return render(request, 'register_face.html')

def save_face_profile(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        face_encodings = request.session.get('face_encodings', [])

        if not name or not face_encodings:
            return JsonResponse({'success': False, 'error': 'Name and face encodings are required.'})

        try:
            face_profile = FaceProfile(name=name)
            face_profile.set_encodings(np.array(face_encodings))
            face_profile.save()

            # Clear the session data
            del request.session['face_encodings']

            return JsonResponse({'success': True, 'message': 'Face profile saved successfully.'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}'})

    return JsonResponse({'success': False, 'error': 'Invalid request method.'})

def test_face(request):
    if request.method == 'POST':
        try:
            image_data = request.POST.get('image')
            if not image_data:
                return JsonResponse({'success': False, 'error': 'No image data received.'})

            # Decode the base64 image
            format, imgstr = image_data.split(';base64,')
            image = base64.b64decode(imgstr)
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

            # Pre-check if the image is too blurry
            if is_blurry(image):
                return JsonResponse({'success': False, 'error': 'Image is too blurry for recognition.'})

            # Align the face before encoding
            image = align_face(image)

            # Detect and encode face using CNN model
            success, face_encoding = detect_and_encode_face(image, use_cnn=True)
            if not success:
                return JsonResponse({'success': False, 'error': 'No face detected in the image.'})

            # Fetch known faces from the database
            known_faces = FaceProfile.objects.all()
            if not known_faces:
                return JsonResponse({'success': False, 'error': 'No faces registered in the database.'})

            best_match = None
            best_match_distance = float('inf')

            for face in known_faces:
                known_encodings = face.get_encodings()
                distances = face_distance(known_encodings, face_encoding)
                min_distance = np.min(distances)

                if min_distance < best_match_distance:
                    best_match_distance = min_distance
                    best_match = face

            if best_match_distance <= THRESHOLD:
                confidence = (1 - best_match_distance) * 100
                return JsonResponse({'success': True, 'name': best_match.name, 'confidence': f"{confidence:.2f}%"})
            else:
                return JsonResponse({'success': True, 'name': None, 'message': 'No close match found.'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}'})

    return render(request, 'test_face.html')