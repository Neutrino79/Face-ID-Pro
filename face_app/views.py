from django.shortcuts import render
from django.http import JsonResponse
from .models import FaceProfile
from face_app.utils import detect_and_encode_face, compare_faces, is_blurry, align_face, face_distance
import base64
import numpy as np
import cv2
import face_recognition

THRESHOLD = 0.6
REQUIRED_SAMPLES = 5

def home(request):
    return render(request, 'home.html')

def register_face(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        image_data = request.POST.get('image')
        sample_count = int(request.POST.get('sample_count', 0))

        if not name or not image_data:
            return JsonResponse({'success': False, 'error': 'Name and image are required.'})

        try:
            # Decode the base64 image
            format, imgstr = image_data.split(';base64,')
            image = base64.b64decode(imgstr)
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

            # Pre-check if the image is too blurry
            if is_blurry(image):
                return JsonResponse({'success': False, 'error': 'Image is too blurry. Please try again.'})

            # Align the face for more accurate encoding
            image = align_face(image)

            # Detect and encode face using CNN model
            success, face_encoding = detect_and_encode_face(image, use_cnn=True)
            if success:
                if sample_count == 0:
                    # First sample, create new FaceProfile
                    face_profile = FaceProfile(name=name)
                    face_profile.set_encodings([face_encoding])
                    face_profile.save()
                else:
                    # Update existing FaceProfile
                    face_profile = FaceProfile.objects.get(name=name)
                    current_encodings = face_profile.get_encodings()
                    updated_encodings = np.vstack((current_encodings, [face_encoding]))
                    face_profile.set_encodings(updated_encodings)
                    face_profile.save()

                sample_count += 1
                if sample_count >= REQUIRED_SAMPLES:
                    return JsonResponse({'success': True, 'message': 'Face registration complete.', 'complete': True})
                else:
                    return JsonResponse({'success': True, 'message': f'Sample {sample_count} of {REQUIRED_SAMPLES} captured.', 'complete': False, 'sample_count': sample_count})
            else:
                return JsonResponse({'success': False, 'error': 'No face detected in the image. Please try again.'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}'})

    return render(request, 'register_face.html')


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