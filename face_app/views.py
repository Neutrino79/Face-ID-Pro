from django.shortcuts import render
from django.http import JsonResponse
from .models import FaceProfile
from .forms import FaceProfileForm
from face_app.utils import detect_and_encode_face, compare_faces, is_blurry
import base64
import numpy as np
import cv2
import face_recognition

THRESHOLD = 0.55  # Fine-tuned threshold for matching accuracy

def home(request):
    return render(request, 'home.html')

def register_face(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        image_data = request.POST.get('image')

        if not name or not image_data:
            return JsonResponse({'success': False, 'error': 'Name and image are required.'})

        try:
            # Decode the base64 image
            format, imgstr = image_data.split(';base64,')
            image = base64.b64decode(imgstr)
            image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

            # Pre-check if the image is too blurry
            #if is_blurry(image):
            #    return JsonResponse({'success': False, 'error': 'Image is too blurry for face registration.'})

            success, face_encoding = detect_and_encode_face(image, use_cnn=True)
            if success:
                face_profile = FaceProfile(name=name)
                face_profile.set_encoding(face_encoding)
                face_profile.save()
                return JsonResponse({'success': True, 'message': 'Face registered successfully.'})
            else:
                return JsonResponse({'success': False, 'error': 'No face detected in the image.'})
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

            ## Pre-check if the image is too blurry
            #if is_blurry(image):
            #    return JsonResponse({'success': False, 'error': 'Image is too blurry for recognition.'})

            success, face_encoding = detect_and_encode_face(image, use_cnn=True)
            if not success:
                return JsonResponse({'success': False, 'error': 'No face detected in the image.'})

            known_faces = FaceProfile.objects.all()
            if not known_faces:
                return JsonResponse({'success': False, 'error': 'No faces registered in the database.'})

            known_encodings = [face.get_encoding() for face in known_faces]
            known_names = [face.name for face in known_faces]

            # Calculate face distances
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            # Find the index with the smallest distance
            best_match_index = np.argmin(face_distances)

            # Check if the best match is within the defined threshold
            if face_distances[best_match_index] < THRESHOLD:
                name = known_names[best_match_index]
                confidence = (1 - face_distances[best_match_index]) * 100
                return JsonResponse({'success': True, 'name': name, 'confidence': f"{confidence:.2f}%"})
            else:
                return JsonResponse({'success': True, 'name': None, 'message': 'No close match found.'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': f'An error occurred: {str(e)}'})

    return render(request, 'test_face.html')
