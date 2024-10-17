from django.db import models
import numpy as np
import json

class FaceProfile(models.Model):
    name = models.CharField(max_length=100)
    face_encodings = models.TextField()  # Store multiple encodings as a JSON string
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def set_encodings(self, encodings):
        self.face_encodings = json.dumps([enc.tolist() for enc in encodings])

    def get_encodings(self):
        return np.array(json.loads(self.face_encodings))

    def __str__(self):
        return self.name