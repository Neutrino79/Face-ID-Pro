from django.db import models
import numpy as np
import base64


class FaceProfile(models.Model):
    name = models.CharField(max_length=100)
    face_encoding = models.BinaryField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def set_encoding(self, encoding):
        self.face_encoding = base64.b64encode(encoding.tobytes())

    def get_encoding(self):
        decoded = base64.b64decode(self.face_encoding)
        return np.frombuffer(decoded, dtype=np.float64)
