from django import forms
from .models import FaceProfile

class FaceProfileForm(forms.ModelForm):
    image = forms.ImageField()

    class Meta:
        model = FaceProfile
        fields = ['name']