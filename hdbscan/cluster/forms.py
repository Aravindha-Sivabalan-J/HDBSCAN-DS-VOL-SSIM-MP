from django import forms
from .models import UploadedVideo

class EnrollmentForm(forms.Form):
    person_id = forms.CharField(label='Person ID (Name)', max_length=100)
    image1 = forms.ImageField(label='Image 1', required=True)
    image2 = forms.ImageField(label='Image 2 (Optional)', required=False)
    image3 = forms.ImageField(label='Image 3 (Optional)', required=False)
    image4 = forms.ImageField(label='Image 4 (Optional)', required=False)
    image5 = forms.ImageField(label='Image 5 (Optional)', required=False)

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedVideo
        fields = ['video_file', 'title']
