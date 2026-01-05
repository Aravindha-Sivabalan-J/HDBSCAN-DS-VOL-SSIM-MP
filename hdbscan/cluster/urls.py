from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('enroll/', views.enroll, name='enroll'),
    path('upload/', views.upload_video, name='upload_video'),
    path('gallery/', views.gallery, name='gallery'),
    path('status/<int:video_id>/', views.video_status, name='video_status'),
]