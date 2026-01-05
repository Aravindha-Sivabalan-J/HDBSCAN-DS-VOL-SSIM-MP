from django.db import models
import os

class UploadedVideo(models.Model):
    title = models.CharField(max_length=100, blank=True)
    video_file = models.FileField(upload_to='videos/')
    processed_file = models.FileField(upload_to='processed_videos/', null=True, blank=True)
    cluster_plot = models.ImageField(upload_to='cluster_plots/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    progress = models.IntegerField(default=0)
    status_message = models.CharField(max_length=255, default="Pending")
    
    def __str__(self):
        return self.title or f"Video {self.id}"

    def filename(self):
        return os.path.basename(self.video_file.name)
