from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.http import JsonResponse
from .forms import EnrollmentForm, VideoUploadForm
from .models import UploadedVideo
import threading
import os
import logging

logger = logging.getLogger(__name__)

# Global instance
_analyzer = None

def get_analyzer():
    global _analyzer
    if _analyzer is None:
        from analysis_pipeline.processor import VideoAnalyzer
        _analyzer = VideoAnalyzer()
    return _analyzer

def index(request):
    return render(request, 'index.html')

def enroll(request):
    context = {}
    if request.method == 'POST':
        form = EnrollmentForm(request.POST, request.FILES)
        if form.is_valid():
            person_id = form.cleaned_data['person_id']
            
            # Collect all uploaded images
            uploaded_images = []
            for i in range(1, 6):
                image_field = f'image{i}'
                if image_field in request.FILES:
                    uploaded_images.append(request.FILES[image_field])
            
            if not uploaded_images:
                context['error'] = "Please upload at least one image."
                context['form'] = form
                return render(request, 'enroll.html', context)
            
            # Save temp images
            if not os.path.exists(settings.MEDIA_ROOT):
                os.makedirs(settings.MEDIA_ROOT)
            
            temp_paths = []
            for idx, image in enumerate(uploaded_images):
                temp_path = os.path.join(settings.MEDIA_ROOT, f'temp_enroll_{person_id}_{idx}.jpg')
                with open(temp_path, 'wb+') as dest:
                    for chunk in image.chunks():
                        dest.write(chunk)
                temp_paths.append(temp_path)
            
            try:
                logger.info(f"👤 Starting enrollment for person: {person_id} with {len(temp_paths)} images")
                analyzer = get_analyzer()
                analyzer.enroll_face_multi(temp_paths, person_id)
                context['success'] = f"Successfully enrolled {person_id} with {len(temp_paths)} image(s)!"
                logger.info(f"✅ Successfully enrolled {person_id} with {len(temp_paths)} images")
            except Exception as e:
                logger.error(f"❌ Enrollment failed for {person_id}: {str(e)}")
                context['error'] = f"Error enrolling face: {str(e)}"
            finally:
                # Clean up temp files
                for temp_path in temp_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    else:
        form = EnrollmentForm()
    
    context['form'] = form
    return render(request, 'enroll.html', context)

def process_video_thread(video_id):
    logger.info(f"🎥 Starting video processing thread for video ID: {video_id}")
    try:
        from django.db import connection
        
        video = UploadedVideo.objects.get(id=video_id)
        input_path = video.video_file.path
        
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_processed.mp4"
        output_rel_path = f"processed_videos/{output_filename}"
        output_path = os.path.join(settings.MEDIA_ROOT, 'processed_videos', output_filename)
        
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define progress callback
        def on_progress(percent, message):
            # We need to get a fresh instance/refresh to update safely in thread
            # Ideally use update() on queryset to avoid race cond, but simpler here
            UploadedVideo.objects.filter(id=video_id).update(progress=percent, status_message=message)
            logger.info(f"Progress: {percent}% - {message}")

        analyzer = get_analyzer()
        logger.info("🚀 Initializing video analysis...")
        on_progress(0, "Starting analysis...")
        
        # Update processor to accept callback
        success, plot_path = analyzer.process_video(input_path, output_path, progress_callback=on_progress)
        
        # Refresh object to update fields
        video.refresh_from_db()
        
        if success:
            video.processed_file.name = output_rel_path
            if plot_path:
                # Assuming plot_path is absolute, make it relative to media root
                # processor returns absolute path. 
                # Let's verify processor output path logic for plot.
                # If processor saves to same dir as output video (processed_videos), 
                # we can deduce relative path.
                
                # In processor: plot_path = output_path.replace('.mp4', '_plot.png')
                plot_filename = os.path.basename(plot_path)
                video.cluster_plot.name = f"processed_videos/{plot_filename}"
                logger.info(f"📈 Cluster plot saved: {plot_filename}")

            video.is_processed = True
            video.status_message = "Completed"
            video.progress = 100
            video.save()
            logger.info(f"✅ Video {video_id} processed successfully - Output: {output_filename}")
        else:
            video.status_message = "Failed: No faces found"
            video.progress = 0
            video.save()
            logger.warning(f"⚠️ Video {video_id} processing failed or no faces found")
            
        connection.close()
            
    except Exception as e:
        logger.error(f"❌ Critical error processing video {video_id}: {e}")
        try:
             UploadedVideo.objects.filter(id=video_id).update(status_message=f"Error: {str(e)[:50]}")
        except: pass
        from django.db import connection
        connection.close()

def upload_video(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.save()
            # Start processing in thread
            t = threading.Thread(target=process_video_thread, args=(video.id,))
            t.daemon = True 
            t.start()
            return redirect('gallery')
    else:
        form = VideoUploadForm()
    return render(request, 'upload.html', {'form': form})

def gallery(request):
    videos = UploadedVideo.objects.all().order_by('-uploaded_at')
    return render(request, 'gallery.html', {'videos': videos})

def video_status(request, video_id):
    video = get_object_or_404(UploadedVideo, id=video_id)
    return JsonResponse({
        'id': video.id,
        'progress': video.progress,
        'status_message': video.status_message,
        'is_processed': video.is_processed,
        'processed_url': video.processed_file.url if video.processed_file else None,
        'plot_url': video.cluster_plot.url if video.cluster_plot else None
    })
