# VP_FASTER - GPU-Accelerated Video Face Recognition & Clustering

A high-performance Django-based video processing pipeline that detects, clusters, and identifies faces in videos using GPU acceleration.

## Workflow

### 1. **Frame Extraction & Queuing**
- CPU reader thread extracts frames from video using OpenCV
- Frames queued into `frame_queue` (max 500 frames) as `(frame_idx, frame)` tuples
- Non-blocking producer-consumer pattern for continuous processing

### 2. **Batch Detection & Embedding Generation**
- GPU worker thread consumes 500-frame batches from queue
- **YOLO (YOLOv8n-face)** processes all 500 frames simultaneously on GPU for face detection
- Face crops extracted from detected bounding boxes
- **ArcFace** generates embeddings in sub-batches of 800 faces on GPU
- Results (frame_idx, bbox, embedding) pushed to `result_queue`

### 3. **GPU Memory Optimization**
- After all frames processed, YOLO and ArcFace models unloaded from GPU
- `torch.cuda.empty_cache()` called to free GPU memory
- Entire GPU memory now available for clustering

### 4. **Clustering**
- **HDBSCAN** (GPU-accelerated via cuML if available) clusters face embeddings
- Groups similar faces into clusters representing unique individuals
- Noise points marked as outliers

### 5. **Identification**
- Each cluster's mean embedding compared against **ChromaDB** vector database
- Enrolled faces matched using cosine similarity
- Clusters labeled as recognized person or "Unknown_X"

### 6. **Video Annotation**
- Original video re-read and annotated frame-by-frame
- Bounding boxes drawn: Green (recognized) / Red (unknown)
- Person names or "Unknown" labels added
- Output saved as processed video

## Tech Stack
- **Django** - Web framework
- **YOLOv8n-face** - Face detection
- **ArcFace (ONNX)** - Face embedding generation
- **HDBSCAN** - Clustering (GPU via cuML)
- **ChromaDB** - Vector database for face enrollment
- **PyTorch** - GPU acceleration
- **OpenCV** - Video I/O

## Features
- ⚡ GPU-accelerated processing (CUDA)
- 🔄 Multi-threaded pipeline for maximum throughput
- 📊 Automatic face clustering
- 🎯 Face recognition against enrolled database
- 📈 Real-time progress tracking
- 🎨 Visual cluster plots (PCA)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place ML models in `hdbscan/analysis_pipeline/models/`:
   - `yolov8n-face.pt`
   - `arcface.onnx`
3. Run migrations: `python manage.py migrate`
4. Start server: `python manage.py runserver`

## Usage
1. **Enroll faces**: Upload reference images with person IDs
2. **Upload video**: System processes automatically in background
3. **View results**: Annotated video with identified faces

## Performance
- Processes 500 frames simultaneously on GPU
- Batch embedding generation (800 faces at once)
- Memory-efficient: Models unloaded after detection phase
- Optimized for NVIDIA GPUs with CUDA support
