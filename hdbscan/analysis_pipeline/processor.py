import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
try:
    import cuml
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    GPU_HDBSCAN_AVAILABLE = True
except ImportError:
    import hdbscan
    GPU_HDBSCAN_AVAILABLE = False

# --- GPU LIBRARY FIX ---
# Dynamically add nvidia-cudnn and nvidia-cublas paths to LD_LIBRARY_PATH
# This is required because pip-installed nvidia packages don't always register with ldconfig
try:
    import site
    site_packages = site.getsitepackages()
    
    # Try to find site-packages containing nvidia
    for path in site_packages:
        nvidia_path = os.path.join(path, "nvidia")
        if os.path.exists(nvidia_path):
            cudnn_lib = os.path.join(nvidia_path, "cudnn", "lib")
            cublas_lib = os.path.join(nvidia_path, "cublas", "lib")
            
            libs_to_add = []
            if os.path.exists(cudnn_lib): libs_to_add.append(cudnn_lib)
            if os.path.exists(cublas_lib): libs_to_add.append(cublas_lib)
            
            if libs_to_add:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                new_path = ":".join(libs_to_add) + ":" + current_ld
                os.environ["LD_LIBRARY_PATH"] = new_path
                # Also try adding to sys.path just in case
                # sys.path.extend(libs_to_add)
                print(f"Updated LD_LIBRARY_PATH with: {libs_to_add}")
                break
except Exception as e:
    print(f"Warning: Failed to auto-configure CUDA paths: {e}")

import onnxruntime
import onnx
import logging
import matplotlib
# Use Agg backend for non-interactive plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ultralytics import YOLO
from .vector_db import FaceDB

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, ssim_threshold=0.77):
        # Locate models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        
        self.yolo_path = os.path.join(models_dir, "yolov8n-face.pt")
        self.arcface_path = os.path.join(models_dir, "arcface.onnx")
        self.ssim_threshold = ssim_threshold
        
        logger.info(f"Loading YOLO from {self.yolo_path}...")
        self.yolo = YOLO(self.yolo_path)
        if torch.cuda.is_available():
            self.yolo.to('cuda')
            logger.info("✅ YOLO loaded with CUDA acceleration")
        else:
            logger.warning("⚠️ CUDA not available for YOLO, using CPU")
        
        logger.info(f"Loading ArcFace from {self.arcface_path}...")
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            # Use PyTorch with GPU for ArcFace
            try:
                from onnx2torch import convert
                onnx_model = onnx.load(self.arcface_path)
                self.arcface_model = convert(onnx_model)
                self.arcface_model = self.arcface_model.to('cuda').eval()
                self.arcface_backend = 'torch'
                logger.info("✅ ArcFace using PyTorch GPU")
            except Exception as e:
                logger.warning(f"⚠️ PyTorch conversion failed: {e}, using ONNX CPU")
                self.ort_session = onnxruntime.InferenceSession(self.arcface_path, providers=['CPUExecutionProvider'])
                self.input_name = self.ort_session.get_inputs()[0].name
                self.arcface_backend = 'onnx'
        else:
            self.ort_session = onnxruntime.InferenceSession(self.arcface_path, providers=['CPUExecutionProvider'])
            self.input_name = self.ort_session.get_inputs()[0].name
            self.arcface_backend = 'onnx'
            logger.warning("⚠️ No GPU available, using CPU")
        
        self.face_db = FaceDB()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def gpu_laplacian_variance(self, frame_tensor):
        """GPU-accelerated Laplacian variance computation for blur detection"""
        # Laplacian kernel
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Apply Laplacian
        laplacian = F.conv2d(frame_tensor, laplacian_kernel, padding=1)
        
        # Compute variance and scale to match cv2.Laplacian range (0-255)
        variance = torch.var(laplacian).item() * (255.0 ** 2)
        return variance

    def gpu_ssim(self, img1, img2, window_size=7):
        """GPU-accelerated SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        sigma = 1.5
        kernel = torch.exp(-torch.arange(-(window_size//2), window_size//2 + 1, dtype=torch.float32).pow(2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel_2d = kernel[:, None] * kernel[None, :]
        kernel_2d = kernel_2d.expand(1, 1, window_size, window_size).to(self.device)
        
        mu1 = F.conv2d(img1, kernel_2d, padding=window_size//2)
        mu2 = F.conv2d(img2, kernel_2d, padding=window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, kernel_2d, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel_2d, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel_2d, padding=window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean().item()

    def _gpu_worker(self, frame_queue, result_queue, stop_event, device='cuda'):
        """GPU worker thread - processes 500 unique frames at once."""
        
        CHUNK_SIZE = 500
        
        while not stop_event.is_set():
            # Collect 500 unique frames
            frame_batch = []
            frame_indices = []
            
            for _ in range(CHUNK_SIZE):
                try:
                    frame_data = frame_queue.get(timeout=0.1)
                    if frame_data is None:  # End signal
                        break
                    frame_idx, frame = frame_data
                    frame_batch.append(frame)
                    frame_indices.append(frame_idx)
                except queue.Empty:
                    break
            
            if not frame_batch:
                break
            
            logger.info(f"GPU processing {len(frame_batch)} frames...")
            
            # Run YOLO on all frames
            all_faces = []
            all_metadata = []
            
            results = self.yolo(frame_batch, verbose=False, device=device, max_det=20, conf=0.5, iou=0.5, stream=True, agnostic_nms=True)
            for result, frame_idx, frame in zip(results, frame_indices, frame_batch):
                h, w, _ = frame.shape
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0: continue
                    all_faces.append(face_crop)
                    all_metadata.append({"frame_idx": frame_idx, "bbox": (x1, y1, x2, y2)})
            
            # Generate embeddings for ALL faces in smaller batches to avoid OOM
            if all_faces:
                logger.info(f"Generating embeddings for {len(all_faces)} faces...")
                
                # Process in batches of 800 to maximize GPU utilization
                EMBEDDING_BATCH_SIZE = 800
                all_embeddings = []
                
                for i in range(0, len(all_faces), EMBEDDING_BATCH_SIZE):
                    batch_faces = all_faces[i:i+EMBEDDING_BATCH_SIZE]
                    batch_embeddings = self.get_embedding_batch(batch_faces)
                    all_embeddings.extend(batch_embeddings)
                    
                    if len(all_faces) > EMBEDDING_BATCH_SIZE:
                        logger.info(f"Processed {min(i + EMBEDDING_BATCH_SIZE, len(all_faces))}/{len(all_faces)} embeddings...")
                
                # Send results to queue
                for embedding, metadata in zip(all_embeddings, all_metadata):
                    result_queue.put({**metadata, "embedding": embedding})
                
                logger.info(f"Completed batch: {len(all_faces)} faces processed")
        
        logger.info(f"GPU worker completed")

    def get_embedding_batch(self, face_crops):
        """Generate embeddings for batch of face crops on GPU."""
        try:
            if self.arcface_backend == 'torch' and self.use_gpu:
                # GPU batch processing with PyTorch
                batch = []
                for face_img in face_crops:
                    face_img = cv2.resize(face_img, (112, 112))
                    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))
                    img = (img - 127.5) / 128.0
                    batch.append(img)
                
                batch_tensor = torch.from_numpy(np.array(batch, dtype=np.float32)).to('cuda')
                
                with torch.no_grad():
                    embeddings = self.arcface_model(batch_tensor)
                    embeddings = embeddings.cpu().numpy()
                
                # Normalize
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-10)
                return embeddings
            else:
                # CPU fallback
                embeddings = []
                for face_img in face_crops:
                    face_img = cv2.resize(face_img, (112, 112))
                    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1))
                    img = np.expand_dims(img, axis=0)
                    img = (img - 127.5) / 128.0
                    img = img.astype(np.float32)
                    
                    embedding = self.ort_session.run(None, {self.input_name: img})[0]
                    embedding = embedding.flatten()
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return np.zeros((len(face_crops), 512))
    
    def get_embedding(self, face_img):
        """Generate single embedding (uses batch internally)."""
        return self.get_embedding_batch([face_img])[0]

    def enroll_face_multi(self, image_paths, person_id):
        """
        Detects faces in multiple images and creates mean embedding.
        Supports 1-5 images per person for robust enrollment.
        """
        logger.info(f"Starting multi-image enrollment for person_id: {person_id} with {len(image_paths)} images")
        
        all_embeddings = []
        device = 'cuda' if hasattr(self.yolo.model, 'device') and 'cuda' in str(self.yolo.model.device) else 'cpu'
        
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_path}, skipping...")
                continue
            
            logger.info(f"Running face detection on image {idx + 1}...")
            results = self.yolo(img, device=device, verbose=False, max_det=5, conf=0.5, stream=True, agnostic_nms=True)
            
            best_face = None
            max_area = 0
            face_count = 0
            
            for result in results:
                for box in result.boxes:
                    face_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        best_face = img[y1:y2, x1:x2]
            
            logger.info(f"Detected {face_count} faces in image {idx + 1}")
            if best_face is None:
                logger.warning(f"No face detected in image {idx + 1}, skipping...")
                continue
            
            logger.info(f"Generating embedding for image {idx + 1}...")
            embedding = self.get_embedding(best_face)
            all_embeddings.append(embedding)
            logger.info(f"✅ Embedding generated for image {idx + 1}")
        
        if not all_embeddings:
            logger.error("No faces detected in any of the uploaded images")
            raise ValueError("No faces detected in any of the uploaded images")
        
        logger.info(f"Computing mean embedding from {len(all_embeddings)} embeddings...")
        # Compute mean embedding
        mean_embedding = np.mean(all_embeddings, axis=0)
        
        # Normalize mean embedding
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        
        logger.info(f"Adding {person_id} to vector database with mean embedding...")
        self.face_db.add_person(person_id, mean_embedding)
        logger.info(f"✅ Successfully enrolled {person_id} with {len(all_embeddings)} image(s)")
        return True

    def generate_cluster_plot(self, embeddings, labels, output_path):
        """
        Generates a 2D PCA plot of the face clusters.
        """
        try:
            if len(embeddings) < 2:
                return None
                
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            unique_labels = set(labels)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    c = 'k' # Black for noise
                    marker = 'x'
                    label_text = "Noise"
                else:
                    c = color
                    marker = 'o'
                    label_text = f"Cluster {label}"
                    
                indices = [i for i, l in enumerate(labels) if l == label]
                plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], c=[c], label=label_text, marker=marker, alpha=0.6)
                
            plt.title('Face Clusters Visualization (PCA)')
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path)
            plt.close()
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate cluster plot: {e}")
            return None

    def process_video(self, video_path, output_path, progress_callback=None):
        def update_progress(percent, msg):
            if progress_callback:
                progress_callback(percent, msg)
            logger.info(f"{percent}% - {msg}")

        # Step 1: Save video to temp location (already uploaded, just reference it)
        temp_video_path = video_path  # Video already on disk from upload
        
        cap = cv2.VideoCapture(temp_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        update_progress(0, f"Extracting and downscaling {total_frames} frames...")
        
        # Step 2: Extract frames, downscale, and apply GPU blur detection
        downscaled_frames = []
        sharp_frame_indices = []  # Track which original frames are sharp
        frame_count = 0
        blur_threshold = 600
        
        logger.info(f"Extracting frames with GPU blur detection (threshold: {blur_threshold})...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create downscaled copy (320x180)
            downscaled = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            
            # Calculate Variance of Laplacian on GPU for blur detection
            gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
            frame_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            variance = self.gpu_laplacian_variance(frame_tensor)
            
            # Only keep sharp frames (variance >= 600)
            if variance >= blur_threshold:
                downscaled_frames.append(downscaled)
                sharp_frame_indices.append(frame_count)
            
            frame_count += 1
            
            if frame_count % 500 == 0:
                prog = int((frame_count / total_frames) * 10)
                update_progress(prog, f"GPU blur detection: {frame_count}/{total_frames} (Sharp: {len(downscaled_frames)})")
                torch.cuda.empty_cache()
        
        cap.release()
        torch.cuda.empty_cache()
        
        blurry_count = frame_count - len(downscaled_frames)
        logger.info(f"✅ GPU blur detection complete: {len(downscaled_frames)} sharp frames, {blurry_count} blurry frames removed")
        logger.info(f"📉 Blur reduction: {(blurry_count / frame_count * 100):.1f}%")
        
        if not downscaled_frames:
            update_progress(100, "No sharp frames found in video.")
            return False, None
        
        update_progress(10, f"Running SSIM on {len(downscaled_frames)} sharp frames...")
        
        # Step 3: Run SSIM on sharp downscaled frames to find unique indices
        unique_indices = []  # Indices within sharp_frame_indices list
        prev_frame_tensor = None
        
        logger.info("Starting SSIM-based unique frame detection on sharp frames...")
        
        for idx, downscaled_frame in enumerate(downscaled_frames):
            # Convert to grayscale tensor for SSIM
            gray = cv2.cvtColor(downscaled_frame, cv2.COLOR_BGR2GRAY)
            frame_tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
            
            # First frame is always unique
            if prev_frame_tensor is None:
                unique_indices.append(idx)
                prev_frame_tensor = frame_tensor
            else:
                # Compute SSIM with previous unique frame
                ssim_score = self.gpu_ssim(prev_frame_tensor, frame_tensor)
                
                if ssim_score < self.ssim_threshold:
                    unique_indices.append(idx)
                    prev_frame_tensor = frame_tensor
            
            if (idx + 1) % 500 == 0:
                prog = 10 + int(((idx + 1) / len(downscaled_frames)) * 10)
                update_progress(prog, f"SSIM filtering: {idx + 1}/{len(downscaled_frames)} (Unique: {len(unique_indices)})")
                torch.cuda.empty_cache()
        
        # Map unique indices back to original frame indices
        unique_original_indices = [sharp_frame_indices[i] for i in unique_indices]
        
        # Clear downscaled frames from memory
        del downscaled_frames
        torch.cuda.empty_cache()
        
        redundant_count = len(sharp_frame_indices) - len(unique_indices)
        logger.info(f"✅ SSIM complete: {len(unique_indices)} unique frames, {redundant_count} redundant frames removed")
        logger.info(f"📉 SSIM reduction: {(redundant_count / len(sharp_frame_indices) * 100):.1f}%")
        logger.info(f"📊 Total reduction: {len(unique_indices)}/{frame_count} frames ({(len(unique_indices) / frame_count * 100):.1f}% retained)")
        
        if not unique_original_indices:
            update_progress(100, "No unique frames found.")
            return False, None
        
        update_progress(20, f"Extracting {len(unique_original_indices)} sharp+unique frames from video...")
        
        # Step 4: Extract only sharp+unique high-quality frames from original video
        unique_frames = []
        cap = cv2.VideoCapture(temp_video_path)
        
        logger.info(f"Extracting {len(unique_original_indices)} sharp+unique frames at full quality...")
        
        current_frame_idx = 0
        unique_idx_set = set(unique_original_indices)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only keep frames that are sharp AND unique
            if current_frame_idx in unique_idx_set:
                unique_frames.append(frame)
                
                if len(unique_frames) % 500 == 0:
                    prog = 20 + int((len(unique_frames) / len(unique_original_indices)) * 5)
                    update_progress(prog, f"Extracting frames: {len(unique_frames)}/{len(unique_original_indices)}")
            
            current_frame_idx += 1
        
        cap.release()
        
        logger.info(f"✅ Extracted {len(unique_frames)} sharp+unique high-quality frames")
        
        update_progress(25, f"Processing {len(unique_frames)} frames...")
        
        # Step 4: Process unique frames (Detection & Embedding)
        detections = []
        frame_queue = queue.Queue(maxsize=500)
        result_queue = queue.Queue()
        stop_event = threading.Event()
        
        # Determine device for GPU worker
        device = 'cuda' if torch.cuda.is_available() and hasattr(self, 'yolo') and hasattr(self.yolo, 'model') else 'cpu'
        
        gpu_thread = threading.Thread(target=self._gpu_worker, args=(frame_queue, result_queue, stop_event, device))
        gpu_thread.daemon = True
        gpu_thread.start()
        
        # Producer: Queue unique frames
        def queue_unique_frames():
            for idx, frame in enumerate(unique_frames):
                frame_queue.put((idx, frame))
            frame_queue.put(None)
        
        reader_thread = threading.Thread(target=queue_unique_frames)
        reader_thread.daemon = True
        reader_thread.start()
        
        # Collector: Gather results
        last_log = 0
        frames_with_faces = set()
        while reader_thread.is_alive() or not result_queue.empty():
            try:
                detection = result_queue.get(timeout=0.1)
                detections.append(detection)
                frames_with_faces.add(detection["frame_idx"])
                
                if len(frames_with_faces) - last_log >= 100:
                    prog = 25 + int((len(frames_with_faces) / len(unique_frames)) * 30)
                    update_progress(prog, f"Processing unique frames: {len(frames_with_faces)}/{len(unique_frames)} (Faces: {len(detections)})")
                    last_log = len(frames_with_faces)
            except queue.Empty:
                continue
        
        reader_thread.join()
        gpu_thread.join()
        
        while not result_queue.empty():
            try:
                detection = result_queue.get_nowait()
                detections.append(detection)
            except queue.Empty:
                break
        
        logger.info(f"Total detections collected: {len(detections)}")
        
        if not detections:
            update_progress(100, "No faces found in video.")
            return False, None

        # Unload models from GPU to free memory for clustering
        logger.info("🔄 Unloading YOLO and ArcFace from GPU...")
        if hasattr(self, 'yolo'):
            del self.yolo
        if hasattr(self, 'arcface_model'):
            del self.arcface_model
        if hasattr(self, 'ort_session'):
            del self.ort_session
        torch.cuda.empty_cache()
        logger.info("✅ GPU memory freed for clustering")

        # Extract embeddings from detections
        update_progress(57, f"Preparing {len(detections)} embeddings for clustering...")
        embeddings = [d["embedding"] for d in detections]
        embeddings_array = np.array(embeddings)
        logger.info(f"✅ Using {len(embeddings)} embeddings generated during detection")
        
        # Clustering
        update_progress(60, f"Clustering {len(embeddings)} faces...")
        logger.info(f"Starting clustering on {len(embeddings)} face embeddings...")
        
        # HDBSCAN
        if len(embeddings_array) >= 3:
            # Skip PCA - use raw embeddings for better quality
            logger.info("Skipping PCA - using raw 512-dim embeddings for clustering...")
            reduced_embeddings = embeddings_array

            # Force GPU HDBSCAN
            update_progress(63, f"Running GPU HDBSCAN clustering on {len(reduced_embeddings)} points...")
            
            if GPU_HDBSCAN_AVAILABLE:
                logger.info("Running GPU-accelerated HDBSCAN (cuML)...")
                import cudf
                reduced_df = cudf.DataFrame(reduced_embeddings)
                clusterer = cumlHDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5)
                labels = clusterer.fit_predict(reduced_df).to_numpy()
                logger.info("✅ GPU HDBSCAN completed")
            else:
                logger.info(f"Running CPU HDBSCAN (cuML not installed) on {len(reduced_embeddings)} points...")
                import hdbscan
                clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_epsilon=0.5, core_dist_n_jobs=-1)
                labels = clusterer.fit_predict(reduced_embeddings)
                logger.info("✅ CPU HDBSCAN completed")
            
            update_progress(64, "Clustering analysis complete...")
        else:
            labels = list(range(len(embeddings_array))) # Too few faces, treat unique
        
        update_progress(65, f"Found {len(set(labels))} unique clusters.")

        # Skip visualization for speed
        generated_plot = None

        # Identification
        update_progress(70, "Identifying clusters against database...")
        cluster_map = {} # label -> person_id
        unique_labels = set(labels)
        
        logger.info(f"Identifying {len(unique_labels)} clusters against enrolled database...")
        
        for label in unique_labels:
            if label == -1:
                logger.info(f"Skipping noise cluster (label -1)")
                continue # Noise
            
            # Get mean embedding for this cluster
            cluster_indices = [i for i, x in enumerate(labels) if x == label]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            logger.info(f"Cluster {label}: {len(cluster_indices)} faces")
            
            mean_emb = np.mean(cluster_embeddings, axis=0)
            
            # Renormalize mean ?
            norm = np.linalg.norm(mean_emb)
            if norm > 0: mean_emb = mean_emb / norm
            
            logger.info(f"Searching database for cluster {label}...")
            person_id = self.face_db.search_person(mean_emb)
            if person_id:
                cluster_map[label] = person_id
                logger.info(f"✅ Cluster {label} identified as: {person_id}")
            else:
                cluster_map[label] = f"Unknown_{label}"
                logger.info(f"❌ Cluster {label} not recognized - marked as Unknown_{label}")
        
        logger.info(f"Identification complete. Recognized: {sum(1 for v in cluster_map.values() if not v.startswith('Unknown'))}/{len(cluster_map)} clusters")
                
        # Annotate unique frames video
        update_progress(80, "Annotating unique frames video...")
        logger.info(f"Creating annotated video from {len(unique_frames)} unique frames...")
        
        logger.info(f"Output video specs: {width}x{height} @ {fps:.1f}fps")
        
        # Try GPU-accelerated encoding
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
            if not out.isOpened():
                raise Exception("GPU encoder failed")
            logger.info("✅ Using H.264 hardware encoding")
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info("⚠️ Using software encoding")
        
        # Build detection lookup
        detection_map = {}
        for idx, d in enumerate(detections):
            frame_idx = d["frame_idx"]
            if frame_idx not in detection_map:
                detection_map[frame_idx] = []
            label = labels[idx]
            name = cluster_map.get(label, "Unknown") if label != -1 else "Unknown"
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            detection_map[frame_idx].append({"bbox": d["bbox"], "name": name, "color": color})
        
        # Annotate and write unique frames
        for frame_idx, frame in enumerate(unique_frames):
            if frame_idx in detection_map:
                for det in detection_map[frame_idx]:
                    x1, y1, x2, y2 = det["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), det["color"], 2)
                    cv2.putText(frame, det["name"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, det["color"], 2)
            
            out.write(frame)
            
            if frame_idx % 100 == 0:
                prog = 80 + int((frame_idx / len(unique_frames)) * 19)
                update_progress(prog, f"Annotating: {frame_idx}/{len(unique_frames)}")
        
        out.release()
        
        update_progress(100, "Processing completed.")
        logger.info(f"Completed. Saved to {output_path}")
        
        return True, generated_plot
