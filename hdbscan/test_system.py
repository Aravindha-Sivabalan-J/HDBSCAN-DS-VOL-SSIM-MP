#!/usr/bin/env python3
"""
System Test Script for Facial Recognition Application
Tests the core components without Django to verify functionality.
"""

import os
import sys
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if models can be loaded successfully."""
    logger.info("🧪 Testing model loading...")
    
    try:
        from analysis_pipeline.processor import VideoAnalyzer
        analyzer = VideoAnalyzer()
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_vector_db():
    """Test ChromaDB functionality."""
    logger.info("🧪 Testing ChromaDB connection...")
    
    try:
        from analysis_pipeline.vector_db import FaceDB
        import numpy as np
        
        db = FaceDB()
        
        # Test adding a dummy embedding
        dummy_embedding = np.random.rand(512)
        db.add_person("test_person", dummy_embedding)
        
        # Test searching
        result = db.search_person(dummy_embedding, threshold=0.1)
        
        if result == "test_person":
            logger.info("✅ ChromaDB test passed")
            return True
        else:
            logger.error("❌ ChromaDB search failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ ChromaDB test failed: {e}")
        return False

def test_cuda_availability():
    """Check CUDA availability for GPU acceleration."""
    logger.info("🧪 Testing CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logger.warning("⚠️ CUDA not available - will use CPU")
            return False
    except ImportError:
        logger.warning("⚠️ PyTorch not installed - cannot check CUDA")
        return False

def check_media_directories():
    """Ensure media directories exist."""
    logger.info("🧪 Checking media directories...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    media_dirs = [
        os.path.join(base_dir, 'media'),
        os.path.join(base_dir, 'media', 'videos'),
        os.path.join(base_dir, 'media', 'processed_videos'),
        os.path.join(base_dir, 'media', 'cluster_plots')
    ]
    
    for dir_path in media_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"📁 Created directory: {dir_path}")
        else:
            logger.info(f"✅ Directory exists: {dir_path}")
    
    return True

def main():
    """Run all system tests."""
    logger.info("🚀 Starting Facial Recognition System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Media Directories", check_media_directories),
        ("CUDA Availability", test_cuda_availability),
        ("Vector Database", test_vector_db),
        ("Model Loading", test_model_loading),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} : {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All systems operational! Ready to process videos.")
    else:
        logger.warning("⚠️ Some tests failed. Check logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)