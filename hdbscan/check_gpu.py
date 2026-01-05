#!/usr/bin/env python3
"""
GPU and CUDA verification script
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        logger.info(f"🔍 PyTorch version: {torch.__version__}")
        logger.info(f"🔍 CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"🔍 CUDA version: {torch.version.cuda}")
            logger.info(f"🔍 GPU device: {torch.cuda.get_device_name(0)}")
            logger.info(f"🔍 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ PyTorch CUDA check failed: {e}")
        return False

def check_onnx_providers():
    """Check ONNX Runtime providers."""
    try:
        import onnxruntime as ort
        logger.info(f"🔍 ONNX Runtime version: {ort.__version__}")
        
        available_providers = ort.get_available_providers()
        logger.info(f"🔍 Available providers: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            logger.info("✅ CUDA provider available for ONNX")
            # Test if CUDA actually works
            try:
                import numpy as np
                test_session = ort.InferenceSession(
                    "/home/cannyminds/Desktop/VP_FASTER/hdbscan/analysis_pipeline/models/arcface.onnx",
                    providers=['CUDAExecutionProvider']
                )
                if 'CUDAExecutionProvider' in test_session.get_providers():
                    logger.info("✅ CUDA provider functional")
                    return True
                else:
                    logger.warning("⚠️ CUDA provider failed to initialize")
                    return False
            except:
                logger.warning("⚠️ CUDA provider available but not functional (cuDNN issue)")
                return False
        else:
            logger.warning("⚠️ CUDA provider NOT available for ONNX")
            return False
    except Exception as e:
        logger.error(f"❌ ONNX provider check failed: {e}")
        return False

def test_model_gpu_usage():
    """Test actual GPU usage with models."""
    try:
        from analysis_pipeline.processor import VideoAnalyzer
        
        logger.info("🧪 Testing model GPU usage...")
        analyzer = VideoAnalyzer()
        
        # Check YOLO device
        import torch
        if hasattr(analyzer.yolo.model, 'device'):
            device = analyzer.yolo.model.device
            logger.info(f"🔍 YOLO device: {device}")
            if 'cuda' in str(device):
                logger.info("✅ YOLO using GPU")
            else:
                logger.warning("⚠️ YOLO using CPU")
        
        # Test ONNX session
        try:
            providers = analyzer.ort_session.get_providers()
            logger.info(f"🔍 ArcFace providers: {providers}")
            if 'CUDAExecutionProvider' in providers:
                logger.info("✅ ArcFace using GPU")
            else:
                logger.warning("⚠️ ArcFace using CPU")
        except:
            logger.warning("⚠️ Could not check ArcFace providers")
            
        return True
    except Exception as e:
        logger.error(f"❌ Model GPU test failed: {e}")
        return False

def main():
    logger.info("🚀 GPU and CUDA Verification")
    logger.info("=" * 40)
    
    pytorch_ok = check_pytorch_cuda()
    onnx_ok = check_onnx_providers()
    models_ok = test_model_gpu_usage()
    
    logger.info("\n" + "=" * 40)
    logger.info("📊 GPU STATUS SUMMARY")
    logger.info("=" * 40)
    
    if pytorch_ok and onnx_ok and models_ok:
        logger.info("🎉 GPU acceleration fully operational!")
    elif pytorch_ok:
        logger.warning("⚠️ Partial GPU support - YOLO will use GPU, ArcFace may use CPU")
    else:
        logger.warning("⚠️ No GPU acceleration - models will use CPU")

if __name__ == "__main__":
    main()