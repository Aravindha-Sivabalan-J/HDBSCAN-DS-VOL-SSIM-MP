#!/usr/bin/env python3
"""
Quick status check for the Facial Recognition System
"""

import os
import sys

def check_system_status():
    """Quick system status check."""
    print("🔍 Facial Recognition System Status Check")
    print("=" * 45)
    
    # Check if models exist
    models_dir = "analysis_pipeline/models"
    required_models = ["yolov8n-face.pt", "arcface.onnx"]
    
    print("\n📁 Model Files:")
    for model in required_models:
        model_path = os.path.join(models_dir, model)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✅ {model} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {model} - MISSING")
    
    # Check ChromaDB
    chroma_path = "chroma_db_store"
    if os.path.exists(chroma_path):
        print(f"\n💾 ChromaDB: ✅ Found at {chroma_path}")
        
        # Count files in ChromaDB to estimate entries
        try:
            db_files = []
            for root, dirs, files in os.walk(chroma_path):
                db_files.extend(files)
            print(f"  📊 Database files: {len(db_files)}")
        except:
            pass
    else:
        print(f"\n💾 ChromaDB: ❌ Not found")
    
    # Check media directories
    media_dirs = ["media/videos", "media/processed_videos", "media/cluster_plots"]
    print(f"\n📂 Media Directories:")
    
    for media_dir in media_dirs:
        if os.path.exists(media_dir):
            file_count = len([f for f in os.listdir(media_dir) if os.path.isfile(os.path.join(media_dir, f))])
            print(f"  ✅ {media_dir} ({file_count} files)")
        else:
            print(f"  ❌ {media_dir} - Missing")
    
    # Check Django database
    if os.path.exists("db.sqlite3"):
        size_kb = os.path.getsize("db.sqlite3") / 1024
        print(f"\n🗄️  Django DB: ✅ db.sqlite3 ({size_kb:.1f} KB)")
    else:
        print(f"\n🗄️  Django DB: ❌ db.sqlite3 not found")
    
    # Check requirements
    print(f"\n📋 Quick Dependencies Check:")
    required_packages = ["django", "chromadb", "ultralytics", "hdbscan", "cv2"]
    
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                print(f"  ✅ opencv-python")
            else:
                __import__(package)
                print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Not installed")
    
    print(f"\n🚀 To start the server: python manage.py runserver")
    print(f"🧪 To run tests: python test_system.py")

if __name__ == "__main__":
    check_system_status()