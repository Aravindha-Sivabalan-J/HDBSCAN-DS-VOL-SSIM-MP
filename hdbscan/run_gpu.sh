#!/bin/bash

# Define the python executable for the super_final environment
PYTHON_EXEC="/home/cannyminds/miniconda3/envs/super_final/bin/python"

echo "Configuring GPU Environment for super_final..."

# robustly find site-packages and nvidia libs using the active python
SITE_PACKAGES=$($PYTHON_EXEC -c "import site; print(site.getsitepackages()[0])")

CUDNN_PATH="$SITE_PACKAGES/nvidia/cudnn/lib"
CUBLAS_PATH="$SITE_PACKAGES/nvidia/cublas/lib"
CUDA_RUNTIME_PATH="$SITE_PACKAGES/nvidia/cuda_runtime/lib"

echo "Found libraries at:"
echo "  CUDNN: $CUDNN_PATH"
echo "  CUBLAS: $CUBLAS_PATH"
echo "  CUDA RUNTIME: $CUDA_RUNTIME_PATH"

# Export to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CUDNN_PATH:$CUBLAS_PATH:$CUDA_RUNTIME_PATH:$LD_LIBRARY_PATH"

# Verify provider library exists
PROVIDER_LIB="$SITE_PACKAGES/onnxruntime/capi/libonnxruntime_providers_cuda.so"
if [ -f "$PROVIDER_LIB" ]; then
    echo "SUCCESS: Found CUDA provider library at $PROVIDER_LIB"
else
    echo "ERROR: CUDA provider library NOT found at $PROVIDER_LIB"
fi

echo "Starting Django Server..."
$PYTHON_EXEC manage.py runserver
