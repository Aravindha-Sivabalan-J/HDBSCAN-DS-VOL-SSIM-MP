import torch
print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"CuDNN Version: {torch.backends.cudnn.version()}")
else:
    print("CUDA not available in Torch.")
