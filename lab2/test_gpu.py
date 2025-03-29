import torch
print(f"GPU available: {torch.cuda.is_available()}")  
print(f"GPU name: {torch.cuda.get_device_name(0)}")  
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Compute capability: {torch.cuda.get_device_capability()}")

# installed cuda 12.4 for windows
# reboot
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124