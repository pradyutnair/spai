import torch
import numpy as np
# Change path
import os
import sys
from warnings import filterwarnings
filterwarnings("ignore")
# Change path to this spai directory
print(f"🎯 Current working directory: {os.getcwd()}")
sys.path.append(os.getcwd())
# Go to build.py file
from spai.models.build import build_cls_model  # Change this import
from spai.config import get_config
import torchvision.transforms as transforms

# Create a random input tensor with shape (batch_size, channels, height, width)
batch_size = 2
img_size = 224
input_tensor = torch.randn(batch_size, 3, img_size, img_size)
print(f"🎯 Input tensor shape: {input_tensor.shape}")

# Normalize input to [0, 1] range
input_tensor = torch.clamp(input_tensor, min=0., max=1.)

# Load config and create model with semantic fusion enabled
print("🎯Loading configuration...")
config = get_config({"cfg": "configs/spai.yaml"})
config.defrost()
config.MODEL_WEIGHTS = "dinov2"  # Use CLIP backbone for semantic features
config.freeze()

print("\n 🎯 Creating model...")
model = build_cls_model(config)  # Use build_cls_model instead of build_model

# Print model architecture
print("\n 🎯 Model Architecture:")
print(model)

# Print model parameters
# print("\nModel Parameters:")
# for param in model.parameters():
#     print(param)

# Apply a forward pass to the model
output = model(input_tensor)
print(f"🎯 Output shape: {output.shape}")

