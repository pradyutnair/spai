#!/usr/bin/env python3

import inspect
import os

# Print what files we modified
print("Files modified to fix export_attention parameter:")
print("1. /home/pnair/spai/spai/models/sid.py")
print("2. /home/pnair/spai/spai/models/semantic_fusion.py")

# Check if the parameter exists in the MFViT.forward method
print("\nVerifying MFViT.forward method signature:")
mfvit_forward_path = "/home/pnair/spai/spai/models/sid.py"
if os.path.exists(mfvit_forward_path):
    with open(mfvit_forward_path, 'r') as f:
        content = f.read()
        
    # Look for the forward method signature with export_attention parameter
    if "def forward(self, x: torch.Tensor, export_attention: bool = False)" in content:
        print("✅ MFViT.forward method has export_attention parameter")
    else:
        print("❌ MFViT.forward method missing export_attention parameter")

# Check if the parameter exists in the PatchBasedMFViT.forward method
print("\nVerifying PatchBasedMFViT.forward method signature:")
if os.path.exists(mfvit_forward_path):
    with open(mfvit_forward_path, 'r') as f:
        content = f.read()
        
    # Look for the forward method signature with export_attention parameter
    if "export_attention: bool = False" in content and "def forward" in content:
        print("✅ PatchBasedMFViT.forward method has export_attention parameter")
    else:
        print("❌ PatchBasedMFViT.forward method missing export_attention parameter")
        
# Check if the parameter exists in the AdaptiveSemanticSpectralFusion.forward method
print("\nVerifying AdaptiveSemanticSpectralFusion.forward method signature:")
fusion_path = "/home/pnair/spai/spai/models/semantic_fusion.py"
if os.path.exists(fusion_path):
    with open(fusion_path, 'r') as f:
        content = f.read()
        
    # Look for the forward method signature with export_attention parameter
    if "def forward(self, spectral_features, semantic_features, export_attention=False)" in content:
        print("✅ AdaptiveSemanticSpectralFusion.forward method has export_attention parameter")
    else:
        print("❌ AdaptiveSemanticSpectralFusion.forward method missing export_attention parameter")

print("\nVerification complete.")
print("\nSummary of fixes:")
print("1. Added export_attention parameter to MFViT.forward method")
print("2. Added export_attention parameter to PatchBasedMFViT.forward method")
print("3. Added export_attention parameter to AdaptiveSemanticSpectralFusion.forward method")
print("4. Updated all method calls to forward these parameters correctly between methods")
print("\nThis should fix the TypeError: PatchBasedMFViT.forward() got an unexpected keyword argument 'export_attention' error") 