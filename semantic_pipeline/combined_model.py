import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Import your semantic pipeline
from semantic import SemanticPipeline, build_semantic_pipeline
class CombinedModel(nn.Module):
    """
    Combined model that fuses SPAI and the Semantic Pipeline architectures.
    
    This model:
    1. Uses a pretrained SPAI model for spectral features
    2. Uses the SemanticPipeline for semantic features
    3. Fuses these features with an MLP for the final classification
    """
    
    def __init__(
        self,
        spai_model_path: str,
        semantic_output_dim: int = 1096,
        hidden_dims: List[int] = [1024, 512],
        output_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize the combined model.
        
        Args:
            spai_model_path: Path to the pretrained SPAI model
            semantic_output_dim: Output dimension for the semantic features
            hidden_dims: Dimensions of hidden layers in the fusion MLP
            output_classes: Number of output classes (2 for binary classification)
            dropout: Dropout rate for the fusion MLP
        """
        super(CombinedModel, self).__init__()
        
        # 1. Load and freeze the SPAI model
        # Import here to avoid circular imports
        from spai.models.build import build_cls_model
        from spai.config import get_config


        
        # Build the SPAI model using your project's build function
        cfg = get_config({"cfg": "configs/spai.yaml"})
        self.spai_model = build_cls_model(cfg)
        print(f"SPAI model type: {type(self.spai_model).__name__}")
        
        # Load weights
        if spai_model_path and Path(spai_model_path).exists():
            checkpoint = torch.load(spai_model_path, map_location='cpu', weights_only=False)
            
            # Check if this is a training checkpoint or just model weights
            if "model" in checkpoint:
                # It's a training checkpoint, extract just the model weights
                print("Loading model from training checkpoint")
                self.spai_model.load_state_dict(checkpoint["model"])
            elif isinstance(checkpoint, dict) and not any(k in checkpoint for k in ["optimizer", "epoch"]):
                # It seems to be just model weights
                print("Loading model weights directly")
                self.spai_model.load_state_dict(checkpoint)
            else:
                print("WARNING: Could not load SPAI model weights. Proceeding with initialized model.")
        # Freeze SPAI model
        self.spai_model.eval()
        for param in self.spai_model.parameters():
            param.requires_grad = False
        
        # 2. Create and partially freeze the semantic pipeline
        self.semantic_model = build_semantic_pipeline(
            convnext_path=None,  # Use default pretrained weights
            output_dim=semantic_output_dim
        )
        
        # Ensure ConvNeXt backbone is frozen (projection layer remains trainable)
        self.semantic_model.backbone.eval()
        for param in self.semantic_model.backbone.parameters():
            param.requires_grad = False
        
        # Make sure projection layer is trainable
        for param in self.semantic_model.convnext_proj.parameters():
            param.requires_grad = True
            
        # 3. Create a fusion MLP
        # SPAI features + Semantic features 
        # Note: Get the actual feature dimension from SPAI model if possible
        spai_feature_dim = 1096  # This might need to be adjusted based on your SPAI implementation
        combined_dim = spai_feature_dim + semantic_output_dim
        
        # Build MLP layers
        layers = []
        input_dim = combined_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(input_dim, output_classes))
        
        self.fusion_mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the combined model.
        
        Args:
            x: Input tensor [B, C, H, W] or [B, T, C, H, W]
                    
        Returns:
            Classification logits
        """
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()
            
        # Handle both 4D and 5D inputs 
        if x.dim() == 5:  # [B, T, C, H, W]
            b, t, c, h, w = x.shape
            # Extract RGB image for semantic pipeline
            tokens = x[:, -1]  
        else:  # [B, C, H, W]
            tokens = x  # Just use the input directly
        
        # 1. Get SPAI features
        with torch.no_grad():
            spai_features = self._extract_spai_features(x)
            # Ensure features are float32
            if spai_features.dtype != torch.float32:
                spai_features = spai_features.float()
        
        # 2. Get semantic features
        semantic_features = self.semantic_model(tokens)
        # Ensure features are float32
        if semantic_features.dtype != torch.float32:
            semantic_features = semantic_features.float()
        
        # 3. Concatenate features
        combined_features = torch.cat([spai_features, semantic_features], dim=1)
        
        # 4. Process through MLP
        logits = self.fusion_mlp(combined_features)
        
        # Ensure output is float32
        if logits.dtype != torch.float32:
            logits = logits.float()
        
        return logits
    
    def _extract_spai_features(self, x):
        """Extract features from SPAI model before classification."""
        feature_store = []
        
        def hook_fn(module, input, output):
            # Ensure the stored feature is float32
            if output.dtype != torch.float32:
                feature_store.append(output.detach().float())
            else:
                feature_store.append(output.detach())
        
        # Check what type of model we have
        if hasattr(self.spai_model, 'patches_attention'):
            hook_handle = self.spai_model.norm.register_forward_hook(hook_fn)
            
            # Ensure input is float32
            spai_input = x[:, 0] if x.dim() > 4 else x
            if spai_input.dtype != torch.float32:
                spai_input = spai_input.float()
            
            with torch.no_grad():
                _ = self.spai_model(spai_input)
                
            hook_handle.remove()
        else:
            # Similar approach for MFViT
            hook_handle = self.spai_model.features_processor.register_forward_hook(hook_fn)
            
            # Ensure input is float32
            spai_input = x[:, 0] if x.dim() > 4 else x
            if spai_input.dtype != torch.float32:
                spai_input = spai_input.float()
            
            with torch.no_grad():
                _ = self.spai_model(spai_input)
                
            hook_handle.remove()
        
        # Final check to ensure we're returning float32
        if not feature_store or feature_store[0].dtype != torch.float32:
            return feature_store[0].float() if feature_store else torch.zeros(1, dtype=torch.float32)
        return feature_store[0]
    
    def get_trainable_parameters(self):
        """Return only the parameters that should be trained."""
        # We want to train:
        # 1. The fusion MLP
        # 2. The projection layer in the semantic model
        
        # Make sure the backbone in semantic model is frozen
        # But NOT the projection layer
        for param in self.semantic_model.backbone.parameters():
            param.requires_grad = False
        
        # Ensure projection layer is trainable
        for param in self.semantic_model.convnext_proj.parameters():
            param.requires_grad = True
        
        # Collect all trainable parameters
        trainable_params = []
        
        # Add fusion MLP parameters
        trainable_params.extend(self.fusion_mlp.parameters())
        
        # Add semantic model projection layer parameters
        trainable_params.extend(self.semantic_model.convnext_proj.parameters())
        
        # Print parameter counts for debugging
        mlp_params = sum(p.numel() for p in self.fusion_mlp.parameters())
        proj_params = sum(p.numel() for p in self.semantic_model.convnext_proj.parameters())
        print(f"Training {mlp_params} parameters in fusion MLP")
        print(f"Training {proj_params} parameters in projection layer")
        print(f"Total trainable parameters: {mlp_params + proj_params}")
        
        return trainable_params

def build_combined_model(
    spai_path: str,
    semantic_output_dim: int = 1096,
    hidden_dims: List[int] = [1024, 512],
    output_classes: int = 2
):
    """Helper function to build the combined model"""
    return CombinedModel(
        spai_model_path=spai_path,
        semantic_output_dim=semantic_output_dim,
        hidden_dims=hidden_dims,
        output_classes=output_classes
    )







def test_feature_extraction(spai_model_path):
    """
    Test that feature extraction is working properly.
    
    Args:
        spai_model_path: Path to SPAI model weights
    """
    # Create combined model
    model = build_combined_model(spai_model_path)
    
    # Create dummy input
    dummy_input = torch.randn(2,5, 3, 224, 224)  # [B, C, H, W]
    
    # Extract features
    with torch.no_grad():
        # Run the full model
        output = model(dummy_input)
        
        # Also run only the feature extraction
        features = model._extract_spai_features(dummy_input)
    
    # Print shapes to verify
    print(f"SPAI features shape: {features.shape}")
    print(f"Semantic features shape: {model.semantic_model(dummy_input[:, 4]).shape}")
    print(f"Combined output shape: {output.shape}")
    
    # If your model is PatchBasedMFViT, check the norm layer output dimension
    if hasattr(model.spai_model, 'norm'):
        print(f"Expected feature dim from norm: {model.spai_model.norm.normalized_shape[0]}")
    
    # If your model is MFViT, we expect features from FrequencyRestorationEstimator
    else:
        fre = model.spai_model.features_processor
        if hasattr(fre, 'original_features_processor') and fre.original_features_processor is not None:
            if fre.disable_reconstruction_similarity:
                print(f"Expected features from FRE: {fre.original_features_processor.proj2[-2].out_features}")
            else:
                n_features = len(model.spai_model.vit.intermediate_layers)
                print(f"Expected features from FRE: {6 * n_features + fre.original_features_processor.proj2[-2].out_features}")
        else:
            n_features = len(model.spai_model.vit.intermediate_layers)
            print(f"Expected features from FRE: {6 * n_features}")
    
    return features, output

# Add this to your if __name__ == "__main__" block
if __name__ == "__main__":
    # Example usage
    spai_model_path = "/home/scur2605/spai/weights/spai.pth"
    combined_model = build_combined_model(spai_model_path)
    
    # Test feature extraction
    features, output = test_feature_extraction(spai_model_path)
    
    # Dummy input tensor
    dummy_input = torch.randn(8, 5, 3, 224, 224)  # [B, T, C, H, W]
    
    # Forward pass
    output = combined_model(dummy_input)
    
    print("Output shape:", output.shape)  # Should be [B, output_classes]
    


    
