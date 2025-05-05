import torch
import torch.nn as nn
import open_clip
from typing import Optional, Union, Tuple



class SemanticPipeline(nn.Module):
    """
    Semantic understanding pipeline that extracts rich semantic features from images
    using a ConvNeXt-XXL backbone.
    """
    
    def __init__(
        self, 
        convnext_path: Optional[str] = None,
        output_dim: int = 1096,
        freeze_backbone: bool = True
    ):
        """
        Initialize the semantic pipeline.
        
        Args:
            convnext_path: Path to pretrained ConvNeXt-XXL weights
            output_dim: Dimension of output semantic features
            freeze_backbone: Whether to freeze the ConvNeXt backbone
        """
        super(SemanticPipeline, self).__init__()
        
        print("Building semantic understanding pipeline with ConvNeXt-XXL")
        
        # Create the ConvNeXt model
        self.convnext_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg"  # or other available OpenCLIP weights
        )

        # Extract just the visual backbone
        self.backbone = self.convnext_model.visual.trunk
        
        # Replace pooling with identity to access the feature maps
        self.backbone.head.global_pool = nn.Identity()
        self.backbone.head.flatten = nn.Identity()
        
        # Add a global pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # This matches the implementation in AIDE_Model
        self.convnext_proj = nn.Sequential(
            nn.Linear(3072, output_dim),
        )
        
        # Freeze the backbone if specified
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Register normalization parameters as buffers
        self.register_buffer('clip_mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1))
        self.register_buffer('clip_std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1))
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input images to match ConvNeXt expectations.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Normalized tensor
        """
        return x * (self.imagenet_std / self.clip_std) + (self.imagenet_mean - self.clip_mean) / self.clip_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract semantic features from input images.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Semantic features of shape [B, output_dim]
        """
        # Apply normalization for ConvNeXt
        x = self.normalize_input(x)
        
        # Extract features - use no_grad if backbone is frozen
        if not any(p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                features = self.backbone(x)  # [B, 3072, H/32, W/32]
                # Verify shape matches AIDE implementation
                assert features.size()[1] == 3072, f"Expected 3072 channels, got {features.size()[1]}"
        else:
            features = self.backbone(x)
        
        # Apply global pooling and reshape
        pooled_features = self.global_pool(features).view(x.size(0), -1)  # [B, 3072]
        
        # Project to lower dimension using the projection layer
        semantic_features = self.convnext_proj(pooled_features)  # [B, output_dim]
        
        return semantic_features

    def is_pretrained(self):
        """
        Check if the model is using pretrained weights.
        """
        # This is a simple check - we could make this more sophisticated
        return hasattr(self, 'convnext_model') and hasattr(self.convnext_model, '_pretrained')


def build_semantic_pipeline(
    convnext_path: Optional[str] = None,
    output_dim: int = 1096
) -> SemanticPipeline:
    """
    Build and return the semantic pipeline module.
    
    Args:
        convnext_path: Path to pretrained ConvNeXt weights
        output_dim: Dimension of output semantic features
        
    Returns:
        Initialized semantic pipeline
    """
    return SemanticPipeline(
        convnext_path=convnext_path,
        output_dim=output_dim,
        freeze_backbone=True
    )



if __name__ == "__main__":
    # Example usage
    import os
    from PIL import Image
    import torchvision.transforms as transforms
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Specify paths
    image_path = "/home/scur2605/spai/images/1966.png"  

    # 2. Create semantic pipeline with weights
    semantic_model = build_semantic_pipeline(
        convnext_path=None,
        output_dim=2048
    ).to(device)
    
    # 3. Load and preprocess the image
    if os.path.exists(image_path):
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Create transform pipeline
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        
        # Transform image and add batch dimension
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        # Process the image
        with torch.no_grad():
            features = semantic_model(img_tensor)
        
        print(f"Processed image: {image_path}")
        print(f"Image features shape: {features.shape}")
        print(f"First few feature values: {features[0, :5]}")
    else:
        print(f"Image not found: {image_path}")
        
        # Process a batch of random images as fallback
        batch = torch.randn(8, 3, 224, 224).to(device)
        features = semantic_model(batch)
        print(f"Random batch features shape: {features.shape}")

