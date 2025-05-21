import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union

# Utility for user selection
FUSION_TYPES = ['concat', 'gated', 'attention']
class MultiScaleSemanticFusion(nn.Module):
    """
    Enhanced semantic fusion module that incorporates multi-scale features and frequency-specific attention.
    """
    def __init__(
        self,
        patch_dim: int,
        semantic_dim: int,
        num_scales: int = 3,
        use_frequency_attention: bool = True
    ):
        super().__init__()
        self.patch_dim = patch_dim
        self.semantic_dim = semantic_dim
        self.num_scales = num_scales
        self.use_frequency_attention = use_frequency_attention

        # Multi-scale feature projection
        self.scale_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_dim, semantic_dim),
                nn.LayerNorm(semantic_dim),
                nn.GELU()
            ) for _ in range(num_scales)
        ])

        # Frequency-specific attention
        if use_frequency_attention:
            self.freq_attention = nn.MultiheadAttention(
                embed_dim=semantic_dim,
                num_heads=4,
                batch_first=True
            )
            self.freq_gate = nn.Sequential(
                nn.Linear(semantic_dim * 2, semantic_dim),
                nn.Sigmoid()
            )

        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(semantic_dim * (num_scales + 1), semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, patch_dim)
        )

    def forward(
        self,
        patch_feats: torch.Tensor,  # B x N x L x D
        semantic_vec: torch.Tensor,  # B x semantic_dim
        freq_features: Optional[List[torch.Tensor]] = None  # List of frequency-specific features
    ) -> torch.Tensor:
        B, N, L, D = patch_feats.shape
        
        # Project semantic vector to match patch features
        semantic_vec = semantic_vec.unsqueeze(1).unsqueeze(2).expand(-1, N, L, -1)
        
        # Multi-scale feature processing
        scale_features = []
        for i, projector in enumerate(self.scale_projectors):
            scale_feat = projector(patch_feats)
            scale_features.append(scale_feat)
        
        # Frequency-specific attention if enabled
        if self.use_frequency_attention and freq_features is not None:
            freq_attn = []
            for freq_feat in freq_features:
                # Reshape for attention
                freq_feat = freq_feat.reshape(B*N, L, -1)
                sem_vec = semantic_vec.reshape(B*N, L, -1)
                
                # Compute attention
                attn_out, _ = self.freq_attention(freq_feat, sem_vec, sem_vec)
                freq_attn.append(attn_out.reshape(B, N, L, -1))
            
            # Combine frequency attention
            freq_attn = torch.stack(freq_attn, dim=-1).mean(dim=-1)
            
            # Gate mechanism
            gate = self.freq_gate(torch.cat([semantic_vec, freq_attn], dim=-1))
            semantic_vec = semantic_vec * gate + freq_attn * (1 - gate)
        
        # Combine all features
        combined = torch.cat([semantic_vec] + scale_features, dim=-1)
        
        # Final fusion
        fused = self.fusion(combined)
        
        return fused

# Keep existing SemanticFusionModule for backward compatibility
class SemanticFusionModule(nn.Module):
    """
    Modular fusion module for combining patch-level features with a global semantic context vector.
    Supports: 'concat', 'gated', 'attention'.
    """
    def __init__(self, patch_dim: int, semantic_dim: int, fusion_type: str = 'concat'):
        super().__init__()
        self.fusion_type = fusion_type.lower()
        self.patch_dim = patch_dim
        self.semantic_dim = semantic_dim

        if self.fusion_type == 'concat':
            self.out_dim = patch_dim + semantic_dim
        elif self.fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(patch_dim + semantic_dim, patch_dim),
                nn.Sigmoid()
            )
            self.out_dim = patch_dim
        elif self.fusion_type == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=patch_dim, num_heads=1, batch_first=True)
            self.out_dim = patch_dim
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, patch_feats: torch.Tensor, semantic_vec: torch.Tensor) -> torch.Tensor:
        """
        patch_feats: (B, N, L, D)
        semantic_vec: (B, semantic_dim)
        Returns: (B, N, L, out_dim)
        """
        B, N, L, D = patch_feats.shape
        if self.fusion_type == 'concat':
            sem = semantic_vec.unsqueeze(1).unsqueeze(2).expand(-1, N, L, -1)
            fused = torch.cat([patch_feats, sem], dim=-1)
        elif self.fusion_type == 'gated':
            sem = semantic_vec.unsqueeze(1).unsqueeze(2).expand(-1, N, L, -1)
            gate_input = torch.cat([patch_feats, sem], dim=-1)
            gate = self.gate(gate_input)
            fused = patch_feats * gate
        elif self.fusion_type == 'attention':
            # Reshape for attention: (B*N, L, D), semantic as (B*N, 1, D)
            x = patch_feats.reshape(B*N, L, D)
            sem = semantic_vec.unsqueeze(1).expand(-1, N, -1).reshape(B*N, 1, self.semantic_dim)
            # Project semantic to patch_dim if needed
            if self.semantic_dim != D:
                sem_proj = nn.Linear(self.semantic_dim, D).to(semantic_vec.device)
                sem = sem_proj(sem)
            attn_out, _ = self.attn(x, sem, sem)
            fused = attn_out.reshape(B, N, L, D)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        return fused

class SemanticSpectralFusion(nn.Module):
    """
    Fuses semantic and spectral features using cross-attention.
    
    Expects:
    - spectral_features: [batch_size, seq_len, spectral_dim]
    - semantic_features: [batch_size, seq_len, semantic_dim]
    
    Returns:
    - fused_features: [batch_size, fusion_dim]
    """
    def __init__(self, semantic_dim=768, spectral_dim=1024, fusion_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.spectral_dim = spectral_dim
        self.fusion_dim = fusion_dim
        
        # Projection layers to align dimensions
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
        self.spectral_proj = nn.Linear(spectral_dim, fusion_dim)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Important: use batch_first=True for [B, S, D] format
        )
        
        # Optional: learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(2))
        
        # Final projection layer 
        self.final_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, spectral_features, semantic_features):
        """
        Forward pass for semantic-spectral fusion
        
        Args:
            spectral_features (torch.Tensor): Shape [B, S, D_spec] or [B, D_spec]
            semantic_features (torch.Tensor): Shape [B, S, D_sem] or [B, D_sem]
            
        Returns:
            torch.Tensor: Fused features with shape [B, D_fusion]
        """
        # Handle 2D inputs (no sequence dimension)
        if len(spectral_features.shape) == 2:
            spectral_features = spectral_features.unsqueeze(1)  # [B, 1, D]
        if len(semantic_features.shape) == 2:
            semantic_features = semantic_features.unsqueeze(1)  # [B, 1, D]
            
        # Make sure we have 3D tensors: [batch_size, seq_len, dim]
        assert len(spectral_features.shape) == 3, f"Spectral features must be 3D, got shape {spectral_features.shape}"
        assert len(semantic_features.shape) == 3, f"Semantic features must be 3D, got shape {semantic_features.shape}"
        
        # Project features to common dimension
        spectral_proj = self.spectral_proj(spectral_features)  # [B, S, D_fusion]
        semantic_proj = self.semantic_proj(semantic_features)  # [B, S, D_fusion]
        
        # Cross attention: spectral attends to semantic
        # For MultiheadAttention with batch_first=True:
        # - query: [B, T, D] - target sequence
        # - key, value: [B, S, D] - source sequence
        attn_output, attn_weights = self.cross_attention(
            query=spectral_proj,       # [B, S_q, D]
            key=semantic_proj,         # [B, S_k, D]
            value=semantic_proj,       # [B, S_v, D]
            need_weights=True,
            average_attn_weights=True
        )
        
        # Weighted fusion
        weights = F.softmax(self.modality_weights, dim=0)
        fused = weights[0] * spectral_proj + weights[1] * attn_output
        
        # Apply final projection
        fused = self.final_proj(fused)
        
        # Average pooling along sequence dimension if needed
        if fused.size(1) > 1:
            fused = fused.mean(dim=1)  # [B, D]
        else:
            fused = fused.squeeze(1)   # [B, D]
            
        return fused
