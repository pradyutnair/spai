import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

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

# Utility for user selection
FUSION_TYPES = ['concat', 'gated', 'attention']
