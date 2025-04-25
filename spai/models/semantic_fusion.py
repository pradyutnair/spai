import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

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
