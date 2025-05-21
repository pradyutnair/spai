import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple, Dict
from einops import rearrange

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
    Fusion module for combining semantic and spectral features
    """
    def __init__(
        self,
        semantic_dim=768,
        spectral_dim=1024,
        fusion_dim=1024,
        num_heads=8,
        dropout=0.1,
        use_layer_norm=True  # Make this parameter optional with a default value
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.spectral_dim = spectral_dim
        self.fusion_dim = fusion_dim
        
        # Linear projections with proper initialization
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim, bias=True)
        self.spectral_proj = nn.Linear(spectral_dim, fusion_dim, bias=True)
        
        # Cross-attention with proper initialization
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, 
            num_heads=num_heads,
            dropout=dropout,
            bias=True,
            batch_first=True
        )
        
        # Normalization layers for stability (optional)
        if use_layer_norm:
            self.semantic_norm = nn.LayerNorm(fusion_dim)
            self.spectral_norm = nn.LayerNorm(fusion_dim)
            self.cross_attn_norm = nn.LayerNorm(fusion_dim)
        else:
            self.semantic_norm = nn.Identity()
            self.spectral_norm = nn.Identity()
            self.cross_attn_norm = nn.Identity()
        
        # Final projection with normalization
        self.final_proj = nn.Sequential(
            nn.LayerNorm(fusion_dim) if use_layer_norm else nn.Identity(),
            nn.Linear(fusion_dim, fusion_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Trainable weights for modality importance
        self.modality_weights = nn.Parameter(torch.ones(2))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with values that prevent gradient explosion"""
        nn.init.xavier_uniform_(self.semantic_proj.weight, gain=0.02)
        nn.init.xavier_uniform_(self.spectral_proj.weight, gain=0.02)
        
        if self.semantic_proj.bias is not None:
            nn.init.constant_(self.semantic_proj.bias, 0.01)
        if self.spectral_proj.bias is not None:
            nn.init.constant_(self.spectral_proj.bias, 0.01)

    def forward(self, spectral_features, semantic_features):
        """
        Forward pass for combining spectral and semantic features.
        spectral_features: (B, L_spec, D_spec) or (B, D_spec)
        semantic_features: (B, L_sem, D_sem) or (B, D_sem)
        """
        batch_size_spec = spectral_features.shape[0]
        batch_size_sem = semantic_features.shape[0]

        if batch_size_spec != batch_size_sem:
            raise ValueError(
                f"Batch size mismatch: spectral_features ({batch_size_spec}) vs semantic_features ({batch_size_sem})"
            )
        batch_size = batch_size_spec

        # Commented out debug prints, uncomment if needed
        # print(f"Input shapes - Spectral: {spectral_features.shape}, Semantic: {semantic_features.shape}")
        # print(f"Configured dimensions - Spectral: {self.spectral_dim}, Semantic: {self.semantic_dim}, Fusion: {self.fusion_dim}")

        spectral_features = torch.nan_to_num(spectral_features, nan=0.0)
        semantic_features = torch.nan_to_num(semantic_features, nan=0.0)

        # Store original sequence lengths and reshape for projection if 3D
        spec_is_3d = len(spectral_features.shape) == 3
        sem_is_3d = len(semantic_features.shape) == 3

        spec_seq_len = spectral_features.shape[1] if spec_is_3d else 1
        sem_seq_len = semantic_features.shape[1] if sem_is_3d else 1

        if spec_is_3d:
            spectral_features_proj_input = spectral_features.reshape(batch_size * spec_seq_len, -1)
        else:
            spectral_features_proj_input = spectral_features

        if sem_is_3d:
            semantic_features_proj_input = semantic_features.reshape(batch_size * sem_seq_len, -1)
        else:
            semantic_features_proj_input = semantic_features
            
        # Project features
        spectral_proj_flat = self.spectral_proj(spectral_features_proj_input)
        semantic_proj_flat = self.semantic_proj(semantic_features_proj_input)

        # Reshape back to 3D for attention
        if spec_is_3d:
            spectral_proj = spectral_proj_flat.reshape(batch_size, spec_seq_len, self.fusion_dim)
        else:
            spectral_proj = spectral_proj_flat.unsqueeze(1) # Unsqueeze to (B, 1, fusion_dim)
        
        if sem_is_3d:
            semantic_proj = semantic_proj_flat.reshape(batch_size, sem_seq_len, self.fusion_dim)
        else:
            semantic_proj = semantic_proj_flat.unsqueeze(1) # Unsqueeze to (B, 1, fusion_dim)

        # Normalize projected features
        spectral_proj = self.spectral_norm(spectral_proj)
        semantic_proj = self.semantic_norm(semantic_proj)
        
        # Cross-attention: semantic queries spectral
        attn_output, _ = self.cross_attention(
            query=semantic_proj, # (B, L_sem, fusion_dim)
            key=spectral_proj,   # (B, L_spec, fusion_dim)
            value=spectral_proj  # (B, L_spec, fusion_dim)
        ) # attn_output: (B, L_sem, fusion_dim)
        
        # Residual connection
        fused_features = semantic_proj + attn_output # (B, L_sem, fusion_dim)
        
        # Normalize and final project
        fused_features = self.cross_attn_norm(fused_features)
        fused_features = self.final_proj(fused_features)
        
        if torch.isnan(fused_features).any():
            # print("WARNING: NaN detected in fusion output!")
            fused_features = torch.nan_to_num(fused_features, nan=0.0)
            
        # Modality weighting
        norm_weights = F.softmax(self.modality_weights, dim=0)
        
        # Weighted combination of fused (semantic-queried-spectral) and original semantic projection
        output = norm_weights[0] * fused_features + norm_weights[1] * semantic_proj
        
        return output


class AdaptiveSemanticSpectralFusion(nn.Module):
    """
    Enhanced fusion module for combining semantic and spectral features with:
    1. Bidirectional cross-attention mechanisms
    2. Frequency-aware attention gates
    3. Multi-level feature fusion
    4. Visualizable attention weights
    5. Adaptive frequency band contribution weighting
    """
    def __init__(
        self,
        semantic_dim=768,
        spectral_dim=1024,
        fusion_dim=1024,
        num_heads=8,
        dropout=0.1,
        ffn_ratio=4,
        num_frequency_bands=3  # Low, medium, high frequency bands
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.spectral_dim = spectral_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.num_frequency_bands = num_frequency_bands
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        self.last_frequency_weights = None
        self.last_modality_contribution = None
        
        # Linear projections for different inputs
        self.semantic_proj = nn.Linear(semantic_dim, fusion_dim)
        self.spectral_proj = nn.Linear(spectral_dim, fusion_dim)
        
        # Frequency band-specific projections
        self.frequency_band_projections = nn.ModuleList([
            nn.Linear(fusion_dim, fusion_dim) 
            for _ in range(num_frequency_bands)
        ])
        
        # Bidirectional cross-attention
        # 1. Semantic attends to spectral (semantic queries)
        self.sem_to_spec_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 2. Spectral attends to semantic (spectral queries)
        self.spec_to_sem_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Frequency band adaptive weighting with temperature
        self.frequency_band_weights = nn.Parameter(torch.ones(num_frequency_bands))
        self.frequency_temperature = nn.Parameter(torch.tensor(1.0))
        
        # Modality importance gates (learnable)
        self.early_fusion_gates = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # FFN for each modality after attention
        self.semantic_ffn = self._build_ffn(fusion_dim, ffn_ratio, dropout)
        self.spectral_ffn = self._build_ffn(fusion_dim, ffn_ratio, dropout)
        
        # Layer norms
        self.semantic_norm1 = nn.LayerNorm(fusion_dim)
        self.semantic_norm2 = nn.LayerNorm(fusion_dim)
        self.spectral_norm1 = nn.LayerNorm(fusion_dim)
        self.spectral_norm2 = nn.LayerNorm(fusion_dim)
        self.cross_attention_norm = nn.LayerNorm(fusion_dim)
        self.final_norm = nn.LayerNorm(fusion_dim)
        
        # Final adaptive fusion with contributions from both modalities
        self.modality_contribution = nn.Parameter(torch.ones(2))
        
        # Final fusion FFN
        self.final_fusion = self._build_ffn(fusion_dim, ffn_ratio, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _build_ffn(self, dim, ffn_ratio, dropout):
        """Helper to build a feed-forward network with residual connection"""
        return nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
            nn.Dropout(dropout)
        )
    
    def _init_weights(self):
        """Initialize weights with values that prevent gradient explosion"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.02)
        
        # Initialize frequency band weights with bias toward mid-frequencies
        with torch.no_grad():
            if self.num_frequency_bands >= 3:
                # Initialize with higher weight for mid frequencies
                mid_idx = self.num_frequency_bands // 2
                self.frequency_band_weights[mid_idx] = 1.5
    
    def get_last_attention_weights(self):
        """Return the last computed attention weights for visualization"""
        return {
            'attention_map': self.last_attention_weights,
            'frequency_weights': self.last_frequency_weights,
            'modality_contribution': self.last_modality_contribution
        }
    
    def _apply_frequency_band_attention(self, features):
        """Apply frequency band-specific attention and weighting"""
        batch_size, seq_len, dim = features.shape
        band_features = []
        
        for i, band_proj in enumerate(self.frequency_band_projections):
            # Each projection emphasizes different frequency characteristics
            band_feat = band_proj(features)
            band_features.append(band_feat)
        
        # Apply temperature-scaled softmax to band weights
        temp_scaled_weights = self.frequency_band_weights / self.frequency_temperature
        frequency_weights = F.softmax(temp_scaled_weights, dim=0)
        self.last_frequency_weights = frequency_weights.detach().cpu()
        
        # Weighted combination of band features
        weighted_features = torch.zeros_like(features)
        for i, band_feat in enumerate(band_features):
            weighted_features += frequency_weights[i] * band_feat
            
        return weighted_features
    
    def forward(self, spectral_features, semantic_features):
        """
        Forward pass for enhanced semantic-spectral fusion.
        Arguments:
          spectral_features: (B, L_spec, D_spec) or (B, D_spec)
          semantic_features: (B, L_sem, D_sem) or (B, D_sem)
        Returns:
          output: (B, L_sem, fusion_dim)
        """
        # Handle NaN values
        spectral_features = torch.nan_to_num(spectral_features, nan=0.0)
        semantic_features = torch.nan_to_num(semantic_features, nan=0.0)
        
        # Get batch sizes and ensure they match
        batch_size_spec = spectral_features.shape[0]
        batch_size_sem = semantic_features.shape[0]
        
        if batch_size_spec != batch_size_sem:
            raise ValueError(
                f"Batch size mismatch: spectral_features ({batch_size_spec}) vs semantic_features ({batch_size_sem})"
            )
        batch_size = batch_size_spec
        
        # Handle different input shapes and project to fusion_dim
        # 1. Handle spectral features
        spec_is_3d = len(spectral_features.shape) == 3
        if not spec_is_3d:
            # If 2D, add sequence dimension: [B, D] -> [B, 1, D]
            spectral_features = spectral_features.unsqueeze(1)
        
        # 2. Handle semantic features
        sem_is_3d = len(semantic_features.shape) == 3
        if not sem_is_3d:
            # If 2D, add sequence dimension: [B, D] -> [B, 1, D]
            semantic_features = semantic_features.unsqueeze(1)
        
        # Project to common dimension
        spectral_proj = self.spectral_proj(spectral_features)  # [B, L_spec, fusion_dim]
        semantic_proj = self.semantic_proj(semantic_features)  # [B, L_sem, fusion_dim]
        
        # Apply frequency band attention to spectral features
        spectral_freq_weighted = self._apply_frequency_band_attention(spectral_proj)
        
        # Apply layer norms
        spec_norm = self.spectral_norm1(spectral_freq_weighted)
        sem_norm = self.semantic_norm1(semantic_proj)
        
        # Bidirectional cross-attention
        # 1. Semantic features attend to spectral features
        sem_queries_spec, sem_to_spec_attn = self.sem_to_spec_attention(
            query=sem_norm,
            key=spec_norm,
            value=spec_norm,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Store attention weights for visualization
        self.last_attention_weights = sem_to_spec_attn.detach().cpu()
        
        # 2. Spectral features attend to semantic features
        spec_queries_sem, _ = self.spec_to_sem_attention(
            query=spec_norm,
            key=sem_norm,
            value=sem_norm,
            need_weights=False
        )
        
        # Apply residual connections and layer norms
        semantic_attended = semantic_proj + sem_queries_spec
        spectral_attended = spectral_freq_weighted + spec_queries_sem
        
        semantic_attended = self.semantic_norm2(semantic_attended)
        spectral_attended = self.spectral_norm2(spectral_attended)
        
        # Apply FFNs with residual connections
        semantic_output = semantic_attended + self.semantic_ffn(semantic_attended)
        spectral_output = spectral_attended + self.spectral_ffn(spectral_attended)
        
        # Early fusion - content-based adaptive weighting
        # Compute importance weights for both modalities based on content
        semantic_global = semantic_output.mean(dim=1, keepdim=True)  # [B, 1, fusion_dim]
        spectral_global = spectral_output.mean(dim=1, keepdim=True)  # [B, 1, fusion_dim]
        
        # Concatenate global representations for gate computation
        global_concat = torch.cat([semantic_global, spectral_global], dim=-1)  # [B, 1, fusion_dim*2]
        gates = self.early_fusion_gates(global_concat)  # [B, 1, 2]
        
        # Apply adaptive weighting
        semantic_contribution = gates[:, :, 0].unsqueeze(-1) * semantic_output  # [B, L_sem, fusion_dim]
        
        # Match sequence length of spectral to semantic for addition
        # This handles cases where sequence lengths differ
        if spectral_output.size(1) != semantic_output.size(1):
            # If shapes don't match, use mean pooling across sequence dimension
            spectral_pooled = spectral_output.mean(dim=1, keepdim=True)  # [B, 1, fusion_dim]
            # Expand to match semantic sequence length
            spectral_output_matched = spectral_pooled.expand(-1, semantic_output.size(1), -1)
        else:
            spectral_output_matched = spectral_output
        
        spectral_contribution = gates[:, :, 1].unsqueeze(-1) * spectral_output_matched  # [B, L_sem, fusion_dim]
        
        # Combine contributions
        fused = semantic_contribution + spectral_contribution
        fused = self.cross_attention_norm(fused)
        
        # Final fusion with learned modality contribution
        # Get a scalar for the contribution of each modality
        mod_weights = F.softmax(self.modality_contribution, dim=0)
        self.last_modality_contribution = mod_weights.detach().cpu()
        
        # Final weighted combination with FFN
        final_output = mod_weights[0] * semantic_output + mod_weights[1] * spectral_output_matched
        final_output = self.final_norm(final_output)
        final_output = final_output + self.final_fusion(final_output)
        
        # Check for NaNs in the output
        if torch.isnan(final_output).any():
            final_output = torch.nan_to_num(final_output, nan=0.0)
            
        return final_output
