# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and improving AUC performance.
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N,) logits
        targets: (N,) binary targets
        """
        # Convert logits to probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SemanticConsistencyLoss(nn.Module):
    """
    Semantic consistency loss to ensure semantic features are consistent
    across different frequency components.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, semantic_orig, semantic_low, semantic_high):
        """
        semantic_orig: (B, D) semantic features from original image
        semantic_low: (B, D) semantic features from low frequency image  
        semantic_high: (B, D) semantic features from high frequency image
        """
        # Normalize features
        semantic_orig = F.normalize(semantic_orig, dim=-1)
        semantic_low = F.normalize(semantic_low, dim=-1)
        semantic_high = F.normalize(semantic_high, dim=-1)
        
        # Compute cosine similarities
        sim_orig_low = F.cosine_similarity(semantic_orig, semantic_low, dim=-1)
        sim_orig_high = F.cosine_similarity(semantic_orig, semantic_high, dim=-1)
        sim_low_high = F.cosine_similarity(semantic_low, semantic_high, dim=-1)
        
        # Encourage high similarity (minimize negative similarity)
        consistency_loss = -(sim_orig_low + sim_orig_high + sim_low_high).mean()
        
        return consistency_loss


class FrequencyConsistencyLoss(nn.Module):
    """
    Frequency consistency loss to ensure frequency domain features
    maintain semantic relationships.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, freq_features_orig, freq_features_low, freq_features_high):
        """
        freq_features_*: (B, N, L, D) frequency domain features
        """
        # Pool features across patches
        orig_pooled = freq_features_orig.mean(dim=(1, 2))  # (B, D)
        low_pooled = freq_features_low.mean(dim=(1, 2))    # (B, D)
        high_pooled = freq_features_high.mean(dim=(1, 2))  # (B, D)
        
        # Normalize
        orig_pooled = F.normalize(orig_pooled, dim=-1)
        low_pooled = F.normalize(low_pooled, dim=-1)
        high_pooled = F.normalize(high_pooled, dim=-1)
        
        # Compute similarities
        sim_orig_low = F.cosine_similarity(orig_pooled, low_pooled, dim=-1)
        sim_orig_high = F.cosine_similarity(orig_pooled, high_pooled, dim=-1)
        
        # Encourage consistency
        consistency_loss = -(sim_orig_low + sim_orig_high).mean()
        
        return consistency_loss


class EnhancedSemanticSpectralLoss(nn.Module):
    """
    Enhanced loss combining focal loss with semantic and frequency consistency.
    """
    def __init__(
        self, 
        focal_alpha=0.25, 
        focal_gamma=2.0,
        semantic_weight=0.1,
        frequency_weight=0.05
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.semantic_consistency = SemanticConsistencyLoss()
        self.frequency_consistency = FrequencyConsistencyLoss()
        self.semantic_weight = semantic_weight
        self.frequency_weight = frequency_weight
        
    def forward(self, predictions, targets, semantic_features=None, frequency_features=None):
        """
        predictions: (B,) model predictions
        targets: (B,) binary targets
        semantic_features: dict with 'orig', 'low', 'high' semantic features
        frequency_features: dict with 'orig', 'low', 'high' frequency features
        """
        # Primary focal loss
        focal_loss = self.focal_loss(predictions, targets)
        total_loss = focal_loss
        
        # Add semantic consistency if available
        if semantic_features is not None:
            semantic_loss = self.semantic_consistency(
                semantic_features['orig'],
                semantic_features['low'], 
                semantic_features['high']
            )
            total_loss += self.semantic_weight * semantic_loss
            
        # Add frequency consistency if available
        if frequency_features is not None:
            freq_loss = self.frequency_consistency(
                frequency_features['orig'],
                frequency_features['low'],
                frequency_features['high']
            )
            total_loss += self.frequency_weight * freq_loss
            
        return total_loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        normalize_features: bool = True
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.normalize_features: bool = normalize_features

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        if self.normalize_features:
            features = F.normalize(features, dim=2)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class BCESupConWithLogits(nn.Module):

    def __init__(
        self,
        supcon_alpha: float = 0.2,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature, contrast_mode, base_temperature)
        self.bce = nn.BCEWithLogitsLoss()
        self.supcon_alpha = supcon_alpha

    def forward(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        labels: torch.LongTensor = None,
        mask=None
    ) -> torch.Tensor:
        return (self.bce(predictions, labels)
                + self.supcon_alpha * self.supcon(features, labels, mask))


def build_loss(config) -> nn.Module:
    """Build loss function based on configuration."""
    
    if config.TRAIN.LOSS == "focal":
        # Enhanced focal loss for better AUC optimization
        focal_alpha = getattr(config.TRAIN, 'FOCAL_ALPHA', 0.25)
        focal_gamma = getattr(config.TRAIN, 'FOCAL_GAMMA', 2.0)
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
    elif config.TRAIN.LOSS == "enhanced_semantic_spectral":
        # Enhanced loss with semantic and frequency consistency
        focal_alpha = getattr(config.TRAIN, 'FOCAL_ALPHA', 0.25)
        focal_gamma = getattr(config.TRAIN, 'FOCAL_GAMMA', 2.0)
        semantic_weight = getattr(config.TRAIN, 'SEMANTIC_CONSISTENCY_WEIGHT', 0.1)
        frequency_weight = getattr(config.TRAIN, 'FREQUENCY_CONSISTENCY_WEIGHT', 0.05)
        criterion = EnhancedSemanticSpectralLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            semantic_weight=semantic_weight,
            frequency_weight=frequency_weight
        )
        
    elif config.TRAIN.LOSS == "bce_supcont":
       criterion = BCESupConWithLogits()
       
    elif config.TRAIN.LOSS == "bce":
        criterion = nn.BCEWithLogitsLoss()
        
    elif config.TRAIN.LOSS == "triplet":
        criterion = nn.TripletMarginLoss(margin=config.TRAIN.TRIPLET_LOSS_MARGIN)
        
    elif config.TRAIN.LOSS == "supcont":
        criterion = SupConLoss()
        
    else:
        raise RuntimeError(f"Unknown loss type: {config.TRAIN.LOSS}")

    return criterion
