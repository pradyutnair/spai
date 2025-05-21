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

import dataclasses
import pathlib
from typing import Optional, Union, List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import five_crop
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange
from PIL import Image

from . import vision_transformer
from . import filters
from . import utils
from . import backbones
from .semantic_fusion import SemanticSpectralFusion
from spai.utils import save_image_with_attention_overlay


class PatchBasedMFViT(nn.Module):
    def __init__(
        self,
        vit_backbone: Union[vision_transformer.VisionTransformer,
                       backbones.CLIPBackbone,
                       backbones.DINOv2Backbone,
                       'DINOv2FeatureEmbedding'],
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_patch_size: int,
        img_patch_stride: int,
        cls_vector_dim: int,
        attn_embed_dim: int,
        num_heads: int,
        dropout: float = .0,
        frozen_backbone: bool = True,
        minimum_patches: int = 0,
        initialization_scope: str = "all",
        mfvit_output_dim_actual: int = 0
    ) -> None:
        super().__init__()

        self.mfvit = MFViT(
            vit_backbone,
            features_processor,
            None,
            masking_radius,
            img_patch_size,
            frozen_backbone=frozen_backbone,
            initialization_scope=initialization_scope
        )

        self.img_patch_size: int = img_patch_size
        self.img_patch_stride: int = img_patch_stride
        self.minimum_patches: int = minimum_patches
        self.cls_vector_dim: int = cls_vector_dim
        self.attn_embed_dim: int = attn_embed_dim
        self.num_heads: int = num_heads
        self.mfvit_output_dim_actual: int = mfvit_output_dim_actual

        # Projector if mfvit output dim doesn't match this module's expected cls_vector_dim
        self.input_feature_projector = None
        if self.mfvit_output_dim_actual > 0 and self.mfvit_output_dim_actual != self.cls_vector_dim:
            print(f"✨ PatchBasedMFViT: Adding projector to map MFViT output {self.mfvit_output_dim_actual} -> expected {self.cls_vector_dim}")
            self.input_feature_projector = nn.Linear(self.mfvit_output_dim_actual, self.cls_vector_dim)

        # Cross-Attention with a learnable vector layers.
        dim_head: int = attn_embed_dim // num_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(cls_vector_dim, attn_embed_dim*2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros((num_heads, 1, attn_embed_dim//num_heads)))
        nn.init.trunc_normal_(self.patch_aggregator, std=.02)
        self.to_out = nn.Sequential(
            nn.Linear(attn_embed_dim, cls_vector_dim, bias=False),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(cls_vector_dim)
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            # Initialize only the newly added components, by excluding mfvit.
            for m_name, m in self._modules.items():
                if m_name == "mfvit" or m_name == "input_feature_projector":
                    continue
                else:
                    m.apply(_init_weights)
            # Initialize projector separately if it exists
            if self.input_feature_projector is not None:
                 self.input_feature_projector.apply(_init_weights)
        else:
            raise TypeError(f"Non-supported weight initialization type: {initialization_scope}")

    def forward(
        self,
        x: Union[torch.Tensor, list[torch.Tensor]],
        feature_extraction_batch_size: Optional[int] = None,
        export_dirs: Optional[list[pathlib.Path]] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list['AttentionMask']]]:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        :param feature_extraction_batch_size:
        :param export_dirs:
        """
        if isinstance(x, torch.Tensor):
            x =  self.forward_batch(x.float())
        elif isinstance(x, list):
            if feature_extraction_batch_size is None:
                feature_extraction_batch_size = len(x)
            if export_dirs is not None:
                x = self.forward_arbitrary_resolution_batch_with_export(
                    x, feature_extraction_batch_size, export_dirs
                )
            else:
                x = self.forward_arbitrary_resolution_batch(x, feature_extraction_batch_size)
        else:
            raise TypeError('x must be a tensor or a list of tensors')

        return x

    def patches_attention(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        drop_bottom: bool = False,
        drop_top: bool = False
    ) -> torch.Tensor:
        """Perform cross attention between a learnable vector and the patches of an image."""
        # Ensure x has the same dtype as the model weights (for AMP/mixed precision)
        x = x.to(self.to_kv.weight.dtype)
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), kv)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.to_out(x)
        x = x.squeeze(dim=1)
        if return_attn:
            return x, attn
        else:
            return x

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = utils.patchify_image(
            x,
            (self.img_patch_size, self.img_patch_size),
            (self.img_patch_stride, self.img_patch_stride)
        )  # B x L x C x H x W

        patch_features: list[torch.Tensor] = []
        for i in range(x.size(1)):
            features = self.mfvit(x[:, i].float())  
            patch_features.append(features)
        x = torch.stack(patch_features, dim=1)  # B x L x D
        del patch_features

        # Project features if dimensions mismatch before patches_attention
        if self.input_feature_projector is not None:
            x = x.to(self.input_feature_projector.weight.dtype)
            x = self.input_feature_projector(x)

        # Squeeze the middle dimension of size 1 to make x [B_orig, L, 1096]
        if x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)

        x = self.patches_attention(x)  # B x D
        x = self.norm(x)  # B x D
        x = self.cls_head(x)  # B x 1

        return x

    def forward_arbitrary_resolution_batch(
        self,
        x: list[torch.Tensor],
        feature_extraction_batch_size: int
    ) -> torch.Tensor:
        """Forward pass of a batch of images of different resolutions.

        Batch size on the tensors should equal one.

        :param x: list of 1 x C x H_i x W_i tensors, where i denote the i-th image in the list.
        :param feature_extraction_batch_size:

        :returns: A B x 1 tensor.
        """
        # Rearrange the patches from all images into a single tensor.
        patched_images: list[torch.Tensor] = []
        for img in x:
            patched: torch.Tensor = utils.patchify_image(
                img,
                (self.img_patch_size, self.img_patch_size),
                (self.img_patch_stride, self.img_patch_stride)
            )  # 1 x L_i x C x H x W
            if patched.size(1) < self.minimum_patches:
                patched: tuple[torch.Tensor, ...] = five_crop(
                    img, [self.img_patch_size, self.img_patch_size]
                )
                patched: torch.Tensor = torch.stack(patched, dim=1)
            patched_images.append(patched)
        x = patched_images
        del patched_images
        # x = [
        #     utils.patchify_image(
        #         img,
        #         (self.img_patch_size, self.img_patch_size),
        #         (self.img_patch_stride, self.img_patch_stride)
        #     )  # 1 x L_i x C x H x W
        #     for img in x
        # ]
        img_patches_num: list[int] = [img.size(1) for img in x]
        x = torch.cat(x, dim=1)  # 1 x SUM(L_i) x C x H x W
        x = x.squeeze(dim=0)  # SUM(L_i) x C x H x W

        # Process the patches in groups of feature_extraction_batch_size.
        features: list[torch.Tensor] = []
        for i in range(0, x.size(0), feature_extraction_batch_size):
            features.append(self.mfvit(x[i:i+feature_extraction_batch_size]))
        x = torch.cat(features, dim=0)  # SUM(L_i) x D
        del features

        # Project features if dimensions mismatch before patches_attention
        if self.input_feature_projector is not None:
            x = x.to(self.input_feature_projector.weight.dtype)
            x = self.input_feature_projector(x)

        # Attend to patches according to the image they belong to.
        attended: list[torch.Tensor] = []
        processed_sum: int = 0
        for i in img_patches_num:
            patch_group = x[processed_sum:processed_sum+i] # Shape: [num_patches_for_this_image, 1, ClsVectorDim]
            # Unsqueeze to add batch dimension for patches_attention
            patch_group_batched = patch_group.unsqueeze(0) # Shape: [1, num_patches_for_this_image, 1, ClsVectorDim]
            
            # Squeeze the redundant middle dimension (of size 1, originally from MFViT's sequence length)
            # before passing to patches_attention, which expects [Batch, NumPatches, FeatureDim]
            if patch_group_batched.dim() == 4 and patch_group_batched.shape[2] == 1:
                patch_group_batched_squeezed = patch_group_batched.squeeze(2) # Shape: [1, num_patches_for_this_image, ClsVectorDim]
            else:
                # This path should ideally not be taken if MFViT output and projector are consistent
                patch_group_batched_squeezed = patch_group_batched

            attended.append(self.patches_attention(patch_group_batched_squeezed))
            processed_sum += i
        x = torch.cat(attended, dim=0)  # B x D
        del attended

        x = self.norm(x)  # B x D
        x = self.cls_head(x)  # B x 1

        return x

    def forward_arbitrary_resolution_batch_with_export(
        self,
        x: list[torch.Tensor],
        feature_extraction_batch_size: int,
        export_dirs: list[pathlib.Path],
        export_image_patches: bool = False
    ) -> tuple[torch.Tensor, list['AttentionMask']]:
        """Forward passes any resolution images and exports their spectral context attention masks.

        The batch size of the tensors in the `x` list should be equal to 1, i.e. each
        tensor in the list should correspond to a single image.

        :param x: List of 1 x C x H_i x W_i tensors, where i denotes the i-th image in the list.
        :param feature_extraction_batch_size: The maximum number of image patches that will
            be processed under a single batch. It should be set to a value high-enough to fully
            utilize the accelerator used, and low-enough to not cause out-of-memory errors.
        :param export_dirs: A list of directories that will be used for exporting the
            spectral context attention masks of each image.
        :param export_image_patches: When this flag is set to True, each patch considered
            by the spectral context attention will be exported in a separate file. Beware
            that when there is overlap among the patches, or on very large images, the
            number of these patches could be very large.

        :returns: A tuple containing a B x 1 tensor, where B is the batch size, and a list
            of attention masks for each image in the batch.
        """
        predictions: list[torch.Tensor] = []
        attention_masks: list[AttentionMask] = []

        # Process each image in the batch, one by one, and export its corresponding
        # spectral context attention mask.
        for img, export_dir in zip(x, export_dirs):
            # Patchify the image.
            orig_height: int = img.size(2)
            orig_width: int = img.size(3)
            patched: torch.Tensor = utils.patchify_image(
                img,
                (self.img_patch_size, self.img_patch_size),
                (self.img_patch_stride, self.img_patch_stride)
            )  # 1 x L_i x C x H x W
            if patched.size(1) < self.minimum_patches:
                patched: tuple[torch.Tensor, ...] = five_crop(
                    img, [self.img_patch_size, self.img_patch_size]
                )
                patched: torch.Tensor = torch.stack(patched, dim=1)

            # Encode each patch and export it if requested.
            features: list[torch.Tensor] = []
            if export_image_patches:
                # Process the patches one by one and export them.
                for i in range(0, patched.size(1)):
                    export_file = export_dir / f"patch_{i}.png"
                    features.append(self.mfvit.forward_with_export(
                        patched[:, i], export_file=export_file
                    ))
            else:
                # Process the patches in groups of feature_extraction_batch_size.
                for i in range(0, patched.size(1), feature_extraction_batch_size):
                    features.append(self.mfvit(patched[0, i:i+feature_extraction_batch_size]))
            x = torch.cat(features, dim=0)  # SUM(L_i) x D
            del features

            # Project features if dimensions mismatch before patches_attention
            if self.input_feature_projector is not None:
                x = x.to(self.input_feature_projector.weight.dtype)
                x = self.input_feature_projector(x)

            # Attend to patches.
            x, attn = self.patches_attention(x.unsqueeze(0), return_attn=True)  # 1 x D, 1 x L_i
            patches_attn_dir: pathlib.Path = export_dir / f"patches_attn"
            patches_attn_dir.mkdir(exist_ok=True, parents=True)

            x = self.norm(x)  # 1 x D
            x = self.cls_head(x)  # 1 x 1

            # Export the spectral context attention mask.
            attn_list: list[float] = attn.detach().cpu().mean(dim=1).tolist()[0][0]
            if export_image_patches:
                for i in range(0, patched.size(1)):
                    export_file = patches_attn_dir / f"{attn_list[i]:.3f}_patch_{i}_.png"
                    Image.fromarray(
                        (patched[:, i].detach().cpu().permute(0, 2, 3, 1).squeeze(
                            dim=0).numpy() * 255).astype(
                            np.uint8)).save(export_file)
            attn_img_file = (patches_attn_dir
                                / f"attn_overlay_{F.sigmoid(x).detach().cpu().tolist()[0]}.png")
            attn_mask_file = (patches_attn_dir
                                / f"attn_mask_{F.sigmoid(x).detach().cpu().tolist()[0]}.png")
            attn_overlay_file = (
                patches_attn_dir
                    / f"attn_mask_colormap_{F.sigmoid(x).detach().cpu().tolist()[0]}.png"
            )
            save_image_with_attention_overlay(
                patched.detach().cpu(),
                attn_list,
                orig_height,
                orig_width,
                self.img_patch_size,
                self.img_patch_stride,
                attn_img_file,
                mask_path=attn_mask_file,
                overlay_path=attn_overlay_file
            )

            predictions.append(x)
            attention_masks.append(AttentionMask(mask=attn_mask_file,
                                                 overlay=attn_overlay_file,
                                                 overlayed_image=attn_img_file))
        x = torch.cat(predictions, dim=0)
        return x, attention_masks

    def get_vision_transformer(self) -> vision_transformer.VisionTransformer:
        return self.mfvit.get_vision_transformer()

    def unfreeze_backbone(self) -> None:
        self.mfvit.unfreeze_backbone()

    def freeze_backbone(self) -> None:
        self.mfvit.freeze_backbone()

    def export_onnx_patch_aggregator(self, export_file: pathlib.Path) -> None:
        # Export the spectral context attention and the classifier.
        outer_instance = self
        class SCAClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.outer_instance = outer_instance
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Input size: B x L x D"""
                x = self.outer_instance.patches_attention(x)  # B x D
                x = self.outer_instance.norm(x)  # B x D
                x = self.outer_instance.cls_head(x)  # B x 1
                return x
        model: SCAClassifier = SCAClassifier()
        x: torch.Tensor = torch.rand((3, 4, outer_instance.cls_vector_dim))
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)
        # onnx_options: torch.onnx.ExportOptions = torch.onnx.ExportOptions(dynamic_shapes=True)
        # onnx_program: torch.onnx.ONNXProgram = torch.onnx.dynamo_export(
        #     model, x, export_options=onnx_options
        # )
        batch_dim: torch.export.Dim = torch.export.Dim("batch_size")
        seq_dim: torch.export.Dim = torch.export.Dim("seq_dim")
        ep: torch.export.ExportedProgram = torch.export.export(
            model,
            args=(x,),
            dynamic_shapes={
                "x": {0: batch_dim, 1: seq_dim},
            }
        )
        onnx_program = torch.onnx.export(ep, dynamo=True, report=True, verify=True)
        onnx_program.save(str(export_file))

    def export_onnx(
        self,
        patch_encoder: pathlib.Path,
        patch_aggregator: pathlib.Path,
        include_fft_preprocessing: bool = True,
    ) -> None:
        if include_fft_preprocessing:
            self.mfvit.export_onnx(patch_encoder)
        else:
            self.mfvit.export_onnx_without_fft(patch_encoder)
        self.export_onnx_patch_aggregator(patch_aggregator)


class MFViT(nn.Module):
    """Model that constructs features according to the ability to restore missing frequencies."""
    def __init__(
        self,
        vit: Union[vision_transformer.VisionTransformer,
                   backbones.CLIPBackbone,
                   backbones.DINOv2Backbone,
                   'DINOv2FeatureEmbedding'],
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_size: int,
        frozen_backbone: bool = True,
        initialization_scope: str = "all"
    ):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.cls_head = cls_head

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            # Initialize only the newly added components, by excluding vit.
            for m_name, m in self._modules.items():
                if m_name == "vit":
                    continue
                else:
                    m.apply(_init_weights)
        else:
            raise TypeError(f"Non-supported weight initialization type: {initialization_scope}")

        self.frozen_backbone: bool = frozen_backbone

        self.frequencies_mask: nn.Parameter = nn.Parameter(
            filters.generate_circular_mask(img_size, masking_radius),
            requires_grad=False
        )

        # Initialize semantic-spectral fusion
        if isinstance(self.vit, (backbones.DINOv2Backbone, DINOv2FeatureEmbedding)):
            print("🚀 Using DINOv2 backbone for semantic features")
            self.semantic_fusion = SemanticSpectralFusion(
                spectral_dim=features_processor.proj_dim,
                semantic_dim=768,  # DINOv2 ViT-B/14 dimension
                fusion_dim=features_processor.proj_dim,
                num_heads=8,
                dropout=0.1
            )
        else:
            print("⚠️ Warning: Using non-DINOv2 backbone, semantic fusion disabled")
            self.semantic_fusion = None

        if (isinstance(self.vit, vision_transformer.VisionTransformer)
                or isinstance(self.vit, (backbones.DINOv2Backbone, DINOv2FeatureEmbedding))):
            # ImageNet normalization
            self.backbone_norm = transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
            )
        elif isinstance(self.vit, backbones.CLIPBackbone):
            # CLIP normalization
            self.backbone_norm = transforms.Normalize(
                mean=backbones.CLIP_MEAN, std=backbones.CLIP_STD
            )
        else:
            raise TypeError(f"Unsupported backbone type: {type(vit)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        """
        with torch.cuda.amp.autocast():
            # Spectral branch
            low_freq: torch.Tensor
            hi_freq: torch.Tensor
            low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

            low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
            hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

            # Normalize all components according to ImageNet.
            x_norm = self.backbone_norm(x)
            low_freq_norm = self.backbone_norm(low_freq)
            hi_freq_norm = self.backbone_norm(hi_freq)

            # Semantic branch
            semantic_vec = None
            if isinstance(self.vit, (backbones.DINOv2Backbone, DINOv2FeatureEmbedding)):
                semantic_vec = self.vit(x)

            if self.frozen_backbone:
                with torch.no_grad():
                    x_feats, low_feats, hi_feats = self._extract_features(
                        x_norm, low_freq_norm, hi_freq_norm
                    )
            else:
                x_feats, low_feats, hi_feats = self._extract_features(
                    x_norm, low_freq_norm, hi_freq_norm
                )

            spectral_features = self.features_processor(x_feats, low_feats, hi_feats)
            
            # Apply semantic-spectral fusion if semantic vector is available
            if semantic_vec is not None and hasattr(self, 'semantic_fusion'):
                # Always ensure semantic_vec is properly shaped for attention: [batch_size, seq_len, dim]
                if len(semantic_vec.shape) == 4:  # If shape is [B, 1, 1, D]
                    semantic_vec = semantic_vec.squeeze(2).squeeze(1)  # Convert to [B, D]
                
                if len(semantic_vec.shape) == 2:  # If shape is [B, D]
                    semantic_vec = semantic_vec.unsqueeze(1)  # Make it [B, 1, D] for attention
                
                # Ensure spectral_features is also 3D if needed: [batch_size, seq_len, dim]
                if len(spectral_features.shape) == 2:
                    # Reshape to [B, 1, D] format for attention
                    batch_size = spectral_features.size(0)
                    feat_dim = spectral_features.size(1)
                    
                    # Check if semantic fusion module expects a specific dimension
                    if hasattr(self.semantic_fusion, 'spectral_proj'):
                        spectral_dim = self.semantic_fusion.spectral_proj.in_features
                        
                        # Handle dimension mismatch with dynamic projection if needed
                        if feat_dim != spectral_dim:
                            temp_proj = nn.Linear(feat_dim, spectral_dim).to(spectral_features.device)
                            spectral_features = temp_proj(spectral_features)
                    
                    # Now reshape to 3D for attention
                    spectral_features = spectral_features.unsqueeze(1)  # [B, 1, D]
                
                semantic_dim = semantic_vec.size(-1)
                spectral_dim = spectral_features.size(-1)
                fusion_dim = getattr(self.semantic_fusion, 'fusion_dim', spectral_dim)
                
                # print(f"Input shapes - Spectral: {spectral_features.shape}, Semantic: {semantic_vec.shape}")
                # print(f"Configured dimensions - Spectral: {spectral_dim}, Semantic: {semantic_dim}, Fusion: {fusion_dim}")
                
                fused_features = self.semantic_fusion(spectral_features, semantic_vec)
            else:
                fused_features = spectral_features

        return fused_features

    def forward_with_export(self, x: torch.Tensor, export_file: pathlib.Path) -> torch.Tensor:
        """Forward pass of a batch of images.

        The images should not have been normalized before and the value of each pixel should
        lie in [0, 1].

        :param x: B x C x H x W
        :export_file:
        """

        low_freq: torch.Tensor
        hi_freq: torch.Tensor
        low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

        export_file.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray((x.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy()*255).astype(np.uint8)).save(export_file)
        Image.fromarray((hi_freq.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy()*255).astype(np.uint8)).save(
            f"{export_file.parent}/{export_file.stem}_hi_freq{export_file.suffix}")
        Image.fromarray((low_freq.detach().cpu().permute(0, 2, 3, 1).squeeze(dim=0).numpy() * 255).astype(np.uint8)).save(
            f"{export_file.parent}/{export_file.stem}_low_freq{export_file.suffix}")

        # Normalize all components according to ImageNet.
        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        semantic_vec = None
        if isinstance(self.vit, backbones.CLIPBackbone):
            semantic_vec = self.vit.get_image_embedding(x)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq, semantic_vec)
        if self.cls_head is not None:
            x = self.cls_head(x)

        return x

    def get_vision_transformer(self) -> vision_transformer.VisionTransformer:
        return self.vit

    def unfreeze_backbone(self) -> None:
        self.frozen_backbone = False

    def freeze_backbone(self) -> None:
        self.frozen_backbone = True

    def _extract_features(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast():
            if isinstance(self.vit, DINOv2FeatureEmbedding):
                x = self.vit(x)
                low_freq = self.vit(low_freq)
                hi_freq = self.vit(hi_freq)
            else:
                x = self.vit(x)
                low_freq = self.vit(low_freq)
                hi_freq = self.vit(hi_freq)
        return x, low_freq, hi_freq

    def export_onnx(self, export_file: pathlib.Path) -> None:
        outer_instance = self
        class ExportableMFVit(nn.Module):
            def __init__(self):
                super().__init__()
                if (isinstance(outer_instance.vit, vision_transformer.VisionTransformer)
                        or isinstance(outer_instance.vit, backbones.DINOv2Backbone)):
                    # ImageNet normalization
                    self.backbone_norm = utils.ExportableImageNormalization(
                        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                    )
                elif isinstance(outer_instance.vit, backbones.CLIPBackbone):
                    # CLIP normalization
                    self.backbone_norm = utils.ExportableImageNormalization(
                        mean=backbones.CLIP_MEAN, std=backbones.CLIP_STD
                    )
                else:
                    raise TypeError(f"Unsupported backbone type: {type(outer_instance.vit)}")

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """Input size: B x C x H x W"""
                low_freq, hi_freq = filters.filter_image_frequencies(
                    x.float(), outer_instance.frequencies_mask
                )

                low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
                hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

                # Normalize all components according to ImageNet.
                x = self.backbone_norm(x)
                low_freq = self.backbone_norm(low_freq)
                hi_freq = self.backbone_norm(hi_freq)

                semantic_vec = None
                if isinstance(outer_instance.vit, backbones.CLIPBackbone):
                    semantic_vec = outer_instance.vit.get_image_embedding(x)

                with torch.no_grad():
                    x, low_freq, hi_freq = outer_instance._extract_features(x, low_freq, hi_freq)

                x = outer_instance.features_processor.exportable_forward(x, low_freq, hi_freq, semantic_vec)

                return x
        model: ExportableMFVit = ExportableMFVit()
        x: torch.Tensor = torch.rand((1, 3, 224, 224))
        # torch._dynamo.mark_dynamic(x, 0)
        onnx_options: torch.onnx.ExportOptions = torch.onnx.ExportOptions(dynamic_shapes=True)
        onnx_program: torch.onnx.ONNXProgram = torch.onnx.dynamo_export(
            model, x, export_options=onnx_options
        )
        # onnx_program: torch.onnx.ONNXProgram = torch.onnx.export(
        #     model, (x, ), dynamo=True
        # )
        onnx_program.save(str(export_file))

    def export_onnx_without_fft(self, export_file: pathlib.Path) -> None:
        outer_instance = self
        class ExportableMFVit(nn.Module):
            def __init__(self):
                super().__init__()
                self.outer_instance = outer_instance
            def forward(
                self,
                x: torch.Tensor,
                x_low: torch.Tensor,
                x_high: torch.Tensor
            ) -> torch.Tensor:
                """Input size: B x C x H x W"""
                # with torch.no_grad():
                x, x_low, x_high = self.outer_instance._extract_features(x, x_low, x_high)
                semantic_vec = None
                if isinstance(outer_instance.vit, backbones.CLIPBackbone):
                    semantic_vec = outer_instance.vit.get_image_embedding(x)
                x = self.outer_instance.features_processor.exportable_forward(x, x_low, x_high, semantic_vec)
                return x
        model: ExportableMFVit = ExportableMFVit()
        x: torch.Tensor = torch.rand((3, 3, 224, 224))
        # x_low: torch.Tensor = torch.rand((3, 3, 224, 224))
        # x_hi: torch.Tensor = torch.rand((3, 3, 224, 224))

        # Required image preprocessing.
        x_low, x_hi = filters.filter_image_frequencies(
            x.float(), outer_instance.frequencies_mask
        )
        x_low = torch.clamp(x_low, min=0., max=1.).to(x.dtype)
        x_hi = torch.clamp(x_hi, min=0., max=1.).to(x.dtype)
        # Normalize all components according to ImageNet.
        x = self.backbone_norm(x)
        x_low = self.backbone_norm(x_low)
        x_hi = self.backbone_norm(x_hi)

        # Batch size should be a dynamic shape.
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x_low, 0)
        torch._dynamo.mark_dynamic(x_hi, 0)

        # onnx_options: torch.onnx.ExportOptions = torch.onnx.ExportOptions(dynamic_shapes=True)
        # onnx_program: torch.onnx.ONNXProgram = torch.onnx.dynamo_export(
        #     model, x, x_low, x_hi, export_options=onnx_options
        # )
        batch_dim: torch.export.Dim = torch.export.Dim("batch_size", min=1, max=2048)
        ep: torch.export.ExportedProgram = torch.export.export(
            model,
            args=(x, x_low, x_hi),
            dynamic_shapes={
                "x": {0: batch_dim},
                "x_low": {0: batch_dim},
                "x_high": {0: batch_dim},
            }
        )
        onnx_program = torch.onnx.export(ep, dynamo=True, report=True, verify=True)
        # onnx_program: torch.onnx.ONNXProgram = torch.onnx.export(
        #     model, (x, x_low, x_hi), export_file, dynamo=True,
        #     # input_names=["x", "x_low", "x_high"], output_names=["y"],
        #     # dynamic_axes={
        #     #     "x": {0: "batch_size"},
        #     #     "x_low": {0: "batch_size"},
        #     #     "x_high": {0: "batch_size"},
        #     #     "y": {0: "batch_size"}
        #     # },
        #     # export_params=True
        # )
        onnx_program.save(str(export_file))


class ClassificationVisionTransformer(nn.Module):

    def __init__(
        self,
        vit: vision_transformer.VisionTransformer,
        features_processor: 'DenseIntermediateFeaturesProcessor',
        cls_head: Optional[nn.Module],
        frozen_backbone: bool = True
    ):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.cls_head = cls_head
        self.apply(_init_weights)
        self.frozen_backbone: bool = frozen_backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            with torch.no_grad():
                x = self.vit(x)
        else:
            x = self.vit(x)
        x = self.features_processor(x)
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x

    def get_vision_transformer(self) -> vision_transformer.VisionTransformer:
        return self.vit

    def unfreeze_backbone(self) -> None:
        self.frozen_backbone = False

    def freeze_backbone(self) -> None:
        self.frozen_backbone = True


class FrequencyRestorationEstimator(nn.Module):
    """Estimates the ability to restore missing frequencies from low frequencies."""

    def __init__(
        self,
        features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = False,
        patch_projection_per_feature: bool = False,
        proj_last_layer_activation_type: Optional[str] = "gelu",
        original_image_features_branch: bool = False,
        dropout: float = 0.5,
        disable_reconstruction_similarity: bool = False,
    ):
        super().__init__()
        
        # Store parameters
        self.features_num = features_num
        self.proj_dim = proj_dim

        if proj_last_layer_activation_type == "gelu":
            proj_last_layer_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            proj_last_layer_activation = nn.Identity
        else:
            raise RuntimeError(
                "Unsupported activation type for the "
                f"last projection layer: {proj_last_layer_activation_type}"
            )

        if patch_projection and patch_projection_per_feature:
            self.patch_projector: nn.Module = FeatureSpecificProjector(
                features_num,
                proj_layers,
                input_dim,
                proj_dim,
                proj_last_layer_activation,
                dropout=dropout,
            )
        elif patch_projection:
            self.patch_projector: nn.Module = Projector(
                proj_layers,
                input_dim,
                proj_dim,
                proj_last_layer_activation,
                dropout=dropout,
            )
        else:
            self.patch_projector: nn.Module = nn.Identity()

        self.original_features_processor = None
        if original_image_features_branch:
            self.original_features_processor = FeatureImportanceProjector(
                features_num, proj_dim, proj_dim, proj_layers, dropout=dropout
            )

        # A flag that when set stops the computation of reconstruction similarity scores.
        # Useful for performing ablation studies.
        self.disable_reconstruction_similarity: bool = disable_reconstruction_similarity
        if self.disable_reconstruction_similarity:
            assert self.original_features_processor is not None, (
                "Frequency Reconstruction Similarity cannot be disabled without "
                "Original Features Processor."
            )

    def forward(
        self, x: torch.Tensor, low_freq: torch.Tensor, hi_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.
        :param low_freq:
        :param hi_freq:

        :returns: Dimensionality B x (6 * N)
        """
        # Handle both 2D and 4D inputs
        if len(x.shape) == 2:  # B x D
            x = x.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x D
        if len(low_freq.shape) == 2:
            low_freq = low_freq.unsqueeze(1).unsqueeze(1)
        if len(hi_freq.shape) == 2:
            hi_freq = hi_freq.unsqueeze(1).unsqueeze(1)

        # For single feature case (N=1), expand to match expected dimensions
        if x.shape[1] == 1:
            x = x.expand(-1, self.features_num, -1, -1)
            low_freq = low_freq.expand(-1, self.features_num, -1, -1)
            hi_freq = hi_freq.expand(-1, self.features_num, -1, -1)

        orig = self.patch_projector(x)  # B x N x L x D
        low_freq = self.patch_projector(low_freq)  # B x N x L x D
        hi_freq = self.patch_projector(hi_freq)  # B x N x L x D

        if self.disable_reconstruction_similarity:
            x = self.original_features_processor(orig)  # B x proj_dim
        else:
            sim_x_low_freq: torch.Tensor = F.cosine_similarity(
                orig, low_freq, dim=-1
            )  # B x N x L
            sim_x_hi_freq: torch.Tensor = F.cosine_similarity(
                orig, hi_freq, dim=-1
            )  # B x N x L
            sim_low_freq_hi_freq: torch.Tensor = F.cosine_similarity(
                low_freq, hi_freq, dim=-1
            )  # B x N x L

            sim_x_low_freq_mean: torch.Tensor = sim_x_low_freq.mean(dim=-1)  # B x N
            sim_x_low_freq_std: torch.Tensor = sim_x_low_freq.std(dim=-1)  # B x N
            sim_x_hi_freq_mean: torch.Tensor = sim_x_hi_freq.mean(dim=-1)  # B x N
            sim_x_hi_freq_std: torch.Tensor = sim_x_hi_freq.std(dim=-1)  # B x N
            sim_low_freq_hi_freq_mean: torch.Tensor = sim_low_freq_hi_freq.mean(
                dim=-1
            )  # B x N
            sim_low_freq_hi_freq_std: torch.Tensor = sim_low_freq_hi_freq.std(
                dim=-1
            )  # B x N

            x: torch.Tensor = torch.cat(
                [
                    sim_x_low_freq_mean,
                    sim_x_low_freq_std,
                    sim_x_hi_freq_mean,
                    sim_x_hi_freq_std,
                    sim_low_freq_hi_freq_mean,
                    sim_low_freq_hi_freq_std,
                ],
                dim=1,
            )  # B x (6 * N)

            if self.original_features_processor is not None:
                orig = self.original_features_processor(orig)  # B x proj_dim
                x = torch.cat([x, orig], dim=1)  # B x (proj_dim + 6 * N)

        return x


class FeatureSpecificProjector(nn.Module):
    def __init__(
            self,
            intermediate_features_num: int,
            proj_layers: int,
            input_dim: int,
            proj_dim: int,
            last_layer_activation = nn.GELU,
            dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList([
            Projector(proj_layers, input_dim, proj_dim, last_layer_activation, dropout=dropout)
            for _ in range(intermediate_features_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected: list[torch.Tensor] = []
        for i, projector in enumerate(self.projectors):
            projected.append(projector(x[:, i, :, :]))
        x = torch.stack(projected, dim=1)
        return x


class Projector(nn.Module):
    def __init__(
        self,
        proj_layers: int,
        input_dim: int,
        proj_dim: int,
        last_layer_activation = nn.GELU,
        input_norm: bool = True,
        output_norm: bool = True,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        patch_proj_layers: list[nn.Module] = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            patch_proj_layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                    nn.GELU() if i < proj_layers - 1 else last_layer_activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.projector: nn.Sequential = nn.Sequential(*patch_proj_layers)
        self.norm2 = nn.LayerNorm(proj_dim) if output_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.projector(x)
        x = self.norm2(x)
        return x


class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mlp_ratio: int = 1,
        dropout: float = 0.5
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim*mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*mlp_ratio, input_dim*mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim*mlp_ratio, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
         return self.head(x)


class FeatureImportanceProjector(nn.Module):

    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1, intermediate_features_num, proj_dim]))
        # Change input dimension to match the concatenated mean and std
        self.proj1 = Projector(proj_layers, 2*input_dim, proj_dim, input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim, input_norm=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input to (B*N, L, D) where B=batch_size, N=num_features, L=seq_len, D=dim
        B, N, L, D = x.shape
        x = x.reshape(B*N, L, D)
        
        x_mean: torch.Tensor = x.mean(dim=1)  # (B*N, D)
        x_std: torch.Tensor = x.std(dim=1)  # (B*N, D)
        x = torch.cat([x_mean, x_std], dim=-1)  # (B*N, 2*D)

        x = self.proj1(x)  # (B*N, proj_dim)
        x = x.reshape(B, N, -1)  # (B, N, proj_dim)
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # (B, proj_dim)
        x = self.proj2(x)

        return x

    def exportable_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, L, D = x.shape
        x = x.reshape(B*N, L, D)
        
        x_mean: torch.Tensor = x.mean(dim=1)  # (B*N, D)
        x_std: torch.Tensor = utils.exportable_std(x, dim=1)  # (B*N, D)
        x = torch.cat([x_mean, x_std], dim=-1)  # (B*N, 2*D)

        x = self.proj1(x)  # (B*N, proj_dim)
        x = x.reshape(B, N, -1)  # (B, N, proj_dim)
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # (B, proj_dim)
        x = self.proj2(x)

        return x


class DenseIntermediateFeaturesProcessor(nn.Module):

    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = False,
        patch_projection_per_feature: bool = False,
        patch_pooling: str = "mean",
        dropout: float = 0.5
    ):
        super().__init__()

        self.patch_pooling: str = patch_pooling

        self.feature_specific_patch_projection = None
        self.patch_projection = None
        if patch_projection:
            if patch_projection_per_feature:
                patch_projection_modules: list[nn.Module] = []
                for _ in range(intermediate_features_num):
                    patch_proj_layers: list[nn.Module] = [nn.Dropout(dropout)]
                    for i in range(proj_layers):
                        patch_proj_layers.extend(
                            [
                                nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                            ]
                        )
                    patch_projection_modules.append(nn.Sequential(*patch_proj_layers))
                self.feature_specific_patch_projection = nn.ModuleList(patch_projection_modules)
            else:
                patch_proj_layers: list[nn.Module] = [nn.Dropout(dropout)]
                for i in range(proj_layers):
                    patch_proj_layers.extend(
                        [
                            nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ]
                    )
                self.patch_projection = nn.Sequential(*patch_proj_layers)

        self.alpha = nn.Parameter(torch.randn([1, intermediate_features_num, proj_dim]))
        proj1_layers: list[nn.Module] = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            proj1_layers.extend(
                [
                    nn.Linear(input_dim if i == 0 and not patch_projection else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers: list[nn.Module] = [nn.Dropout(dropout)]
        for _ in range(proj_layers):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.
        """

        if self.feature_specific_patch_projection is not None:
            projected: list[torch.Tensor] = []
            for i, patch_projection in enumerate(self.feature_specific_patch_projection):
                projected.append(patch_projection(x[:, i, :, :]))
            x = torch.stack(projected, dim=1)
        elif self.patch_projection is not None:
            x = self.patch_projection(x)

        if self.patch_pooling == "l2_max":
            x = F.normalize(x, p=2, dim=2)  # L2-normalization over the patch tokens
            x = x.max(dim=2)[0]  # B x N x D
        elif self.patch_pooling == "mean":
            # Average all the image tokens to a single image token
            x = x.mean(dim=2)  # B x N x D
        else:
            raise RuntimeError(f"Unsuported pooling approach: {self.patch_pooling}")

        x = self.proj1(x)  # B x N x proj_dim
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # B x proj_dim
        x = self.proj2(x)

        return x


class MeanNormDenseIntermediateFeaturesProcessor(nn.Module):

    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
    ):
        super().__init__()

        self.intermediate_features_num: int = intermediate_features_num
        assert self.intermediate_features_num > 0

        self.norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.intermediate_features_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.

        :returns: Dimensionality B x N*D where N the number of intermediate layers.
        """
        x = x.mean(dim=2)  # B x N x D
        normalized: list[torch.Tensor] = []
        for i in range(self.intermediate_features_num):
            normalized.append(self.norms[i](x[:, i, :]))
        x = torch.cat(normalized, dim=1)
        x = torch.flatten(x, start_dim=1)
        return x


class NormMaxDenseIntermediateFeaturesProcessor(nn.Module):

    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
    ):
        super().__init__()

        self.intermediate_features_num: int = intermediate_features_num
        assert self.intermediate_features_num > 0

        self.norms = nn.ModuleList([
            nn.LayerNorm(input_dim) for _ in range(self.intermediate_features_num)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.

        :returns: Dimensionality B x N*D where N the number of intermediate layers.
        """
        x = F.normalize(x, p=2, dim=2)  # L2-normalization over the patch tokens
        x = x.max(dim=2)[0]  # B x N x D

        normalized: list[torch.Tensor] = []
        for i in range(self.intermediate_features_num):
            normalized.append(self.norms[i](x[:, i, :]))

        x = torch.cat(normalized, dim=1)
        x = torch.flatten(x, start_dim=1)
        return x




@dataclasses.dataclass
class AttentionMask:
    mask: Optional[pathlib.Path] = None
    overlay: Optional[pathlib.Path] = None
    overlayed_image: Optional[pathlib.Path] = None


def build_cls_vit(config) -> ClassificationVisionTransformer:
    # Build features extractor.
    vit: vision_transformer.VisionTransformer = vision_transformer.build_vit(config)

    # Build features processor.
    if config.MODEL.VIT.FEATURES_PROCESSOR == "rine":
        features_processor: nn.Module = DenseIntermediateFeaturesProcessor(
            intermediate_features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
            input_dim=config.MODEL.VIT.EMBED_DIM,
            proj_dim=config.MODEL.VIT.PROJECTION_DIM,
            proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
            patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
            patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
            patch_pooling=config.MODEL.VIT.PATCH_POOLING
        )
        cls_vector_dim: int = config.MODEL.VIT.PROJECTION_DIM
    elif config.MODEL.VIT.FEATURES_PROCESSOR == "mean_norm":
        features_processor: nn.Module = MeanNormDenseIntermediateFeaturesProcessor(
            intermediate_features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
            input_dim=config.MODEL.VIT.EMBED_DIM
        )
        cls_vector_dim: int = (config.MODEL.VIT.EMBED_DIM
                               * len(config.MODEL.VIT.INTERMEDIATE_LAYERS))
    elif config.MODEL.VIT.FEATURES_PROCESSOR == "norm_max":
        features_processor: nn.Module = NormMaxDenseIntermediateFeaturesProcessor(
            intermediate_features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
            input_dim=config.MODEL.VIT.EMBED_DIM
        )
        cls_vector_dim: int = (config.MODEL.VIT.EMBED_DIM
                               * len(config.MODEL.VIT.INTERMEDIATE_LAYERS))
    else:
        raise RuntimeError(f"Unsupported features processor: {config.MODEL.VIT.FEATURES_PROCESSOR}")

    cls_head: Optional[ClassificationHead]
    if config.TRAIN.MODE == "contrastive":
        cls_head = None
    elif config.TRAIN.MODE == "supervised":
        # Build classification head.
        cls_head = ClassificationHead(
            input_dim=cls_vector_dim,
            num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1
        )
    else:
        raise RuntimeError(f"Unsupported train mode: {config.TRAIN.MODE}")

    return ClassificationVisionTransformer(vit, features_processor, cls_head)


class DINOv2FeatureEmbedding(nn.Module):
    """Projector that embeds DINOv2 features into a lower-dimensional space."""
    def __init__(self, model_name="dinov2_vitb14", 
                 device='cuda' if torch.cuda.is_available() else 'cpu', 
                 proj_dim=512):
        super().__init__()
        self.device = device
        
        # Load DINOv2 model
        if model_name == "dinov2_vitl14":
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.output_dim = 1024
        elif model_name == "dinov2_vitg14":
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            self.output_dim = 1536
        elif model_name == "dinov2_vitb14":
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.output_dim = 768
        elif model_name == "dinov2_vits14":
            self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.output_dim = 384
        else:
            raise ValueError(f"Unknown DINOv2 model: {model_name}")
            
        self.dino_model.to(device)
        self.dino_model.eval()  # Freeze model
        
        for param in self.dino_model.parameters():
            param.requires_grad = False
            
        print(f"Loaded DINOv2 model {model_name} with output dim {self.output_dim}")

    def forward(self, images):
        # Handle input types: list of tensors vs 4D tensor
        if isinstance(images, list):
            processed_images = torch.stack([
                self.preprocess_image(img).to(self.device) for img in images
            ])
        else:
            processed_images = torch.stack([
                self.preprocess_image(img).to(self.device) for img in images
            ])

        # Extract features using DINOv2
        with torch.no_grad():
            # Get features from DINOv2
            features = self.dino_model(processed_images)
            
            # DINOv2 returns features in shape (B, D)
            # Reshape to match expected format: B x 1 x 1 x D
            B, D = features.shape
            # Add sequence length dimension (L=1) and feature dimension (N=1)
            patch_tokens = features.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x D

        return patch_tokens.float()

    def preprocess_image(self, image_tensor):
        # DINOv2 expects images normalized with ImageNet stats
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225))
        ])
        return preprocess(image_tensor)


def build_mf_vit(config) -> MFViT:
    # Build features extractor.
    if config.MODEL_WEIGHTS != "dinov2":
        raise ValueError("MODEL_WEIGHTS must be set to 'dinov2' to use semantic features")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit: DINOv2FeatureEmbedding = DINOv2FeatureEmbedding(
        model_name="dinov2_vitb14",
        device=device
    )
    initialization_scope: str = "local"

    fre: FrequencyRestorationEstimator = FrequencyRestorationEstimator(
        features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
        input_dim=config.MODEL.VIT.EMBED_DIM,
        proj_dim=config.MODEL.VIT.PROJECTION_DIM,
        proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
        patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
        patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
        proj_last_layer_activation_type=config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE,
        original_image_features_branch=config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH,
        disable_reconstruction_similarity=config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY,
    )

    # Output dimension of the internal MFViT module (after potential fusion)
    # If DINOv2 is used, MFViT's semantic_fusion module will output features of fusion_dim,
    # which is initialized with features_processor.proj_dim (i.e., config.MODEL.VIT.PROJECTION_DIM).
    mfvit_actual_output_dim = config.MODEL.VIT.PROJECTION_DIM if config.MODEL_WEIGHTS == "dinov2" else (6 * len(config.MODEL.VIT.INTERMEDIATE_LAYERS) + (config.MODEL.VIT.PROJECTION_DIM if config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH else 0))
    if config.MODEL_WEIGHTS != "dinov2" and config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH and config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY:
         mfvit_actual_output_dim = config.MODEL.VIT.PROJECTION_DIM

    # For PatchBasedMFViT, its *own* layers (to_kv, norm, cls_head) expect dimensions from the checkpoint.
    # Use config values that would correspond to the checkpoint's structure for these specific layers.
    # The error log indicates: to_kv input 1096, attn_embed_dim 1536. cls_head input 1096.
    # These should ideally come from config if they were fixed at checkpoint creation.
    # Let's use the values from the error log as a guide for now if not directly in config.
    # For example, if PATCH_VIT.CLS_VECTOR_DIM and PATCH_VIT.ATTN_EMBED_DIM from the config
    # are expected to match the checkpoint.

    cls_head_for_fixed_resolution: Optional[ClassificationHead] = None
    if config.TRAIN.MODE == "supervised" and config.MODEL.RESOLUTION_MODE == "fixed":
        cls_head_for_fixed_resolution = ClassificationHead(
            input_dim=mfvit_actual_output_dim, # MFViT (fixed) directly uses its own output
            num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
            mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
            dropout=config.MODEL.SID_DROPOUT
        )

    if config.MODEL.RESOLUTION_MODE == "fixed":
        model = MFViT(
            vit,
            fre,
            cls_head_for_fixed_resolution,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_size=config.DATA.IMG_SIZE,
            initialization_scope=initialization_scope
        )
    elif config.MODEL.RESOLUTION_MODE == "arbitrary":
        # These are the dimensions PatchBasedMFViT's *internal layers* expect, based on the checkpoint.
        # The error log implies: cls_vector_dim_for_patch_vit_layers = 1096
        #                        attn_embed_dim_for_patch_vit_layers = 1536 (since to_kv output is 3072)
        # We should use config values that reflect these checkpoint dimensions.
        # Assuming config.MODEL.PATCH_VIT.CLS_VECTOR_DIM and config.MODEL.PATCH_VIT.ATTN_EMBED_DIM
        # are set to these checkpoint-compatible values (1096 and 1536 respectively).
        cls_vector_dim_for_patch_vit_layers = config.MODEL.PATCH_VIT.CLS_VECTOR_DIM 
        attn_embed_dim_for_patch_vit_layers = config.MODEL.PATCH_VIT.ATTN_EMBED_DIM

        cls_head_for_patch_based: Optional[ClassificationHead] = None
        if config.TRAIN.MODE == "supervised":
            cls_head_for_patch_based = ClassificationHead(
                input_dim=cls_vector_dim_for_patch_vit_layers, # Head expects the checkpoint's vector dim
                num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
                mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
                dropout=config.MODEL.SID_DROPOUT
            )

        model = PatchBasedMFViT(
            vit,
            fre,
            cls_head_for_patch_based,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_patch_size=config.DATA.IMG_SIZE,
            img_patch_stride=config.MODEL.PATCH_VIT.PATCH_STRIDE,
            cls_vector_dim=cls_vector_dim_for_patch_vit_layers,
            attn_embed_dim=attn_embed_dim_for_patch_vit_layers,
            num_heads=config.MODEL.PATCH_VIT.NUM_HEADS,
            dropout=config.MODEL.SID_DROPOUT,
            minimum_patches=config.MODEL.PATCH_VIT.MINIMUM_PATCHES,
            initialization_scope=initialization_scope,
            mfvit_output_dim_actual=mfvit_actual_output_dim
        )
    else:
        raise RuntimeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")

    model = model.to(device)

    if isinstance(model, MFViT) and not isinstance(model, PatchBasedMFViT):
        backbone_test = model.vit
    elif isinstance(model, PatchBasedMFViT):
        backbone_test = model.mfvit.vit 
    else:
        raise RuntimeError(f"Unsupported model type for DINOv2 test: {type(model)}")

    if isinstance(backbone_test, DINOv2FeatureEmbedding):
        test_input = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).to(device)
        with torch.no_grad():
            semantic_vec = backbone_test(test_input)
        if semantic_vec is None or semantic_vec.shape[-1] != 768:
            raise ValueError(f"Semantic feature extraction failed or dim mismatch. Expected 768, got {semantic_vec.shape[-1] if semantic_vec is not None else None}")
    else:
        print(f"⚠️ Skipping DINOv2 semantic feature extraction test for backbone type: {type(backbone_test)}")

    return model


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
