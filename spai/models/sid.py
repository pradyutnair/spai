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
from typing import Literal, Optional, Union

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
from spai.utils import save_image_with_attention_overlay

from typing import Union, List

from . import backbones, filters, utils, vision_transformer
import torchvision.transforms as transforms

class PatchBasedMFViT(nn.Module):
    def __init__(
        self,
        vit: Union[vision_transformer.VisionTransformer,
                   backbones.CLIPBackbone,
                   backbones.DINOv2Backbone],
        features_processor: 'FrequencyRestorationEstimator',
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_patch_size: int,
        img_patch_stride: int,
        cls_vector_dim: int,
        num_heads: int,
        attn_embed_dim: int,
        dropout: float = 0.0,
        frozen_backbone: bool = True,
        minimum_patches: int = 0,
        initialization_scope: str = "all",
        use_semantic_cross_attn_sca: Union[Literal["before", "after"], None] = None,
        use_dual_cross_attn_sca: bool = False,
        semantic_embed_dim: Optional[int] = None,
        semantic_heads: Optional[int] = None,
        semantic_encoder: str = "clip",  # "clip" or "convnext"
    ) -> None:
        super().__init__()

        self.mfvit = MFViT(
            vit,
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

        # Cross-Attention with a learnable vector layers.
        dim_head: int = attn_embed_dim // num_heads
        self.heads = num_heads
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
        # adding semantic cross-attention before or after the SCA module or not using it at all
        self.use_semantic_cross_attn_sca = use_semantic_cross_attn_sca
        self.use_dual_cross_attn_sca = use_dual_cross_attn_sca
        self.semantic_embed_dim = semantic_embed_dim
        self.semantic_heads = (
            semantic_heads if semantic_heads is not None else num_heads
        )

        self.semantic_encoder_type = semantic_encoder

        if self.use_semantic_cross_attn_sca in ["before", "after"]:

            print(f"Using semantic cross-attention: {self.use_semantic_cross_attn_sca}")
            self.semantic_mha = nn.MultiheadAttention(
                embed_dim=cls_vector_dim, num_heads=self.semantic_heads, dropout=dropout
            )
            self.semantic_layer_norm = nn.LayerNorm(cls_vector_dim)

            if self.use_dual_cross_attn_sca:
                print("Using dual cross-attention SCA")
                self.semantic_mha2 = nn.MultiheadAttention(
                    embed_dim=cls_vector_dim,
                    num_heads=self.semantic_heads,
                    dropout=dropout,
                )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.semantic_encoder_type == "clip":
                print("Using CLIP backbone for semantic encoding")
                self.semantic_encoder = backbones.CLIPBackbone(
                    device=self.device
                ).float()
                self.semantic_encoder.eval()
                for param in self.semantic_encoder.parameters():
                    param.requires_grad = False

                assert (
                    self.semantic_embed_dim is not None
                ), "semantic_embed_dim must be set for CLIP"
                self.semantic_projection = nn.Linear(
                    self.semantic_embed_dim, cls_vector_dim
                )
                # self.semantic_projection = nn.Sequential(
                    # nn.LayerNorm(self.semantic_embed_dim),
                    # nn.Linear(self.semantic_embed_dim, cls_vector_dim),
                    # nn.ReLU(),
                    # nn.Linear(cls_vector_dim, cls_vector_dim),
                # )
                self.semantic_projection.requires_grad_(True)  # make trainable

            elif self.semantic_encoder_type == "convnext":
                print("Using ConvNeXt backbone for semantic encoding")
                self.semantic_encoder = SemanticPipeline(output_dim=cls_vector_dim).to(
                    self.device
                )
                self.semantic_encoder.backbone.eval()
                for param in self.semantic_encoder.backbone.parameters():
                    param.requires_grad = False
                for param in self.semantic_encoder.convnext_proj.parameters():
                    param.requires_grad = True
            elif self.semantic_encoder_type == "dino":
                print("Using DINOv2 backbone for semantic encoding")
                self.semantic_encoder = backbones.DINOv2FeatureEmbedding()
                self.semantic_encoder.eval()
                for param in self.semantic_encoder.parameters():
                    param.requires_grad = False
                self.semantic_embed_dim = self.semantic_encoder.output_dim
                print(f"Using DINOv2 backbone for semantic encoding with output dim: {self.semantic_embed_dim}")
                self.semantic_projection = nn.Sequential(
                    nn.LayerNorm(self.semantic_embed_dim),
                    nn.Linear(self.semantic_embed_dim, cls_vector_dim),
                    nn.ReLU(),
                    nn.Linear(cls_vector_dim, cls_vector_dim),
                )
                self.semantic_projection.requires_grad_(True)
            else:
                raise ValueError(
                    f"Unknown semantic_encoder: {self.semantic_encoder_type}"
                )

        if initialization_scope == "all":
            self.apply(_init_weights)
        elif initialization_scope == "local":
            # Initialize only the newly added components, by excluding mfvit.
            for m_name, m in self._modules.items():
                if m_name == "mfvit":
                    continue
                else:
                    m.apply(_init_weights)
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
            x =  self.forward_batch(x)
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
        aggregator: torch.Tensor = self.patch_aggregator.expand(x.size(0), -1, -1, -1)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
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
        # print(f"Input shape: {x.shape}")
        # Compute global image encoding before patchifying
        if self.use_semantic_cross_attn_sca in ["before", "after"]:
            # Get the global image encoding from the CLIP backbone
            # Resize the image to 224x224 for CLIP
            x_resized = F.interpolate(
                x, size=(224, 224), mode="bilinear", align_corners=False
            )

            # CLIP
            if self.semantic_encoder_type == "clip":
                # Get the global image encoding from the CLIP backbone
                global_image_encoding = self.semantic_encoder.get_image_embedding(
                    x_resized.float()
                )
                # Project global image encoding to match the cross-attention query dimension
                global_image_encoding = self.semantic_projection(
                    global_image_encoding.float()
                )  # B x D
            elif self.semantic_encoder_type == "dino":
                # Get the global image encoding from the DINOv2 backbone
                global_image_encoding = self.semantic_encoder(
                    x_resized.float()
                )
                # Project global image encoding to match the cross-attention query dimension
                global_image_encoding = self.semantic_projection(
                    global_image_encoding.float()
                )
            # ConvNeXt
            else:
                # Get the global image encoding from the ConvNeXt backbone
                global_image_encoding = self.semantic_encoder(x_resized.float())

        # Patchify the input image
        x = utils.patchify_image(
            x,
            (self.img_patch_size, self.img_patch_size),
            (self.img_patch_stride, self.img_patch_stride)
        )  # B x L x C x H x W

        # Extract patch features
        patch_features: list[torch.Tensor] = []
        for i in range(x.size(1)):
            patch_features.append(self.mfvit(x[:, i]))
        x = torch.stack(patch_features, dim=1)  # B x L x D
        del patch_features

        # Spectral-semantic cross-attention
        if self.use_semantic_cross_attn_sca == "before":
            semantic_enc = global_image_encoding.unsqueeze(1).expand(
                -1, x.size(1), -1
            )  # B x L x D

            attn_output, attn_weights = self.semantic_mha(
                query=x, key=semantic_enc, value=semantic_enc, need_weights=True
            )

            if self.use_dual_cross_attn_sca:
                attn_output2, _ = self.semantic_mha2(
                    query=semantic_enc, key=x, value=x, need_weights=True
                )
                x_out = (attn_output + attn_output2) / 2
            else:
                x_out = attn_output

            # 💡 Apply fusion gate
            fusion_gate = torch.sigmoid(self.semantic_fusion_gate)
            print(f"[Fusion Gate (before)] = {fusion_gate.item():.4f}")
            print(f"[Semantic Attn Mean (before)] = {attn_weights.mean().item():.4f}")

            x = self.semantic_layer_norm((1 - fusion_gate) * x + fusion_gate * x_out)

        # Spectral context attention (SCA)
        x = self.patches_attention(x)  # B x D
        x = self.norm(x)  # B x D

        if self.use_semantic_cross_attn_sca == "after":
            semantic_enc = global_image_encoding  # B x D

            attn_output, attn_weights = self.semantic_mha(
                query=x, key=semantic_enc, value=semantic_enc, need_weights=True
            )

            if self.use_dual_cross_attn_sca:
                attn_output2, _ = self.semantic_mha2(
                    query=semantic_enc, key=x, value=x, need_weights=True
                )
                x_out = (attn_output + attn_output2) / 2
            else:
                x_out = attn_output

            # fusion_gate = torch.sigmoid(self.semantic_fusion_gate)
            # print(f"[Fusion Gate (after)] = {fusion_gate.item():.4f}")
            print(f"[Semantic Attn Mean (after)] = {attn_weights.mean().item():.4f}")

            # x = self.semantic_layer_norm((1 - fusion_gate) * x + fusion_gate * x_out)
            # dont use fusion gate
            x = self.semantic_layer_norm(x + x_out)

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

        # Attend to patches according to the image they belong to.
        attended: list[torch.Tensor] = []
        processed_sum: int = 0
        for i in img_patches_num:
            attended.append(self.patches_attention(x[processed_sum:processed_sum+i].unsqueeze(0)))
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
                   backbones.DINOv2Backbone],
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

        if (isinstance(self.vit, vision_transformer.VisionTransformer)
                or isinstance(self.vit, backbones.DINOv2Backbone)):
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

        low_freq: torch.Tensor
        hi_freq: torch.Tensor
        low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0., max=1.).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0., max=1.).to(x.dtype)

        # Normalize all components according to ImageNet.
        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
        if self.cls_head is not None:
            x = self.cls_head(x)

        return x

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

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
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

                with torch.no_grad():
                    x, low_freq, hi_freq = outer_instance._extract_features(x, low_freq, hi_freq)

                x = outer_instance.features_processor.exportable_forward(x, low_freq, hi_freq)

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
                x = self.outer_instance.features_processor.exportable_forward(x, x_low, x_high)
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
        disable_reconstruction_similarity: bool = False
    ):
        super().__init__()

        if proj_last_layer_activation_type == "gelu":
            proj_last_layer_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            proj_last_layer_activation = nn.Identity
        else:
            raise RuntimeError("Unsupported activation type for the "
                               f"last projection layer: {proj_last_layer_activation_type}")

        if patch_projection and patch_projection_per_feature:
            self.patch_projector: nn.Module = FeatureSpecificProjector(
                features_num, proj_layers, input_dim, proj_dim, proj_last_layer_activation,
                dropout=dropout
            )
        elif patch_projection:
            self.patch_projector: nn.Module = Projector(
                proj_layers, input_dim, proj_dim, proj_last_layer_activation, dropout=dropout
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
            assert self.original_features_processor is not None, \
                ("Frequency Reconstruction Similarity cannot be disabled without "
                 "Original Features Processor.")

    def forward(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: Dimensionality B x N x L x D where N the number of intermediate layers.
        :param low_freq:
        :param hi_freq:

        :returns: Dimensionality B x (6 * N)
        """
        orig = self.patch_projector(x)  # B x N x L x D
        low_freq = self.patch_projector(low_freq)  # B x N x L x D
        hi_freq = self.patch_projector(hi_freq)  # B x N x L x D

        if self.disable_reconstruction_similarity:
            x = self.original_features_processor(orig)  # B x proj_dim
        else:
            sim_x_low_freq: torch.Tensor = F.cosine_similarity(orig, low_freq, dim=-1)  # B x N x L
            sim_x_hi_freq: torch.Tensor = F.cosine_similarity(orig, hi_freq, dim=-1)  # B x N x L
            sim_low_freq_hi_freq: torch.Tensor = F.cosine_similarity(low_freq, hi_freq, dim=-1)  # B x N x L

            sim_x_low_freq_mean: torch.Tensor = sim_x_low_freq.mean(dim=-1)  # B x N
            sim_x_low_freq_std: torch.Tensor = sim_x_low_freq.std(dim=-1)  # B x N
            sim_x_hi_freq_mean: torch.Tensor = sim_x_hi_freq.mean(dim=-1)  # B x N
            sim_x_hi_freq_std: torch.Tensor = sim_x_hi_freq.std(dim=-1)  # B x N
            sim_low_freq_hi_freq_mean: torch.Tensor = sim_low_freq_hi_freq.mean(dim=-1)  # B x N
            sim_low_freq_hi_freq_std: torch.Tensor = sim_low_freq_hi_freq.std(dim=-1)  # B x N

            x: torch.Tensor = torch.cat([
                sim_x_low_freq_mean,
                sim_x_low_freq_std,
                sim_x_hi_freq_mean,
                sim_x_hi_freq_std,
                sim_low_freq_hi_freq_mean,
                sim_low_freq_hi_freq_std
            ], dim=1)  # B x (6 * N)

            if self.original_features_processor is not None:
                orig = self.original_features_processor(orig)  # B x proj_dim
                x = torch.cat([x, orig], dim=1)  # B x (proj_dim + 6 * N)

        return x

    def exportable_forward(
        self,
        x: torch.Tensor,
        low_freq: torch.Tensor,
        hi_freq: torch.Tensor
    ) -> torch.Tensor:
        orig = self.patch_projector(x)  # B x N x L x D
        low_freq = self.patch_projector(low_freq)  # B x N x L x D
        hi_freq = self.patch_projector(hi_freq)  # B x N x L x D

        sim_x_low_freq: torch.Tensor = F.cosine_similarity(orig, low_freq, dim=-1)  # B x N x L
        sim_x_hi_freq: torch.Tensor = F.cosine_similarity(orig, hi_freq, dim=-1)  # B x N x L
        sim_low_freq_hi_freq: torch.Tensor = F.cosine_similarity(low_freq, hi_freq,
                                                                 dim=-1)  # B x N x L

        sim_x_low_freq_mean: torch.Tensor = sim_x_low_freq.mean(dim=-1)  # B x N
        sim_x_low_freq_std: torch.Tensor = utils.exportable_std(sim_x_low_freq, dim=-1)  # B x N
        sim_x_hi_freq_mean: torch.Tensor = sim_x_hi_freq.mean(dim=-1)  # B x N
        sim_x_hi_freq_std: torch.Tensor = utils.exportable_std(sim_x_hi_freq, dim=-1)  # B x N
        sim_low_freq_hi_freq_mean: torch.Tensor = sim_low_freq_hi_freq.mean(dim=-1)  # B x N
        sim_low_freq_hi_freq_std: torch.Tensor = utils.exportable_std(
            sim_low_freq_hi_freq, dim=-1
        )  # B x N

        x: torch.Tensor = torch.cat([
            sim_x_low_freq_mean,
            sim_x_low_freq_std,
            sim_x_hi_freq_mean,
            sim_x_hi_freq_std,
            sim_low_freq_hi_freq_mean,
            sim_low_freq_hi_freq_std
        ], dim=1)  # B x (6 * N)

        orig = self.original_features_processor.exportable_forward(orig)  # B x proj_dim
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
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1, intermediate_features_num, proj_dim]))
        self.proj1 = Projector(proj_layers, 2*proj_dim, proj_dim, input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim, input_norm=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean: torch.Tensor = x.mean(dim=2)  # B x N x input_dim
        x_std: torch.Tensor = x.std(dim=2)  # B x N x input_dim
        x = torch.cat([x_mean, x_std], dim=-1)  # B x N x 2*input_dim

        x = self.proj1(x)  # B x N x 2*proj_dim
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # B x proj_dim
        x = self.proj2(x)

        return x

    def exportable_forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean: torch.Tensor = x.mean(dim=2)  # B x N x input_dim
        x_std: torch.Tensor = utils.exportable_std(x, dim=2)  # B x N x input_dim
        x = torch.cat([x_mean, x_std], dim=-1)  # B x N x 2*input_dim

        x = self.proj1(x)  # B x N x 2*proj_dim
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)  # B x proj_dim
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


def build_mf_vit(config) -> MFViT:
    # Build features extractor.
    if config.MODEL_WEIGHTS == "mfm":
        vit: vision_transformer.VisionTransformer = vision_transformer.build_vit(config)
        initialization_scope: str = "all"
    elif config.MODEL_WEIGHTS == "clip":
        vit: backbones.CLIPBackbone = backbones.CLIPBackbone()
        initialization_scope: str = "local"
    elif config.MODEL_WEIGHTS == "dinov2":
        vit: backbones.DINOv2Backbone = backbones.DINOv2Backbone()
        initialization_scope: str = "local"
    elif config.MODEL_WEIGHTS in ["dinov2_vitl14", "dinov2_vitg14"]:
        vit: backbones.DINOv2Backbone = backbones.DINOv2Backbone(
            dinov2_model=config.MODEL_WEIGHTS,
            intermediate_layers=config.MODEL.VIT.INTERMEDIATE_LAYERS
        )
        initialization_scope: str = "local"
    else:
        raise RuntimeError(f"Unsupported ViT weights type: {config.MODEL_WEIGHTS}")

    # Build features processor.
    fre: FrequencyRestorationEstimator = FrequencyRestorationEstimator(
        features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
        input_dim=config.MODEL.VIT.EMBED_DIM,
        proj_dim=config.MODEL.VIT.PROJECTION_DIM,
        proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
        patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
        patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
        proj_last_layer_activation_type=config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE,
        original_image_features_branch=config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH,
        dropout=config.MODEL.SID_DROPOUT,
        disable_reconstruction_similarity=config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY
    )
    cls_vector_dim: int = 6 * len(config.MODEL.VIT.INTERMEDIATE_LAYERS)
    if (config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH
            and config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY):
        cls_vector_dim = config.MODEL.VIT.PROJECTION_DIM
    elif config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH:
        cls_vector_dim += config.MODEL.VIT.PROJECTION_DIM

    cls_head: Optional[ClassificationHead]
    if config.TRAIN.MODE == "contrastive":
        cls_head = None
    elif config.TRAIN.MODE == "supervised":
        # Build classification head.
        cls_head = ClassificationHead(
            input_dim=cls_vector_dim,
            num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
            mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
            dropout=config.MODEL.SID_DROPOUT
        )
    else:
        raise RuntimeError(f"Unsupported train mode: {config.TRAIN.MODE}")

    if config.MODEL.RESOLUTION_MODE == "fixed":
        model = MFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_size=config.DATA.IMG_SIZE
        )
    elif config.MODEL.RESOLUTION_MODE == "arbitrary":
        model = PatchBasedMFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_patch_size=config.DATA.IMG_SIZE,
            img_patch_stride=config.MODEL.PATCH_VIT.PATCH_STRIDE,
            cls_vector_dim=cls_vector_dim,
            attn_embed_dim=config.MODEL.PATCH_VIT.ATTN_EMBED_DIM,
            num_heads=config.MODEL.PATCH_VIT.NUM_HEADS,
            dropout=config.MODEL.SID_DROPOUT,
            minimum_patches=config.MODEL.PATCH_VIT.MINIMUM_PATCHES,
            initialization_scope=initialization_scope,
            use_semantic_cross_attn_sca=config.MODEL.SEMANTIC_CROSS_ATTN.CROSS_ATTN_SCA,
            semantic_embed_dim=config.MODEL.SEMANTIC_CROSS_ATTN.EMBED_DIM,
            semantic_heads=config.MODEL.SEMANTIC_CROSS_ATTN.NUM_HEADS,
            use_dual_cross_attn_sca=config.MODEL.SEMANTIC_CROSS_ATTN.DUAL_CROSS_ATTN_SCA,
            semantic_encoder=config.MODEL.SEMANTIC_CROSS_ATTN.SEMANTIC_ENCODER,
        )
    else:
        raise RuntimeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")

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


##### The following code is the addition of MoE model with semantic context

class SemanticContextModel(nn.Module):
    """
    Combines SPAI's spectral features with ConvNeXt semantic features using residual connections
    to structurally bias the model toward spectral features.
    """
    def __init__(
        self,
        spai_model_path: str,
        semantic_output_dim: int = 1096,
        projection_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
        spai_input_size:tuple = (224,224)
    ):
        super().__init__()
        self.spai_input_size = spai_input_size
        print(f'Input size for resizing SPAI model: {self.spai_input_size}')

        # === Load and freeze SPAI model ===
        from spai.models.build import build_mf_vit
        from spai.config import get_config

        cfg = get_config({"cfg": "configs/spai.yaml"})
        self.spai_model = build_mf_vit(cfg)

        checkpoint = torch.load(spai_model_path, map_location="cpu", weights_only=False)
        self.spai_model.load_state_dict(checkpoint.get("model", checkpoint))
        print(f"Loaded SPAI model from {spai_model_path}") 
        load_result = self.spai_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
        print(f"Loaded SPAI model from {spai_model_path}")
        print("Missing keys (randomly initialized):", load_result.missing_keys)
        print("Unexpected keys (in checkpoint, not in model):", load_result.unexpected_keys)
        for param in self.spai_model.parameters():
            param.requires_grad = False
        self.spai_model.eval()

        spectral_features_dim = 1096  # known output dim from SPAI feature extractor

        # === Load and freeze ConvNeXt-XXL from OpenCLIP ===
        import open_clip
        convnext_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg"
        )
        self.semantic_backbone = convnext_model.visual.trunk
        self.semantic_backbone.head.global_pool = nn.Identity()
        self.semantic_backbone.head.flatten = nn.Identity()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        for param in self.semantic_backbone.parameters():
            param.requires_grad = False
        self.semantic_backbone.eval()

        # === Projections (raw -> aligned dimensions) ===
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(3072),
            nn.Linear(3072, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # === Added: fusion layer to process combined features ===
        self.fusion_layer = nn.Sequential(
            nn.Linear(1096 + projection_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # === Modified: classifier with residual connection ===
        # Takes both spectral features directly and fusion output
        self.classifier = nn.Sequential(
            nn.Linear(1096 + 512, 512),  # spectral features + fusion features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

        # === Initialization ===
        self.semantic_projection.apply(_init_weights)
        self.fusion_layer.apply(_init_weights)
        self.classifier.apply(_init_weights)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], feature_extraction_batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with residual connection for spectral features.
        """
        device = next(self.parameters()).device
        normalize = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        convnext_resize = transforms.Resize((224, 224), antialias=True)

        if self.spai_input_size:
            spai_resize = transforms.Resize(self.spai_input_size, antialias=True)
        else:
            spai_resize = lambda x: x  # identity function, does nothing

        # === Validation/inference mode: list of images ===
        if isinstance(x, list):
            spai_input, convnext_input = [], []
            for img in x:
                if img.dim() == 4 and img.size(0) == 1:
                    img = img.squeeze(0)
                elif img.dim() != 3:
                    raise ValueError(f"Expected C×H×W or 1×C×H×W, got {img.shape}")
                if img.max() > 1.0:
                    img = img / 255.0
                img_spai = spai_resize(img)
                img_convnext = convnext_resize(img)
                spai_input.append(img_spai)
                #spai_input.append(img)
                convnext_input.append(normalize(img_convnext))
            x_spai = torch.stack(spai_input).to(device).float()
            x_convnext = torch.stack(convnext_input).to(device).float()


        # === Training mode: batched tensor ===
        else:
            if x.dim() != 4:
                raise ValueError(f"Expected batched input (B×C×H×W), got {x.shape}")
            if x.max() > 1.0:
                x = x / 255.0
            x_spai = x.to(device).float()
            x_convnext = normalize(x).to(device).float()

        # === Feature extraction ===
        with torch.no_grad():
            # SPAI – remove classification head temporarily
            original_cls_head = self.spai_model.cls_head
            self.spai_model.cls_head = nn.Identity()
            spectral_features = self.spai_model(x_spai)
            self.spai_model.cls_head = original_cls_head

            # ConvNeXt
            semantic_features = self.semantic_backbone(x_convnext)
            semantic_features = self.global_pool(semantic_features).flatten(1)

        # === Semantic projection ===
        semantic_proj = self.semantic_projection(semantic_features)  # e.g. 3072 → 256

        # === Combined features with weighting ===
        combined = torch.cat([spectral_features, semantic_proj], dim=1)
        # === Process combined features ===
        fused_features = self.fusion_layer(combined)
        
        # === RESIDUAL CONNECTION: concatenate raw spectral features with fusion output ===
        final_features = torch.cat([spectral_features, fused_features], dim=1)
        
        # === Final classification ===
        output = self.classifier(final_features)

        if not self.training:
            torch.cuda.empty_cache()

        return output

    def unfreeze_backbone(self) -> None:
        """
        Implements unfreeze_backbone for compatibility with SPAI training code.
        Since we want to keep backbones frozen in semantic model, this is a no-op.
        """
        print("Note: unfreeze_backbone() called but semantic model backbones remain frozen by design")
        pass


def build_semantic_context_model(config) -> SemanticContextModel:
    """
    Factory function to build a semantic context model.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Initialized SemanticContextModel
    """
    # Extract configuration parameters
    spai_model_path = config.MODEL.SEMANTIC_CONTEXT.SPAI_MODEL_PATH
    semantic_output_dim = config.MODEL.SEMANTIC_CONTEXT.OUTPUT_DIM
    hidden_dims = config.MODEL.SEMANTIC_CONTEXT.HIDDEN_DIMS
    dropout = config.MODEL.SEMANTIC_CONTEXT.DROPOUT
    spai_input_size = config.MODEL.SEMANTIC_CONTEXT.SPAI_INPUT_SIZE
    spai_input_size = tuple(spai_input_size) if spai_input_size is not None else None
    # Build and return the model
    model = SemanticContextModel(
        spai_model_path=spai_model_path,
        semantic_output_dim=semantic_output_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        spai_input_size = spai_input_size

    )
    
    return model





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