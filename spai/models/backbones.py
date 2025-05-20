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
import clip
import logging

logger = logging.getLogger(__name__)


CLIP_MEAN: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)


class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


class CLIPBackbone(nn.Module):
    def __init__(
        self,
        clip_model: str = "ViT-B/16",
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(clip_model, device=device)
        self.clip = self.clip.float()
        self.device = device
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

    def get_image_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the global image embedding from CLIP for the input image tensor.
        x: (B, C, H, W)
        Returns: (B, D)
        """
        x = x.to(self.device)
        with torch.no_grad():
            img_emb = self.clip.encode_image(x)
        return img_emb

    def get_text_embedding(self, text_prompts) -> torch.Tensor:
        """
        Returns the global text embedding from CLIP for the given text prompts.
        text_prompts: list[str] or str
        Returns: (B, D)
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompts).to(self.device)
            text_emb = self.clip.encode_text(text_tokens)
        return text_emb


    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = x.to(self.device)

            # Step 1: Initial convolution
            x = self.clip.visual.conv1(x)  # (B, C, H', W')
            x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten spatial dims → (B, C, HW)
            x = x.permute(0, 2, 1)  # → (B, num_patches, C)

            # Step 2: Prepend CLS token (reshaped properly)
            cls_token = self.clip.visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            cls_token = cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
            x = torch.cat([cls_token, x], dim=1)  # (B, 1 + num_patches, D)

            # Step 3: Add positional embedding
            x = x + self.clip.visual.positional_embedding[:x.size(1)].to(x.dtype)

            # Step 4: Layer norm, transformer
            x = self.clip.visual.ln_pre(x)  # (B, N, D)
            x = x.permute(1, 0, 2)  # (N, B, D) for transformer
            x = self.clip.visual.transformer(x)
            x = x.permute(1, 0, 2)  # Back to (B, N, D)

            # Step 5: Return only patch tokens (exclude CLS)
            patch_tokens = x[:, 1:, :]  # (B, num_patches, D)
            
        return patch_tokens

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of images using a CLIP backbone and returns intermediate layers."""
        # Make sure that the parameters of LayerNorm are always in FP32, even during FP16
        # training. Otherwise, it will crash, since clip utilizes a custom LayerNorm that
        # always converts the input to LayerNorm to FP32.
        if self.clip.visual.transformer.resblocks[1].ln_1.weight.dtype != torch.float32:
            for m in self.clip.modules():
                if isinstance(m, clip.model.LayerNorm):
                    m.float()

        self.clip.encode_image(x)
        x = torch.stack([h.output for h in self.hooks], dim=2)[1:, :, :, :]
        x = torch.permute(x, (1, 2, 0, 3))

        return x


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        intermediate_layers: tuple[int, ...] = tuple((i for i in range(12)))
    ) -> None:
        super().__init__()
        
        logger.info(f"Initializing DINOv2 model: {dinov2_model}")
        try:
            # Initialize DINOv2 pretrained model.
            self.dino = torch.hub.load("facebookresearch/dinov2", dinov2_model)
            logger.info(f"Successfully loaded DINOv2 model: {dinov2_model}")
            self.intermediate_layers: tuple[int, ...] = intermediate_layers
            logger.info(f"Using intermediate layers: {intermediate_layers}")
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {str(e)}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        logger.debug(f"Forward pass input shape: {x.shape}, dtype: {input_dtype}")
        try:
            x: tuple[torch.Tensor] = self.dino.get_intermediate_layers(x, self.intermediate_layers)
            x: torch.Tensor = torch.stack(x, dim=1)
            x = x.to(input_dtype)
            logger.debug(f"Forward pass output shape: {x.shape}, dtype: {x.dtype}")
            return x
        except Exception as e:
            logger.error(f"Error in DINOv2 forward pass: {str(e)}")
            raise
    