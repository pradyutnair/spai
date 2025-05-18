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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> None:
        super().__init__()

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(clip_model, device=device)
        self.device = device
        
        # Force model to float32
        self.clip = self.clip.float()
        self.clip.visual = self.clip.visual.float()
        
        # Convert all parameters to float32
        for param in self.clip.parameters():
            param.requires_grad = False
            param.data = param.data.to(torch.float32)
            
        # Convert all LayerNorm modules to float32
        for m in self.clip.modules():
            if isinstance(m, (nn.LayerNorm, clip.model.LayerNorm)):
                m.float()
                if m.weight is not None:
                    m.weight.data = m.weight.data.to(torch.float32)
                if m.bias is not None:
                    m.bias.data = m.bias.data.to(torch.float32)

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
        if x.device != self.device:
            x = x.to(self.device)
            
        # Force float32
        x = x.to(dtype=torch.float32)
        
        with torch.no_grad():
            # Ensure CLIP model is in float32 mode
            self.clip.visual.float()
            img_emb = self.clip.encode_image(x)
            img_emb = img_emb.to(dtype=torch.float32)
            
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
            text_emb = text_emb.to(dtype=torch.float32)
        return text_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of images using a CLIP backbone and returns intermediate layers."""
        if x.device != self.device:
            x = x.to(self.device)
            
        # Force float32
        x = x.to(dtype=torch.float32)
        
        # Ensure CLIP model is in float32 mode
        self.clip.visual.float()
        self.clip.encode_image(x)
        
        # Stack and permute outputs
        x = torch.stack([h.output.to(dtype=torch.float32) for h in self.hooks], dim=2)[1:, :, :, :]
        x = torch.permute(x, (1, 2, 0, 3))
        
        return x


class DINOv2Backbone(nn.Module):
    def __init__(
        self,
        dinov2_model: str = "dinov2_vitb14",
        intermediate_layers: tuple[int, ...] = tuple((i for i in range(12)))
    ) -> None:
        super().__init__()

        # Initialize DINOv2 pretrained model.
        self.dino = torch.hub.load("facebookresearch/dinov2", dinov2_model)
        self.intermediate_layers: tuple[int, ...] = intermediate_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x: tuple[torch.Tensor] = self.dino.get_intermediate_layers(x, self.intermediate_layers)
        x: torch.Tensor = torch.stack(x, dim=1)
        x = x.to(input_dtype)
        return x
