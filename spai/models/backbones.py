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
from torchvision.transforms import Compose, Resize, Normalize

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


class DINOv2FeatureEmbedding(nn.Module):
    """Projector that embeds DINOv2 features into a lower-dimensional space."""
    def __init__(self, model_name="dinov2_vitg14", 
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
            
        # Print model info for debugging
        logger.info(f"Loaded DINOv2 model {model_name} with output dim {self.output_dim}")

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

        # Match input dtype with model parameters
        model_dtype = next(self.dino_model.parameters()).dtype
        processed_images = processed_images.to(model_dtype)

        # Extract features using DINOv2
        with torch.no_grad():
            # Get intermediate layer outputs
            outputs = self.dino_model.get_intermediate_layers(processed_images, n=1)
            features = outputs[0]  # Get the last layer output

        # Convert features to float32 for downstream processing
        features = features.float()
        
        return features

    def preprocess_image(self, image_tensor):
        # DINOv2 expects images normalized with ImageNet stats
        preprocess = Compose([
            Resize((224, 224)),
            Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225))
        ])
        return preprocess(image_tensor)
    