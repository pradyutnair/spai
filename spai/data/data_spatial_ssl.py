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

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data.transforms import str_to_pil_interp


class SpatialMaskGenerator:
    def __init__(self,
                 mask_ratio=0.5,
                 mask_patch_size=32):
        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size

    def __call__(self, img_size):
        """Generate random spatial mask for given image size"""
        H, W = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        # Calculate number of patches in each dimension
        h_patches = (H + self.mask_patch_size - 1) // self.mask_patch_size
        w_patches = (W + self.mask_patch_size - 1) // self.mask_patch_size

        # Create a mask for each patch (1=keep, 0=mask)
        mask = torch.ones(h_patches, w_patches)
        mask = torch.bernoulli(mask * (1 - self.mask_ratio))

        # Upscale mask to image size
        mask = mask.repeat_interleave(self.mask_patch_size, dim=0)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=1)

        # Crop to exact image size
        mask = mask[:H, :W]

        return mask.numpy().astype(np.float32)


class SpatialSSLTransform:
    def __init__(self, config):
        self.img_size = config.DATA.IMG_SIZE
        self.mask_ratio = getattr(config.DATA, 'MASK_RATIO', 0.5)
        self.mask_patch_size = getattr(config.DATA, 'MASK_PATCH_SIZE', 32)
        
        # Ensure min_crop_scale exists in config
        min_crop_scale = getattr(config.DATA, 'MIN_CROP_SCALE', 0.2)
        interpolation = getattr(config.DATA, 'INTERPOLATION', 'bicubic')
        
        self.transform_img = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(min_crop_scale, 1.), 
                              interpolation=str_to_pil_interp(interpolation)),
            T.RandomHorizontalFlip(),
        ])
        
        self.spatial_mask_generator = SpatialMaskGenerator(
            mask_ratio=self.mask_ratio,
            mask_patch_size=self.mask_patch_size
        )

    def __call__(self, img):
        img = self.transform_img(img)  # PIL Image (HxWxC, 0-255), no normalization
        img = T.ToTensor()(img)  # Tensor (CxHxW, 0-1)
        
        # Generate spatial mask
        mask = self.spatial_mask_generator(self.img_size)
        mask = torch.from_numpy(mask).float()  # Convert to float tensor
        
        # Add channel dimension and broadcast to match image channels
        mask = mask.unsqueeze(0).expand(img.size(0), -1, -1)  # (C, H, W)
        
        # Apply mask to create masked image
        img_lq = img * mask
        
        return img, img_lq, mask


def collate_fn_spatial(batch):
    """Custom collate function for spatial SSL data"""
    # The batch contains tuples of (img, img_lq, mask, class_label)
    # We need to handle this similar to data_mfm.py
    batch_num = len(batch)
    
    # Extract each component
    imgs = [batch[i][0][0] for i in range(batch_num)]  # Original images
    img_lqs = [batch[i][0][1] for i in range(batch_num)]  # Masked images  
    masks = [batch[i][0][2] for i in range(batch_num)]  # Masks
    labels = [batch[i][1] for i in range(batch_num)]  # Class labels
    
    # Collate each component
    return [
        default_collate(imgs),
        default_collate(img_lqs), 
        default_collate(masks),
        default_collate(labels)
    ]


def build_loader_spatial_ssl(config, logger):
    """Build dataloader for spatial SSL training"""
    transform = SpatialSSLTransform(config)
    logger.info(f'Spatial SSL data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), 
                               rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, 
                          num_workers=config.DATA.NUM_WORKERS, pin_memory=True, 
                          drop_last=True, collate_fn=collate_fn_spatial)
    
    return dataloader
