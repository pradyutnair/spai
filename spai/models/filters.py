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

from typing import Optional

import torch
from torch import fft
from torch import linalg
from torch.nn import functional as F


__author__: str = "Dimitrios Karageorgiou"
__email__: str = "dkarageo@iti.gr"
__version__: str = "1.0.0"
__revision__: int = 1


def filter_image_frequencies(
    image: torch.Tensor,
    mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Filter the image frequencies using the provided mask.

    :param image: B x C x H x W
    :param mask: H x W
    :returns: A tuple containing the low and high frequency components of the image.
    """
    # Ensure image and mask are in the same dtype
    image = image.to(mask.dtype)
    
    # Get image dimensions
    B, C, H, W = image.shape
    
    # Convert mask to float before interpolation
    mask = mask.float()
    
    # Resize mask to match image dimensions if needed
    if mask.shape != (H, W):
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode='nearest').squeeze(0).squeeze(0)
    
    # Apply FFT
    fft_image: torch.Tensor = torch.fft.fft2(image)
    fft_image = torch.fft.fftshift(fft_image, dim=(-2, -1))

    # Filter frequencies
    filtered_image: torch.Tensor = fft_image * mask.unsqueeze(0).unsqueeze(0)
    filtered_image = torch.fft.ifftshift(filtered_image, dim=(-2, -1))
    filtered_image = torch.fft.ifft2(filtered_image)
    filtered_image = torch.real(filtered_image)

    # Get high frequency component
    high_freq: torch.Tensor = image - filtered_image

    return filtered_image, high_freq


def generate_circular_mask(
    input_size: int,
    mask_radius_start: int,
    mask_radius_stop: Optional[int] = None,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    coordinates: torch.Tensor = generate_centered_2d_coordinates_grid(input_size, device)
    radius: torch.Tensor = linalg.vector_norm(coordinates, dim=-1)
    mask: torch.Tensor = torch.where(radius < mask_radius_start, 1, 0)
    if mask_radius_stop is not None:
        mask = torch.where(radius > mask_radius_stop, 1, mask)
    return mask


def generate_centered_2d_coordinates_grid(
    size: int,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    assert size % 2 == 0, "Input size must be even."
    coords_values: torch.Tensor = torch.arange(0, size // 2, dtype=torch.float, device=device)
    coords_values = torch.cat([coords_values.flip(dims=(0,)), coords_values], dim=0)
    coordinates_x: torch.Tensor = coords_values.unsqueeze(dim=0).expand(size, -1)
    coordinates_y: torch.Tensor = torch.t(coordinates_x)
    coordinates: torch.Tensor = torch.stack([coordinates_x, coordinates_y], dim=2)
    return coordinates
