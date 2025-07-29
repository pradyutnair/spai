import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .mfm import SwinTransformerForMFM, VisionTransformerForMFM, VisionTransformerDecoderForMFM, ResNetForMFM
from timm.models.resnet import Bottleneck, ResNet
from functools import partial

class RandomImageMasking(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_ratio = 0.5 # config.DATA.MASK_RATIO
        self.mask_patch_size = 32  # config.DATA.MASK_PATCH_SIZE
        self.normalize_img = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def random_masking(self, x):
        """Apply random masking to input image"""
        B, C, H, W = x.shape

        # Calculate number of patches in each dimension
        h_patches = (H + self.mask_patch_size - 1) // self.mask_patch_size
        w_patches = (W + self.mask_patch_size - 1) // self.mask_patch_size

        # Create a mask for each patch
        mask = torch.ones(B, h_patches, w_patches, device=x.device)
        mask = torch.bernoulli(mask * (1 - self.mask_ratio))  # 1=keep, 0=mask

        # Upscale mask to image size
        mask = mask.repeat_interleave(self.mask_patch_size, dim=1)
        mask = mask.repeat_interleave(self.mask_patch_size, dim=2)

        # Crop to image size
        mask = mask[:, :H, :W]

        # Add channel dimension and broadcast
        mask = mask.unsqueeze(1)  # shape: [B, 1, H, W]
        masked_x = x * mask

        return masked_x, mask

    def forward(self, x, x_lq=None, mask=None):
        # Apply random masking
        x_masked, mask = self.random_masking(x)

        # Normalize images
        x = self.normalize_img(x)
        x_masked = self.normalize_img(x_masked)

        return x_masked, mask


class ImageMaskedMFM(nn.Module):
    def __init__(self, encoder, encoder_stride, decoder, config):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.masking = RandomImageMasking(config)
        self.recover_target_type = config.MODEL.RECOVER_TARGET_TYPE
        
        # Loss function - can use same frequency loss or switch to MSE
        self.criterion = nn.MSELoss()
        
        if self.decoder is None:
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.encoder.num_features,
                    out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(self.encoder_stride),
            )

    def forward(self, x, x_lq=None, mask=None):
        # Apply random masking
        x_masked, mask = self.masking(x)
        
        # Get encoder features from masked image
        z = self.encoder(x_masked, None)
        
        # Reconstruct original image
        x_rec = self.decoder(z)
        
        # Calculate reconstruction loss
        if self.recover_target_type == 'masked':
            # Only compute loss on masked regions
            loss = self.criterion(x_rec * (1 - mask), x * (1 - mask))
        elif self.recover_target_type == 'normal':
            # Compute loss on entire image
            loss = self.criterion(x_rec, x)
        else:
            raise NotImplementedError
        
        return loss


def build_image_masked_mfm(config):
    """Build the image-masked MFM model using same encoder/decoder architecture"""
    model_type = config.MODEL.TYPE
    
    # Build encoder (same as original MFM)
    if model_type == 'swin':
        encoder = SwinTransformerForMFM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            config=config)
        encoder_stride = 32
        decoder = None
    elif model_type == 'vit':
        encoder = VisionTransformerForMFM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_fixed_pos_emb=config.MODEL.VIT.USE_FPE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
            config=config)
        encoder_stride = 16
        if config.MODEL.VIT.DECODER.DEPTH > 0:
            decoder = VisionTransformerDecoderForMFM(
                img_size=config.DATA.IMG_SIZE,
                patch_size=config.MODEL.VIT.PATCH_SIZE,
                in_chans=config.MODEL.VIT.IN_CHANS,
                num_classes=0,
                embed_dim=config.MODEL.VIT.DECODER.EMBED_DIM,
                depth=config.MODEL.VIT.DECODER.DEPTH,
                num_heads=config.MODEL.VIT.DECODER.NUM_HEADS,
                mlp_ratio=config.MODEL.VIT.MLP_RATIO,
                qkv_bias=config.MODEL.VIT.QKV_BIAS,
                drop_rate=config.MODEL.DROP_RATE,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=config.MODEL.VIT.INIT_VALUES,
                use_abs_pos_emb=config.MODEL.VIT.USE_APE,
                use_fixed_pos_emb=config.MODEL.VIT.USE_FPE,
                use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
                use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
                use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING,
                config=config)
        else:
            decoder = None
    elif model_type == 'resnet':
        encoder = ResNetForMFM(
            block=Bottleneck,
            layers=config.MODEL.RESNET.LAYERS,
            in_chans=config.MODEL.RESNET.IN_CHANS,
            num_classes=0,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            config=config)
        encoder_stride = 32
        decoder = None
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    # Create image-masked MFM model
    model = ImageMaskedMFM(encoder=encoder, encoder_stride=encoder_stride, decoder=decoder, config=config)

    return model