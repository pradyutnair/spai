import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from config import get_config
from models.spatial_masking import build_image_masked_mfm
from data.data_spatial_ssl import SpatialMaskGenerator
from utils import load_checkpoint
from logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser('Spatial SSL Testing', add_help=False)
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint file')
    parser.add_argument('--input-dir', type=str, required=True, help='path to input images directory')
    parser.add_argument('--output-dir', type=str, required=True, help='path to save reconstructed images')
    parser.add_argument('--cfg', type=str, help='path to config file (optional, will try to load from checkpoint)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use')
    parser.add_argument('--apply-masking', action='store_true', help='apply spatial masking during testing')
    parser.add_argument('--save-masked', action='store_true', help='save masked input images too')
    parser.add_argument('--image-extensions', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'], 
                       help='image file extensions to process')
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, config_path=None, device='cuda'):
    """Load model and config from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    # Set weights_only=False to handle yacs.config.CfgNode in checkpoint
    # This is safe since we trust our own checkpoint files
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try to get config from checkpoint, fallback to provided config
    if 'config' in checkpoint and config_path is None:
        config = checkpoint['config']
        print("Using config from checkpoint")
    elif config_path is not None:
        # Load config from file
        args_dict = {'cfg': config_path}
        config = get_config(args_dict)
        print(f"Using config from: {config_path}")
    else:
        raise ValueError("No config found in checkpoint and no config file provided")
    
    # Build model
    model = build_image_masked_mfm(config)
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, config


def prepare_image_transforms(img_size=224):
    """Prepare image preprocessing transforms"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return transform


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # Clamp values to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0, 1)
    np_img = tensor.cpu().numpy().transpose(1, 2, 0)
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def get_image_files(input_dir, extensions):
    """Get all image files from directory"""
    input_path = Path(input_dir)
    image_files = []
    
    for ext in extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def test_spatial_masking(args):
    """Main testing function"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and config
    model, config = load_model_from_checkpoint(args.checkpoint, args.cfg, args.device)
    
    # Prepare transforms
    img_size = getattr(config.DATA, 'IMG_SIZE', 224)
    transform = prepare_image_transforms(img_size)
    
    # Initialize spatial mask generator if needed
    mask_generator = None
    if args.apply_masking:
        mask_ratio = getattr(config.DATA, 'MASK_RATIO', 0.5)
        mask_patch_size = getattr(config.DATA, 'MASK_PATCH_SIZE', 32)
        mask_generator = SpatialMaskGenerator(mask_ratio=mask_ratio, mask_patch_size=mask_patch_size)
        print(f"Applying spatial masking: ratio={mask_ratio}, patch_size={mask_patch_size}")
    
    # Get all image files
    image_files = get_image_files(args.input_dir, args.image_extensions)
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print("No images found! Check input directory and file extensions.")
        return
    
    # Initialize metrics
    total_loss = 0.0
    total_masked_loss = 0.0
    total_images = 0
    masked_images = 0
    criterion = torch.nn.MSELoss()
    
    # Process images
    results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                # Load and preprocess image
                pil_img = Image.open(img_path)
                img_tensor = transform(pil_img).unsqueeze(0).to(args.device)  # Add batch dimension
                
                # Apply masking if requested
                if args.apply_masking and mask_generator is not None:
                    # Get image size from tensor (H, W)
                    _, _, H, W = img_tensor.shape
                    mask = mask_generator((H, W))
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(args.device)
                    img_lq = img_tensor * mask_tensor  # Apply mask (0 = masked, 1 = keep)
                else:
                    img_lq = img_tensor
                    mask_tensor = torch.ones_like(img_tensor[:, :1])  # Dummy mask
                
                # Forward pass through model
                # Get encoder features
                z = model.encoder(img_lq, img_lq)
                # Reconstruct image
                img_rec = model.decoder(z)
                
                # Calculate loss
                loss = criterion(img_rec, img_tensor)
                total_loss += loss.item()
                total_images += 1
                
                # Save results
                img_name = img_path.stem
                
                # Save original image
                original_save_path = os.path.join(args.output_dir, f"{img_name}_original.png")
                tensor_to_pil(img_tensor[0]).save(original_save_path)
                
                # Save reconstructed image (from original input)
                reconstructed_save_path = os.path.join(args.output_dir, f"{img_name}_reconstructed.png")
                tensor_to_pil(img_rec[0]).save(reconstructed_save_path)
                
                # Always save masked image and its reconstruction if masking was applied
                if args.apply_masking:
                    # Save masked input image
                    masked_save_path = os.path.join(args.output_dir, f"{img_name}_masked.png")
                    tensor_to_pil(img_lq[0]).save(masked_save_path)
                    
                    # Generate reconstruction from masked input
                    z_masked = model.encoder(img_lq, img_lq)
                    img_rec_masked = model.decoder(z_masked)
                    
                    # Save masked reconstruction
                    masked_rec_save_path = os.path.join(args.output_dir, f"{img_name}_masked_reconstructed.png")
                    tensor_to_pil(img_rec_masked[0]).save(masked_rec_save_path)
                    
                    # Calculate loss for masked reconstruction
                    masked_loss = criterion(img_rec_masked, img_tensor)
                    total_masked_loss += masked_loss.item()
                    masked_images += 1
                elif args.save_masked:
                    # If no masking but user wants to see input
                    input_save_path = os.path.join(args.output_dir, f"{img_name}_input.png")
                    tensor_to_pil(img_lq[0]).save(input_save_path)
                
                # Store results
                result = {
                    'image_name': img_name,
                    'loss': loss.item(),
                    'original_path': str(img_path),
                    'reconstructed_path': reconstructed_save_path
                }
                
                if args.apply_masking:
                    result['masked_path'] = masked_save_path
                    result['masked_reconstructed_path'] = masked_rec_save_path
                    result['masked_loss'] = masked_loss.item()
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate and display metrics
    if total_images > 0:
        avg_loss = total_loss / total_images
        print(f"\n=== Testing Results ===")
        print(f"Total images processed: {total_images}")
        print(f"Average reconstruction loss (original->reconstructed): {avg_loss:.6f}")
        print(f"Average PSNR (original->reconstructed): {-10 * np.log10(avg_loss):.2f} dB")
        
        if masked_images > 0:
            avg_masked_loss = total_masked_loss / masked_images
            print(f"Average reconstruction loss (masked->reconstructed): {avg_masked_loss:.6f}")
            print(f"Average PSNR (masked->reconstructed): {-10 * np.log10(avg_masked_loss):.2f} dB")
        
        # Save results to JSON
        results_summary = {
            'total_images': total_images,
            'average_loss': avg_loss,
            'average_psnr': -10 * np.log10(avg_loss),
            'checkpoint_path': args.checkpoint,
            'config': config.dump() if hasattr(config, 'dump') else str(config),
            'per_image_results': results
        }
        
        if masked_images > 0:
            results_summary['masked_images'] = masked_images
            results_summary['average_masked_loss'] = avg_masked_loss
            results_summary['average_masked_psnr'] = -10 * np.log10(avg_masked_loss)
        
        results_path = os.path.join(args.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        print(f"Reconstructed images saved to: {args.output_dir}")
        
    else:
        print("No images were successfully processed!")


if __name__ == '__main__':
    args = parse_args()
    test_spatial_masking(args)
