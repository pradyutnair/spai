import os
import sys
import argparse
import torch
import torch.nn as nn
import time
from pathlib import Path

# Add the parent directory to Python path - CRITICAL
sys.path.insert(0, '/home/scur2605')

# Local import for the combined model
from semantic_pipeline.combined_model import build_combined_model

# SPAI imports
from spai.config import get_config
from spai.data import build_loader_finetune
from spai.logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser('Combined model training script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--spai-model', type=str, required=True, help='path to SPAI model weights')
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset CSV file')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--output', type=str, default='output', help='output directory')
    parser.add_argument('--tag', type=str, default='default', help='experiment tag')
    parser.add_argument('--data-workers', type=int, default=4, help='data loader workers')
    parser.add_argument('--save-all', action='store_true', help='save all checkpoints')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output) / args.tag
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logger
    logger = create_logger(output_dir=output_dir, name="train")
    logger.info("Starting training")
    
    # Get configuration
    config = get_config({"cfg": args.cfg})
    
    # Update config with command line args
    config.defrost()
    config.DATA.BATCH_SIZE = args.batch_size
    config.DATA.DATA_PATH = args.data_path
    config.DATA.NUM_WORKERS = args.data_workers
    config.TRAIN.EPOCHS = args.epochs
    config.TRAIN.BASE_LR = args.lr
    config.OUTPUT = str(output_dir)
    config.freeze()
    
    # Print config summary
    logger.info(f"Configuration: {config.dump()}")
    
    # Build data loaders using SPAI's pipeline
    logger.info(f"Building datasets from {args.data_path}")
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader_finetune(config, logger)
    
    logger.info(f"Dataset loaded: {len(dataset_train)} training samples, {len(dataset_val)} validation samples")
    
    # Create the combined model
    logger.info(f"Creating combined model with SPAI from {args.spai_model}")
    model = build_combined_model(
        spai_path=args.spai_model,
        semantic_output_dim=1096,
        hidden_dims=[1024, 512],
        output_classes=2
    )
    
    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get trainable parameters (only semantic projection and fusion MLP)
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Training {sum(p.numel() for p in trainable_params)} parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.TRAIN.BASE_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.TRAIN.EPOCHS
    )
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(config.TRAIN.EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (images, targets, _) in enumerate(data_loader_train):
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Debug on first batch to see types
            if batch_idx == 0:
                print(f"DEBUG - Images dtype: {images.dtype}, shape: {images.shape}")
                print(f"DEBUG - Targets dtype: {targets.dtype}, shape: {targets.shape}")
            
            # Convert types if needed
            images = images.float()  # Force float32
            if targets.dim() > 1 and targets.shape[1] > 1:
                # One-hot encoded targets
                targets = targets.argmax(dim=1)
            targets = targets.long()  # Force int64 for classification
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Debug output type
            if batch_idx == 0:
                print(f"DEBUG - Outputs dtype: {outputs.dtype}, shape: {outputs.shape}")
            
            # Force float32 for outputs
            outputs = outputs.float()
            
            # Calculate loss
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 20 == 0:
                logger.info(f'Epoch {epoch+1}/{config.TRAIN.EPOCHS} | '
                           f'Batch {batch_idx}/{len(data_loader_train)} | '
                           f'Loss: {loss.item():.4f} | '
                           f'Acc: {100.*correct/total:.2f}%')
        
        train_acc = 100. * correct / total
        train_loss = train_loss / len(data_loader_train)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets, _ in data_loader_val:
                images, targets = images.to(device), targets.to(device)
                
                # Handle one-hot encoded targets
                if targets.dim() > 1 and targets.shape[1] > 1:
                    targets = targets.argmax(dim=1)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        val_loss = val_loss / len(data_loader_val)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        epoch_time = time.time() - start_time
        logger.info(f'Epoch {epoch+1}/{config.TRAIN.EPOCHS} completed in {epoch_time:.1f}s | '
                   f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                   f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save checkpoint if better than previous best or if save_all is True
        if val_acc > best_acc or args.save_all:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'acc': val_acc,
            }
            
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = output_dir / 'best_model.pth'
                logger.info(f'New best model! Saving to {checkpoint_path}')
            elif args.save_all:
                checkpoint_path = output_dir / f'epoch_{epoch}.pth'
                logger.info(f'Saving checkpoint to {checkpoint_path}')
                
            torch.save(state, checkpoint_path)
    
    logger.info(f'Training complete. Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()