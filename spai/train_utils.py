import torch
import numpy as np
import logging
from torch.cuda.amp import autocast, GradScaler
from spai.utils import inf_nan_to_num

logger = logging.getLogger(__name__)

def stable_forward_pass(model, batch, criterion, amp_enabled=True, amp_dtype=torch.float16):
    """
    Perform a stable forward pass with safeguards against NaN values
    
    Args:
        model: The PyTorch model
        batch: Input batch
        criterion: Loss function
        amp_enabled: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP
        
    Returns:
        tuple: (loss, outputs)
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        samples, targets, _ = batch
        batch_size = samples.size(0)
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
    else:
        logger.error("Unsupported batch format")
        raise ValueError("Unsupported batch format")
    
    # Run with appropriate precision
    if amp_enabled:
        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            # Process each batch view separately for stability
            outputs_views = []
            for i in range(samples.size(1)):
                view = samples[:, i, :, :, :].float()
                # Add debug prints
                print(f"🔄 Processing batch view {i} of shape {view.shape}")
                
                # Forward pass with NaN checking
                output = model(view)
                output = inf_nan_to_num(output)
                outputs_views.append(output)
            
            # Stack outputs and squeeze if needed
            outputs = torch.stack(outputs_views, dim=1)
            outputs = outputs if outputs.size(dim=1) > 1 else outputs.squeeze(dim=1)
            
            # Compute loss with NaN checking
            loss = criterion(outputs.squeeze(), targets)
            loss = inf_nan_to_num(loss)
    else:
        # Same process without AMP
        outputs_views = []
        for i in range(samples.size(1)):
            view = samples[:, i, :, :, :].float()
            print(f"🔄 Processing batch view {i} of shape {view.shape}")
            output = model(view)
            output = inf_nan_to_num(output)
            outputs_views.append(output)
        
        outputs = torch.stack(outputs_views, dim=1)
        outputs = outputs if outputs.size(dim=1) > 1 else outputs.squeeze(dim=1)
        loss = criterion(outputs.squeeze(), targets)
        loss = inf_nan_to_num(loss)
    
    return loss, outputs

def stable_backward_pass(loss, optimizer, scaler=None, max_grad_norm=1.0):
    """
    Perform a stable backward pass with gradient clipping
    
    Args:
        loss: The loss to backpropagate
        optimizer: The optimizer
        scaler: Optional AMP GradScaler
        max_grad_norm: Maximum gradient norm for clipping
    """
    # Skip if loss is NaN
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("Skipping backward pass due to NaN/Inf loss")
        optimizer.zero_grad()
        return False
    
    if scaler is not None:
        # AMP backward pass with gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(optimizer._amp_stash.all_params, max_norm=max_grad_norm)
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            return True
        else:
            logger.warning("Skipping step due to non-finite gradient norm")
            optimizer.zero_grad()
            return False
    else:
        # Standard backward pass with gradient clipping
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimizer.param_groups[0]['params'], max_norm=max_grad_norm
        )
        if torch.isfinite(grad_norm):
            optimizer.step()
            optimizer.zero_grad()
            return True
        else:
            logger.warning("Skipping step due to non-finite gradient norm")
            optimizer.zero_grad()
            return False
