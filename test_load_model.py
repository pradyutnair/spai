#!/usr/bin/env python3

import os
import sys
import torch
import argparse
import logging
from pathlib import Path

from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained, find_pretrained_checkpoints
from spai.models.sid import AdaptiveSemanticSpectralFusion

def test_model_loading(config_path, model_path, verbose=True):
    """Test if the model can be loaded without errors related to parameter size mismatches."""
    # Set up a basic logger
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger("test_model_loading")
    
    if verbose:
        logger.info(f"📋 Testing model loading with config: {config_path}")
        logger.info(f"📋 Model path: {model_path}")
        
    config = get_config({
        "cfg": str(config_path),
        "pretrained": str(model_path),
    })
    
    # Build the model
    if verbose:
        logger.info(f"🏗️ Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_cls_model(config)
    
    # Move to CPU or CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        logger.info(f"💻 Using device: {device}")
    model.to(device)
    
    # Try to load the pretrained model weights
    model_checkpoints = find_pretrained_checkpoints(config)
    if not model_checkpoints:
        logger.error(f"❌ No checkpoint found at {model_path}")
        return False
    
    if verbose:
        logger.info(f"📥 Loading checkpoint from {model_checkpoints[0]}")
    try:
        load_pretrained(config, model, logger, checkpoint_path=model_checkpoints[0], verbose=verbose)
        logger.info("✅ Model loaded successfully!")
        
        # Check for adaptive fusion module
        fusion_module = None
        if hasattr(model, 'mfvit') and hasattr(model.mfvit, 'semantic_fusion'):
            fusion_module = model.mfvit.semantic_fusion
            if isinstance(fusion_module, AdaptiveSemanticSpectralFusion):
                logger.info(f"✅ AdaptiveSemanticSpectralFusion module found")
                logger.info(f"   - Number of frequency bands: {fusion_module.num_frequency_bands}")
                logger.info(f"   - Fusion dimension: {fusion_module.fusion_dim}")
            else:
                logger.warning(f"⚠️ Semantic fusion module is not AdaptiveSemanticSpectralFusion")
        
        # Test a simple forward pass
        input_shape = (1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        if verbose:
            logger.info(f"🧪 Testing forward pass with input shape: {input_shape}")
        test_input = torch.randn(input_shape).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        if verbose:
            logger.info(f"📊 Output shape: {output.shape}")
        logger.info("✅ Forward pass successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model loading to verify fixes")
    parser.add_argument("--cfg", default="configs/spai.yaml", type=str, help="Path to config file")
    parser.add_argument("--model", default="weights/spai.pth", type=str, help="Path to model weights")
    parser.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    args = parser.parse_args()
    
    success = test_model_loading(args.cfg, args.model, verbose=not args.quiet)
    if success:
        print("🎉 All tests passed! The fix for parameter size mismatches is working.")
    else:
        print("❌ Tests failed. The model still has issues.")
        sys.exit(1) 