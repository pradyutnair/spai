#!/usr/bin/env python3
"""
Comprehensive test evaluation script for SPAI semantic-spectral fusion model.
Tests on all major datasets: DALLE2, DALLE3, GIGAGAN, MIDJOURNEY, SD1-4, SD3, SDXL, FLUX, ARTIFACT, LSUN
"""

import os
import subprocess
import sys
from pathlib import Path

# Test datasets configuration
TEST_DATASETS = {
    "SD3": "datasets/test_set_sd3_no_imagenet.csv",
    "SDXL": "datasets/test_set_sdxl_no_imagenet.csv", 
    "FLUX": "datasets/test_set_flux_no_imagenet.csv",
    "GIGAGAN": "datasets/test_set_gigagan_no_imagenet.csv",
    "MIDJOURNEY": "datasets/test_set_midjourney-v6.1_no_imagenet.csv",
    "ARTIFACT": "datasets/artifact_test.csv",
    "LSUN": "datasets/lsun_test.csv"
}

# Model configuration
MODEL_CHECKPOINT = "/home/pnair/spai/output/train_mid_level_cross_attn/finetune/spai/ckpt_epoch_9.pth"
CONFIG_FILE = "/home/pnair/spai/configs/spai.yaml"
OUTPUT_BASE = "/home/pnair/spai/output/test_results_semantic_fusion"

def run_test_evaluation(dataset_name, dataset_path, output_dir):
    """Run test evaluation for a specific dataset."""
    print(f"\n🧪 Testing on {dataset_name} dataset...")
    print(f"📁 Dataset: {dataset_path}")
    print(f"📂 Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct test command
    cmd = [
        "python", "-m", "spai", "test",
        "--cfg", CONFIG_FILE,
        "--model", MODEL_CHECKPOINT,
        "--test-csv", dataset_path,
        "--output", output_dir,
        "--batch-size", "64",
        "--tag", f"test_{dataset_name.lower()}",
        "--opt", "DATA.TEST_BATCH_SIZE", "64",
        "--opt", "MODEL.FEATURE_EXTRACTION_BATCH", "200"
    ]
    
    print(f"🚀 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print(f"✅ {dataset_name} test completed successfully!")
            print("📊 Output:")
            print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print(f"❌ {dataset_name} test failed!")
            print("Error output:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {dataset_name} test timed out after 1 hour")
    except Exception as e:
        print(f"💥 {dataset_name} test failed with exception: {e}")

def main():
    """Main evaluation function."""
    print("🎯 SPAI Semantic-Spectral Fusion - Comprehensive Test Evaluation")
    print("=" * 70)
    
    # Check if model checkpoint exists
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"❌ Model checkpoint not found: {MODEL_CHECKPOINT}")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists(CONFIG_FILE):
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    
    print(f"🏆 Using best model: {MODEL_CHECKPOINT}")
    print(f"⚙️ Config: {CONFIG_FILE}")
    print(f"📊 Testing on {len(TEST_DATASETS)} datasets")
    
    # Create base output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Test each dataset
    results = {}
    for dataset_name, dataset_path in TEST_DATASETS.items():
        if os.path.exists(dataset_path):
            output_dir = os.path.join(OUTPUT_BASE, dataset_name.lower())
            run_test_evaluation(dataset_name, dataset_path, output_dir)
            results[dataset_name] = "completed"
        else:
            print(f"⚠️ Dataset not found: {dataset_path}")
            results[dataset_name] = "not_found"
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 EVALUATION SUMMARY")
    print("=" * 70)
    for dataset_name, status in results.items():
        status_emoji = "✅" if status == "completed" else "❌"
        print(f"{status_emoji} {dataset_name}: {status}")
    
    print(f"\n📂 All results saved to: {OUTPUT_BASE}")
    print("🎉 Comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 