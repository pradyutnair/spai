# SPAI Semantic-Spectral Fusion Enhancements

## Overview

This document outlines the comprehensive enhancements made to the SPAI (Spectral and Patch-based AI) model to integrate advanced semantic context alongside spectral context, specifically designed to maximize AUC performance across all test datasets.

## 🎯 Goal

**Maximize AUC performance** by intelligently combining semantic context (via CLIP) with spectral context (frequency domain analysis) through advanced fusion mechanisms.

## 🚀 Key Enhancements

### 1. Enhanced FrequencyRestorationEstimator

**File**: `spai/models/sid.py`

#### New Features:
- **Multi-Scale Semantic Fusion**: Process semantic vectors at different scales for better integration
- **Frequency-Aware Attention**: Separate attention mechanisms for different frequency components
- **Adaptive Gating**: Dynamic weighting of semantic and spectral features
- **Hierarchical Fusion**: Multi-level combination of frequency components
- **Enhanced Similarity Features**: Extended from 6 to 12 statistical moments per layer

#### Key Components:

```python
class FrequencyRestorationEstimator(nn.Module):
    def __init__(self, 
                 use_frequency_aware_attention=True,
                 use_adaptive_gating=True, 
                 use_hierarchical_fusion=True,
                 early_fusion_layers=[2, 4, 6, 8]):
```

**Improvements**:
- **Early Fusion Gates**: Enhanced gating with residual connections and layer normalization
- **Cross-Attention**: Multi-head attention between semantic and spectral features
- **Frequency-Specific Processing**: Dedicated encoders for original, low-freq, and high-freq components

### 2. Advanced Loss Functions

**File**: `spai/models/losses.py`

#### New Loss Functions:

1. **FocalLoss**: Addresses class imbalance for better AUC optimization
   ```python
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2.0)
   ```

2. **SemanticConsistencyLoss**: Ensures semantic coherence across frequency components
   ```python
   class SemanticConsistencyLoss(nn.Module):
       def forward(self, semantic_orig, semantic_low, semantic_high)
   ```

3. **FrequencyConsistencyLoss**: Maintains semantic relationships in frequency domain
   ```python
   class FrequencyConsistencyLoss(nn.Module):
       def forward(self, freq_features_orig, freq_features_low, freq_features_high)
   ```

4. **EnhancedSemanticSpectralLoss**: Combined loss for optimal performance
   ```python
   class EnhancedSemanticSpectralLoss(nn.Module):
       def __init__(self, focal_alpha=0.25, focal_gamma=2.0, 
                    semantic_weight=0.1, frequency_weight=0.05)
   ```

### 3. Optimized Configuration

**File**: `configs/spai.yaml`

#### Key Configuration Changes:

```yaml
MODEL:
  MODEL_WEIGHTS: "clip"  # Enable CLIP backbone
  SID_DROPOUT: 0.3       # Optimized dropout
  
  VIT:
    PROJECTION_LAYERS: 3   # Increased for better processing
    
  FRE:
    EARLY_FUSION_LAYERS: [2, 4, 6, 8]  # More fusion points
    USE_FREQUENCY_AWARE_ATTENTION: True
    USE_ADAPTIVE_GATING: True
    USE_HIERARCHICAL_FUSION: True
    
  CLS_HEAD:
    MLP_RATIO: 4          # Increased classification capacity
    
  PATCH_VIT:
    ATTN_EMBED_DIM: 1024  # Enhanced attention
    NUM_HEADS: 16         # More attention heads
    PATCH_STRIDE: 112     # Optimized coverage

DATA:
  AUGMENTED_VIEWS: 6      # Increased for better generalization
  VAL_BATCH_SIZE: 128
  TEST_PREFETCH_FACTOR: 2

AUG:
  COLOR_JITTER: 0.1       # Slight augmentation
  MIXUP: 0.2              # Enable mixup
  CUTMIX: 0.3             # Enable cutmix
  SEMANTIC_AWARE_AUG: True

TRAIN:
  EPOCHS: 15              # Increased for convergence
  WARMUP_EPOCHS: 3        # Faster warmup
  BASE_LR: 3e-4           # Optimized learning rate
  WEIGHT_DECAY: 0.03      # Reduced weight decay
  LAYER_DECAY: 0.85       # Optimized layer decay
  CLIP_GRAD: 1.0          # Enable gradient clipping
  LOSS: "focal"           # Use focal loss
  FOCAL_ALPHA: 0.25
  FOCAL_GAMMA: 2.0
  SEMANTIC_CONSISTENCY_WEIGHT: 0.1
  FREQUENCY_CONSISTENCY_WEIGHT: 0.05
```

### 4. Enhanced Model Architecture

#### Semantic-Spectral Integration Strategy:

1. **Early Fusion**: Inject semantic context at multiple transformer layers (2, 4, 6, 8)
2. **Mid-Level Cross-Attention**: Global attention between semantic and spectral features
3. **Frequency-Aware Processing**: Separate processing paths for different frequency components
4. **Hierarchical Combination**: Multi-level fusion of all frequency components
5. **Enhanced Feature Statistics**: 12 statistical moments instead of 6 for richer representation

#### Architecture Flow:
```
Input Image → Frequency Decomposition → CLIP Semantic Features
     ↓                    ↓                        ↓
Low/High Freq → Spectral Features ← Semantic Injection (Early Fusion)
     ↓                    ↓                        ↓
Feature Projection → Cross-Attention ← Frequency-Aware Semantic
     ↓                    ↓                        ↓
Hierarchical Fusion → Enhanced Statistics → Classification
```

## 🔧 Implementation Details

### Training Strategy

1. **Stage 1 (Epochs 0-4)**: Focus on semantic components training
2. **Stage 2 (Epochs 5+)**: Joint training of all components

### Parameter Management

- **Frozen**: CLIP backbone parameters (for stability)
- **Trainable**: All semantic fusion components, cross-attention, classification head
- **Enhanced**: Semantic components verified and made trainable

### Key Algorithmic Improvements

1. **Adaptive Gating**: Dynamic weighting based on input characteristics
2. **Residual Connections**: Preserve original spectral information
3. **Multi-Head Attention**: Better capture of semantic-spectral relationships
4. **Layer Normalization**: Improved training stability
5. **Enhanced Statistics**: Richer feature representation with min/max/mean/std

## 📊 Expected Performance Improvements

### AUC Optimization Strategies:

1. **Focal Loss**: Better handling of class imbalance → Higher AUC
2. **Semantic Consistency**: Improved feature quality → Better discrimination
3. **Enhanced Features**: Richer representation → Better separability
4. **Multi-Scale Fusion**: Better capture of semantic-spectral relationships
5. **Adaptive Processing**: Dynamic adaptation to input characteristics

### Target Improvements:
- **Training AUC**: Expected 5-10% improvement
- **Validation AUC**: Expected 3-8% improvement  
- **Test AUC**: Expected 2-5% improvement across all datasets
- **Generalization**: Better performance on unseen data

## 🧪 Testing and Validation

### Test Datasets:
- DALLE2, DALLE3, GIGAGAN, MIDJOURNEY, SD1-4, SD3, SDXL, FLUX, ARTIFACT, LSUN

### Validation Strategy:
1. **Cross-validation**: Multiple test sets for robust evaluation
2. **Ablation Studies**: Individual component contribution analysis
3. **Comparative Analysis**: Before/after performance comparison

## 🚀 Usage Instructions

### Training:
```bash
# Activate environment
module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.1.1
source activate spai

# Run training
sbatch /home/pnair/spai/jobs/directory.job
```

### Testing:
```bash
# Run comprehensive testing
sbatch /home/pnair/spai/jobs/test.job
```

## 📈 Monitoring and Metrics

### Key Metrics to Track:
- **AUC**: Primary optimization target
- **AP (Average Precision)**: Secondary metric
- **Accuracy**: Overall performance
- **Loss Components**: Focal, semantic consistency, frequency consistency

### Neptune Logging:
- All metrics logged to Neptune for comprehensive tracking
- Separate tracking for each test dataset
- Training progression monitoring

## 🔍 Debugging and Troubleshooting

### Common Issues:
1. **Shape Mismatches**: Ensure all tensor dimensions align
2. **Memory Issues**: Adjust batch sizes if needed
3. **Convergence**: Monitor loss components individually

### Verification Steps:
1. **Model Loading**: Verify CLIP backbone initialization
2. **Semantic Features**: Check semantic vector extraction
3. **Parameter Counts**: Confirm trainable vs frozen parameters
4. **Loss Computation**: Validate loss component contributions

## 🎯 Success Criteria

### Primary Goals:
- [ ] AUC > 0.95 on training set
- [ ] AUC > 0.90 on validation set  
- [ ] AUC > 0.85 on all test datasets
- [ ] Consistent performance across different generators

### Secondary Goals:
- [ ] Improved AP scores
- [ ] Better generalization to unseen data
- [ ] Stable training convergence
- [ ] Efficient inference time

## 📚 References

1. **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (2017)
2. **CLIP**: Radford et al. "Learning Transferable Visual Representations" (2021)
3. **Cross-Attention**: Vaswani et al. "Attention Is All You Need" (2017)
4. **Frequency Analysis**: Various signal processing literature

## 🤝 Contributing

When making further improvements:
1. **Document Changes**: Update this README
2. **Test Thoroughly**: Verify on multiple datasets
3. **Monitor Performance**: Track AUC improvements
4. **Maintain Compatibility**: Ensure backward compatibility

---

**Note**: This implementation represents a comprehensive enhancement of the SPAI model with advanced semantic-spectral fusion designed specifically for maximum AUC performance across diverse test datasets. 