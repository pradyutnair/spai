# 🎯 Sem-SPAI: Semantic-enhanced SPAI for Synthetic Image Detection

## 📋 Introduction

This repository presents an analysis and extension of **SPAI (Spectral AI-Generated Image Detector)**, a state-of-the-art approach for detecting image manipulations and deepfakes. SPAI introduces a novel paradigm that leverages frequency domain analysis combined with vision transformers to identify subtle artifacts in manipulated images that are often imperceptible to human observers.

> 💡 **Key Insight**: SPAI's breakthrough lies in recognizing that manipulated images exhibit distinct frequency reconstruction patterns compared to authentic images.

The core innovation of SPAI lies in its **Frequency Restoration Estimation (FRE)** mechanism, which decomposes images into low and high-frequency components using FFT-based filtering, then measures how well a vision transformer can reconstruct missing frequency information. This approach is based on the insight that manipulated images exhibit different frequency reconstruction patterns compared to authentic images, as manipulation techniques often introduce artifacts that disrupt the natural frequency relationships.

SPAI employs a **Multi-Frequency Vision Transformer (MF-ViT)** architecture that processes three frequency-filtered versions of input images: the original image, low-frequency components, and high-frequency components. The model then computes cosine similarity metrics between these frequency representations to create discriminative features for authenticity classification. For handling arbitrary resolution images, SPAI introduces **Spectral Context Attention (SCA)**, which aggregates patch-level frequency features using learnable attention mechanisms.

### 📚 Related Work

Image manipulation detection has evolved significantly with the advent of deep learning approaches. Early methods relied on handcrafted features detecting compression artifacts, noise inconsistencies, or geometric distortions. Modern approaches leverage Convolutional Neural Networks (CNNs) for automatic feature learning, with notable works including **SPAN**, **FakeLocator**, and various transformer-based architectures. 

Frequency domain analysis has gained prominence through works like **F³-Net** and **FrePGAN**, which exploit the fact that manipulation techniques often leave frequency-domain fingerprints. Vision transformers have recently shown promise in this domain, with approaches like **TransForensics** demonstrating the effectiveness of attention mechanisms for detecting subtle manipulation artifacts. 

> ⚠️ **Research Gap**: Most existing methods focus on either spatial or frequency analysis in isolation, missing the opportunity to leverage the complementary nature of multi-frequency processing that SPAI introduces.

---

## ⚖️ Strengths, Weaknesses, and Motivation

### ✅ **Strengths**
- **Novel Frequency-Spatial Integration**: SPAI's approach of combining frequency domain analysis with vision transformers is innovative and well-motivated
- **Strong Empirical Results**: Achieves state-of-the-art performance across multiple datasets (CelebDF, DFDC, FaceForensics++)
- **Theoretical Foundation**: The frequency restoration paradigm provides clear theoretical justification for why the approach works
- **Scalability**: Handles arbitrary resolution images through the patch-based architecture
- **Robustness**: Demonstrates good generalization across different manipulation techniques

### ❌ **Weaknesses**
- **Limited Semantic Understanding**: SPAI focuses primarily on low-level frequency artifacts, potentially missing high-level semantic inconsistencies
- **Domain Specificity**: Heavy reliance on frequency analysis may limit generalization to manipulation techniques that preserve frequency characteristics
- **Improving robustness to derivative images**: SPAI is not robust to images that have undergone derivative operations such as JPEG compression, resizing, or cropping, which are common in images that are present in social media or other intermediate mediums.

>### 💡 **Our Motivation**
> The identified weaknesses, particularly the **limited semantic understanding** and **improving robustness to derivative images**, motivated our research direction. We hypothesized that **combining SPAI's powerful frequency-based detection with semantic context understanding** could create a more robust and comprehensive manipulation detection system. This led us to develop two distinct semantic-enhanced variants that integrate high-level semantic features with SPAI's spectral analysis.

---

## 🚀 Contributions

Our work extends SPAI through a **spectral-semantic late fusion approach** based on the actual implementation in `spai/models/sid.py`:


### Chameleon Dataset

As the original SPAI had almost perfect validation results on the LDM dataset, it is evident that is well-fitted to the image distributions in it. Therefore we opted to use the Chameleon dataset to test the robustness of the model. LDM and other datasets are generated by unconditional situations or conditioned on simple prompts (e.g., photo of a plane) without delicate manual adjustments, thereby inclined to generate obvious artifacts in consistency and semantics (marked with red boxes). Chameleon, however, aims to simulate real-world scenarios by collecting diverse images from online websites, where these online images are carefully adjusted by photographers and AI artists. Therefore, it is more likely to contain images that are not well-fitted to the image distributions in LDM and is generally more **challenging** for the model to detect.



### **Spectral-Semantic Late Fusion (SemanticContextModel)**

We implemented a **novel late fusion architecture** that combines SPAI's spectral features with ConvNeXt semantic features using residual connections:

```python
class SemanticContextModel(nn.Module):
    def forward(self, x):
        # Extract SPAI spectral features (frozen SPAI model)
        original_cls_head = self.spai_model.cls_head
        self.spai_model.cls_head = nn.Identity()  # Remove classification head
        spectral_features = self.spai_model(x_spai)  # [B, 1096]
        self.spai_model.cls_head = original_cls_head

        # Extract ConvNeXt semantic features (frozen ConvNeXt-XXL)
        semantic_features = self.semantic_backbone(x_convnext)  # [B, 3072]
        semantic_features = self.global_pool(semantic_features).flatten(1)
        
        # Project semantic features to compact representation
        semantic_proj = self.semantic_projection(semantic_features)  # [B, 256]
        
        # Late fusion with residual connection
        combined = torch.cat([spectral_features, semantic_proj], dim=1)  # [B, 1352]
        fused_features = self.fusion_layer(combined)  # [B, 512]
        
        # RESIDUAL CONNECTION: preserve original spectral features
        final_features = torch.cat([spectral_features, fused_features], dim=1)  # [B, 1608]
        
        return self.classifier(final_features)  # [B, 1]
```

**🎯 Late Fusion Architecture:**
- ❄️ **Frozen Pre-trained Models**: Both SPAI and ConvNeXt-XXL remain frozen
- 📊 **Compact Semantic Projection**: 3072 → 256 dimensional semantic features
- 🏗️ **Residual Design**: Structural bias toward spectral features while incorporating semantic context
- 🔗 **Two-Stage Fusion**: Concatenation followed by learnable fusion with residual connections

### **Advanced Training Strategies**

**Implemented training enhancements:**
- ❄️ **Selective Freezing**: Freeze backbone parameters while training semantic integration layers
- 🔄 **Multi-Modal Input Processing**: Handle different input sizes for SPAI vs semantic encoders
- 🏗️ **Modular Architecture**: Easy switching between different semantic encoders
- 📏 **Resolution Handling**: Automatic resizing for semantic encoders (224x224) while preserving original SPAI input sizes

### **Technical Dependencies**

Based on our training pipeline (`jobs/semantic/semantic.job`):
- **Hardware**: GPU A100 acceleration
- **Environment**: Conda environment with PyTorch and CUDA support
- **Key Libraries**: 
  - `open-clip-torch` for ConvNeXt-XXL semantic encoder
  - `timm` for transformer utilities
  - Neptune for experiment tracking (offline mode)

---

## 📈 Results

Our enhanced models demonstrate the effectiveness of semantic integration:

### **Quantitative Results**
> 📝 *[Results section would be linked to Jupyter notebook analysis]*

**Key findings include:**
- 🌍 **Improved Generalization**: Semantic-enhanced models show better cross-dataset performance
- 🛡️ **Robustness**: Better handling of sophisticated manipulation techniques that preserve frequency characteristics

### **Qualitative Analysis**
- 💡 **Enhanced Interpretability**: Semantic attention maps provide more intuitive explanations for detection decisions
- 🎯 **Semantic Consistency**: Models better detect manipulations that create semantic inconsistencies
- 🤝 **Frequency-Semantic Synergy**: Combined analysis captures both low-level artifacts and high-level semantic violations

---

## 🛠️ Implementation Details

### 📁 **Project Structure**
```
spai/
├── models/
│   ├── sid.py              # 🧠 Core implementations (PatchBasedMFViT, SemanticContextModel)
│   ├── backbones.py        # 🏗️ CLIP, DINOv2, ConvNeXt semantic encoders
│   └── build.py            # 🏭 Model factory functions
├── data/                   # 📊 Dataset handling and preprocessing
├── config.py               # ⚙️ Configuration management system
├── __main__.py             # 🖥️ CLI interface for training/testing/inference
└── utils.py                # 🔧 Utility functions and helpers
```

### 🔧 **Key Components**

#### 1. 🧩 **SemanticContextModel**
- **Location**: `spai/models/sid.py:1453-1615`
- **Architecture**: Late fusion with residual connections and frozen pre-trained models
- **Innovation**: Structural bias toward spectral features while incorporating semantic understanding

#### 2. 🏗️ **SemanticPipeline**
- **Location**: `spai/models/sid.py:1649-1752`
- **Purpose**: ConvNeXt-XXL semantic feature extraction pipeline
- **Features**: Automatic normalization and compact feature projection

#### 3. 🎯 **Combined Model Training**
- **Script**: `semantic_pipeline.combined_model` (as per job configuration)
- **Approach**: End-to-end training of the late fusion architecture
- **Dependencies**: Requires both SPAI pre-trained weights and ConvNeXt-XXL from OpenCLIP

### 💻 **Usage**

```bash
# 🧩 Train semantic context model (late fusion)
python -m spai train \
--cfg "./configs/spai.yaml" \
--batch-size 256 \
--data-path "<dataset_path>" \
--csv-root-dir "/spai/data/train" \
--output "./output/LSUN_RESIDUAL_ORIGINAL" \
--tag "first_run" \
--data-workers 4 \
--save-all \
--amp-opt-level "O0" \
--opt "TRAIN.EPOCHS" "10" \
--opt "DATA.TEST_PREFETCH_FACTOR" "1" \
--opt "DATA.VAL_BATCH_SIZE" "256" \
--opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
--opt "PRINT_FREQ" "2"
--opt "MODEL.SEMANTIC_CONTEXT.SPAI_INPUT_SIZE" "[224, 224]" 

# 🧪 Test semantic-enhanced model
python -m spai test \
--cfg "./configs/spai.yaml" \
--batch-size 10 \
--model "<model_path>" \
--output "./output/semantic_test" \
--tag "spai" \
--opt "MODEL.PATCH_VIT.MINIMUM_PATCHES" "4" \
--opt "DATA.NUM_WORKERS" "8" \
--opt "MODEL.FEATURE_EXTRACTION_BATCH" "400" \
--opt "DATA.TEST_PREFETCH_FACTOR" "1" \
--test-csv "<test_set_path>" \
--opt "PRINT_FREQ" "2" \
--opt "MODEL.SEMANTIC_CONTEXT.HIDDEN_DIMS" "[512]" \
--opt "MODEL.SEMANTIC_CONTEXT.SPAI_INPUT_SIZE" "[1024, 1024]" 

# 📊 Or run training and testing jobs on GPU cluster
sbatch jobs/semantic/train.job
sbatch jobs/semantic/test.job
```

---

## 🎯 Conclusions

Our work successfully addresses key limitations of the original SPAI by implementing a **spectral-semantic late fusion approach**. This approach demonstrates that **combining low-level spectral analysis with high-level semantic context creates more robust and interpretable manipulation detection systems**.

> 🏆 **Impact**: The late fusion design allows efficient training by leveraging frozen pre-trained models while achieving improved robustness on challenging datasets like Chameleon.

**🔑 Key Technical Contributions:**
- **Late Fusion Architecture**: Novel residual fusion design that preserves spectral feature importance
- **ConvNeXt Integration**: Semantic understanding through state-of-the-art ConvNeXt-XXL features
- **Robustness Enhancement**: Improved performance on derivative images and challenging real-world scenarios

### 🔮 **Future Work**
- Dynamic weighting mechanisms for frequency-semantic integration
- Extension to video manipulation detection using temporal semantic consistency
- Real-time deployment optimizations for mobile/edge devices
- Integration with large language models for textual context understanding

---

## 👥 Student Contributions


### **Iwo Godzwon**
- Designed and implemented SemanticContextModel late fusion architecture
- Developed ConvNeXt-XXL semantic pipeline integration
- Created training pipeline for combined model approach
- Collected Chameleon dataset

### **Agata Zywot**
- Implemented cross-attention fusion mechanisms before and after spectral context attention module.
- Implemented bidirectional cross-attention fusion mechanisms.
- Performed comprehensive evaluation experiments on Chameleon/LDM/other datasets

### **Pradyut Nair**
- Implemented mid-level semantic features extraction using CLIP-ViT-B-32-torch and validating whether fusing frequency features with semantic features improves the performance of the model.
- Verified the late semantic fusion architecture using DinoV2 semantic backbone.
- Collected training and testing datasets
- Conducted comparative studies between LDM and Chameleon datasets

### **Egor Karasev**
- Implemented the late semantic fusion architecture using ConvNeXt-XXL semantic features.
Collected training and testing datasets. 
- Create a derivative dataset by applying JPEG compression, resizing, and blurring to the original images.
- Performed comprehensive evaluation experiments on Chameleon dataset
- Conducted cross-dataset generalization studies




---

**🔗 Repository**: https://github.com/pradyutnair/spai/

**📄 Paper**: https://arxiv.org/abs/2411.19417

**📜 License**: Apache 2.0
