# Project Plan: Integrating Semantic Context Analysis with CLIP in SPAI

This document details the steps, file modifications, and training requirements for enhancing SPAI to utilize CLIP's semantic context capabilities, based on two approaches:

- **Approach 1**: Use global semantic context for the whole image and inject it into patch-level processing, allowing spectral attention to combine spectral and semantic cues.
- **Approach 2**: Use both global and local semantic context vectors to identify local semantic features in patches, even without positional embeddings.

---

## 1. Overview of Required Changes

- **Architectural Additions**:
  - Add a semantic analysis branch using CLIP's text and image encoders.
  - Design fusion mechanisms to combine spectral and semantic information.
- **Codebase Modifications**:
  - Update backbone and model classes.
  - Add new modules for semantic feature extraction and fusion.
  - Modify data pipelines to handle text prompts and semantic targets.
- **Training Pipeline**:
  - Update configs for new semantic loss terms and data requirements.
  - Add scripts for semantic-aware training and evaluation.

---

## 2. File-by-File Modification Plan

### 2.1. `backbones/CLIPBackbone.py` (or equivalent)

- **Add Methods**:
  - Expose both image and text encoders.
  - Add methods to extract global image embeddings and text embeddings.

- **Example Additions**:
  - `get_image_embedding(image_tensor)`
  - `get_text_embedding(text_prompt)`

---

### 2.2. `models/FrequencyRestorationEstimator.py` (or main model file)

- **Modifications**:
  - Add new input for global semantic context vector.
  - Modify forward pass to:
    - Accept both spectral features and semantic context.
    - Fuse global semantic context into patch-level features (Approach 1).
    - Optionally, compute local semantic context for each patch (Approach 2).
  - Add fusion modules (e.g., attention, gating, concatenation).

- **New Classes/Functions**:
  - `SemanticFusionModule`
  - `compute_global_semantic_context`
  - `compute_local_semantic_context` (optional for Approach 2)

---

### 2.3. `train.py` / `trainer.py` / `main.py`

- **Modifications**:
  - Update data loading to include text prompts or semantic targets.
  - Pass semantic context through the model.
  - Add new loss terms for semantic consistency (e.g., CLIP image-text similarity loss).
  - Update logging and checkpointing to track new metrics.

---

### 2.4. `data/` (Dataset and Dataloader)

- **Modifications**:
  - Update datasets to provide image-caption pairs or generate synthetic prompts.
  - Update dataloaders to yield both images and corresponding text prompts.

---

### 2.5. `config/` (YAML/JSON config files)

- **Modifications**:
  - Add options for enabling semantic context analysis.
  - Specify paths for CLIP models, text prompts, and semantic loss weights.

---

### 2.6. `evaluation/` or `metrics/`

- **Modifications**:
  - Add evaluation scripts for semantic consistency (e.g., CLIP score between predicted and ground-truth captions).
  - Update visualization scripts to show semantic attention maps.

---

## 3. Step-by-Step Implementation Guide

### Step 1: Update CLIP Backbone

- Expose both image and text encoders.
- Add utility functions to obtain global embeddings for images and text.

### Step 2: Modify Model Architecture

- Add input for global semantic context vector in the main model.
- For Approach 1:
  - Inject global semantic context into patch-level features before/after spectral attention.
- For Approach 2:
  - Compute local semantic context for patches (e.g., by cropping and embedding).
  - Fuse both global and local semantic context with spectral features.

### Step 3: Add Fusion Mechanisms

- Implement modules to combine spectral and semantic features (e.g., attention, gating).
- Integrate fusion modules into the model's forward pass.

### Step 4: Update Data Pipeline

- Modify datasets to provide text prompts (captions, labels, or synthetic prompts).
- Update dataloader to yield both images and text.

### Step 5: Update Training Script

- Add new loss terms for semantic consistency (e.g., maximize CLIP similarity between image and text).
- Update optimizer to handle new parameters.
- Add logging for semantic metrics.

### Step 6: Update Configs

- Add flags and parameters for semantic analysis (e.g., `USE_SEMANTIC_CONTEXT`, `SEMANTIC_LOSS_WEIGHT`).
- Specify CLIP model paths and text prompt sources.

### Step 7: Update Evaluation

- Add scripts to compute semantic consistency metrics (e.g., CLIP score).
- Visualize semantic attention/fusion.

---

## 4. Training & Evaluation

- **Data**: Use datasets with image-caption pairs or generate synthetic prompts.
- **Training**:
  - Train with both spectral and semantic losses.
  - Tune fusion module parameters and loss weights.
- **Evaluation**:
  - Evaluate on both spectral and semantic consistency metrics.
  - Visualize fusion outputs and attention maps.

---

## 5. Directory & Config Path Examples

- **CLIP Model Weights**: `pretrained/clip/ViT-B-32.pt`
- **Config Flag**: `config.MODEL.USE_SEMANTIC_CONTEXT = True`
- **Semantic Loss Weight**: `config.LOSS.SEMANTIC_WEIGHT = 0.5`
- **Text Prompts Path**: `data/prompts/imagenet_classnames.txt`

---

## 6. Summary Table

| File/Module                        | Action/Addition                                              |
|-------------------------------------|--------------------------------------------------------------|
| `backbones/CLIPBackbone.py`         | Expose image/text encoders, add embedding methods            |
| `models/FrequencyRestorationEstimator.py` | Add semantic context input, fusion modules, forward pass changes |
| `train.py` / `trainer.py`           | Update data, add semantic loss, update logging               |
| `data/`                             | Provide image-text pairs, update dataloader                  |
| `config/`                           | Add semantic context flags/paths                             |
| `evaluation/`                       | Add semantic metrics, visualization                          |

---

## 7. Example Pseudocode (Approach 1)

```python
# In model forward
image_features = self.spectral_encoder(image)
global_semantic = self.clip_backbone.get_image_embedding(image)
fused_features = self.semantic_fusion(image_features, global_semantic)
output = self.head(fused_features)