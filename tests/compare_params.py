import torch
import yaml
from pathlib import Path
from spai.utils import remap_pretrained_keys_vit, remap_pretrained_keys_swin
from spai.models import build_cls_model
import sys
sys.path.append("/home/pnair/spai")
# ─── Configuration ─────────────────────────────────────────────────────────
checkpoint_path   = Path("/home/pnair/spai/weights/mfm_pretrain_vit_base.pth")
model_type        = "vit"      # or "swin"
pretrained_prefix = "model"    # or "" if your checkpoint keys are top-level
config_file       = "/home/pnair/spai/configs/spai.yaml"

# ─── 1) Load the checkpoint ────────────────────────────────────────────────
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
checkpoint_model = ckpt.get("model", ckpt)
if pretrained_prefix:
    pref = pretrained_prefix + "."
    checkpoint_model = {
        k[len(pref):]: v
        for k, v in checkpoint_model.items()
        if k.startswith(pref)
    }

# ─── 2) Build your model from the YAML config ─────────────────────────────
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)
    # Convert dict to object with attributes
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)
    cfg = Config(cfg)

model = build_cls_model(cfg)

# ─── 3) Get model state dict ──────────────────────────────────────────────
model_state = model.state_dict()

# ─── 4) Compare keys ───────────────────────────────────────────────────────
model_keys = set(model_state.keys())
ckpt_keys  = set(checkpoint_model.keys())

missing    = sorted(model_keys  - ckpt_keys)
unexpected = sorted(ckpt_keys   - model_keys)

print(f"\nMissing ({len(missing)}) keys (in model but not in ckpt):")
for k in missing:
    print("  ", k)

print(f"\nUnexpected ({len(unexpected)}) keys (in ckpt but not in model):")
for k in unexpected:
    print("  ", k)

# ─── 5) Create filtered state dict ────────────────────────────────────────
filtered_state_dict = {}
for k, v in checkpoint_model.items():
    if k in model_state:
        filtered_state_dict[k] = v

# ─── 6) Load filtered state dict ──────────────────────────────────────────
res = model.load_state_dict(filtered_state_dict, strict=False)
print("\nload_state_dict result:")
print(res)

# ─── 7) Print shape mismatches ────────────────────────────────────────────
print("\nShape mismatches:")
for k in model_keys:
    if k in checkpoint_model:
        if model_state[k].shape != checkpoint_model[k].shape:
            print(f"  {k}:")
            print(f"    Model shape: {model_state[k].shape}")
            print(f"    Checkpoint shape: {checkpoint_model[k].shape}")
