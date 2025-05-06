import torch
import pytest
from spai.models.semantic_fusion import SemanticFusionModule, FUSION_TYPES

def get_dummy_inputs(batch=2, n=3, l=4, d=8, semantic_dim=6):
    patch_feats = torch.randn(batch, n, l, d)
    semantic_vec = torch.randn(batch, semantic_dim)
    return patch_feats, semantic_vec

@pytest.mark.parametrize("fusion_type", FUSION_TYPES)
def test_fusion_shapes(fusion_type):
    print(fusion_type)
    patch_feats, semantic_vec = get_dummy_inputs()
    module = SemanticFusionModule(patch_dim=8, semantic_dim=6, fusion_type=fusion_type)
    out = module(patch_feats, semantic_vec)
    if fusion_type == 'concat':
        assert out.shape == (2, 3, 4, 14)
    else:
        assert out.shape == (2, 3, 4, 8)

@pytest.mark.parametrize("fusion_type", FUSION_TYPES)
def test_fusion_broadcast_and_grad(fusion_type):
    patch_feats, semantic_vec = get_dummy_inputs()
    module = SemanticFusionModule(patch_dim=8, semantic_dim=6, fusion_type=fusion_type)
    out = module(patch_feats, semantic_vec)
    assert torch.isfinite(out).all()
    loss = out.sum()
    loss.backward()
    # Check gradients exist
    for p in module.parameters():
        if p.requires_grad:
            assert p.grad is not None

def test_invalid_fusion_type():
    with pytest.raises(ValueError):
        SemanticFusionModule(8, 6, fusion_type='invalid')


if __name__ == "__main__":
    test_fusion_shapes('concat')
    test_fusion_broadcast_and_grad('concat')
    test_invalid_fusion_type()
