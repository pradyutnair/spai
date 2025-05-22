import torch
import torch.nn as nn

def simulate_load_state_dict_mismatch():
    """
    Simulate the patch_aggregator parameter size mismatch scenario and test our fix.
    """
    print("Testing patch_aggregator parameter size mismatch fix...")
    
    # Create a simple model with our adjusted hook
    class TestModel(nn.Module):
        def __init__(self, num_heads=8, dim_head=192):
            super().__init__()
            self.num_heads = num_heads
            self.dim_head = dim_head
            self.patch_aggregator = nn.Parameter(torch.zeros((num_heads, 1, dim_head)))
            nn.init.trunc_normal_(self.patch_aggregator, std=.02)
            
            # Register our hook to handle mismatches
            self._register_load_state_dict_pre_hook(self._adjust_patch_aggregator_hook)
            
        def _adjust_patch_aggregator_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            """Pre-hook for load_state_dict to handle patch_aggregator shape mismatch"""
            # Check if patch_aggregator is in state_dict and has different shape
            patch_aggregator_key = prefix + 'patch_aggregator'
            if patch_aggregator_key in state_dict:
                checkpoint_aggregator = state_dict[patch_aggregator_key]
                model_aggregator = self.patch_aggregator
                
                # If shapes don't match, resize the parameter
                if checkpoint_aggregator.shape != model_aggregator.shape:
                    checkpoint_num_heads, _, checkpoint_dim_head = checkpoint_aggregator.shape
                    model_num_heads, _, model_dim_head = model_aggregator.shape
                    
                    print(f"⚠️ Adjusting patch_aggregator shape from checkpoint {checkpoint_aggregator.shape} to model {model_aggregator.shape}")
                    
                    # Create a new tensor with the model's shape
                    new_aggregator = torch.zeros_like(model_aggregator)
                    
                    # Determine how many heads and dimensions to copy
                    heads_to_copy = min(checkpoint_num_heads, model_num_heads)
                    dim_to_copy = min(checkpoint_dim_head, model_dim_head)
                    
                    # Copy the values from the checkpoint to the new tensor
                    new_aggregator[:heads_to_copy, :, :dim_to_copy] = checkpoint_aggregator[:heads_to_copy, :, :dim_to_copy]
                    
                    # Replace the checkpoint's tensor with the resized one
                    state_dict[patch_aggregator_key] = new_aggregator
                    
                    print(f"✅ Successfully resized patch_aggregator")
    
    # Create our model with 8 heads and 192 dim_head
    model = TestModel(num_heads=8, dim_head=192)
    
    # Create a fake state dict with mismatched dimensions (12 heads, 128 dim_head)
    # This simulates the checkpoint's patch_aggregator shape
    checkpoint_state_dict = {
        'patch_aggregator': torch.randn(12, 1, 128)  # Different shape than our model
    }
    
    # Try to load the mismatched state dict
    try:
        model.load_state_dict(checkpoint_state_dict, strict=False)
        
        # Verify the shape after loading
        print(f"Model patch_aggregator shape after loading: {model.patch_aggregator.shape}")
        
        if model.patch_aggregator.shape == (8, 1, 192):
            print("✅ Test passed! The parameter has the correct shape after loading.")
            return True
        else:
            print(f"❌ Test failed! Expected shape (8, 1, 192) but got {model.patch_aggregator.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    success = simulate_load_state_dict_mismatch()
    if success:
        print("🎉 The fix for patch_aggregator parameter size mismatch is working correctly!")
    else:
        print("❌ The fix is not working correctly.") 