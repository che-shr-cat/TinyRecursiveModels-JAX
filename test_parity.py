
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch
import os
from models.trm_jax import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config
from pretrain_jax import PretrainConfig

def load_torch_weights(model: TinyRecursiveReasoningModel_ACTV1, torch_path: str):
    state_dict = torch.load(torch_path, map_location="cpu")
    
    # Helper to set weight
    def set_param(param, value):
        param.value = jnp.array(value.numpy())

    # Map weights
    # This requires careful mapping of names.
    # Example:
    # PyTorch: model.inner.embed_tokens.embedding.weight
    # JAX: model.inner.embed_tokens.embedding.embedding.value
    
    # Embeddings
    set_param(model.inner.embed_tokens.embedding.embedding, state_dict["model.inner.embed_tokens.embedding.weight"])
    
    # ... (Need to implement full mapping)
    # This is tedious and requires inspecting state_dict keys.
    # For now, just a placeholder.
    print("Loading weights (placeholder)...")

def test_parity():
    print("Testing parity...")
    
    if not os.path.exists("debug_data/torch_weights.pth"):
        print("Debug data not found. Run export_debug_data.py first.")
        return

    # Load inputs
    inputs = jnp.array(np.load("debug_data/inputs.npy"))
    puzzle_identifiers = jnp.array(np.load("debug_data/puzzle_identifiers.npy"))
    
    # Config (must match export)
    config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=256,
        num_puzzle_identifiers=10,
        vocab_size=100,
        H_cycles=3,
        L_cycles=6,
        H_layers=0,
        L_layers=2,
        hidden_size=512,
        expansion=4,
        num_heads=8,
        pos_encodings="rope",
        halt_max_steps=16,
        halt_exploration_prob=0.0,
        puzzle_emb_ndim=512
    )
    
    rngs = nnx.Rngs(0)
    model = TinyRecursiveReasoningModel_ACTV1(config, rngs=rngs)
    
    # Load weights
    load_torch_weights(model, "debug_data/torch_weights.pth")
    
    # Init carry
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers,
        "labels": jnp.zeros_like(inputs) # Dummy
    }
    carry = model.initial_carry(batch)
    
    # Forward
    new_carry, outputs = model(carry, batch)
    
    # Compare
    torch_z_H = np.load("debug_data/z_H.npy")
    jax_z_H = new_carry.inner_carry.z_H
    
    diff = np.abs(torch_z_H - jax_z_H)
    print(f"Max diff z_H: {np.max(diff)}")
    
    if np.max(diff) < 1e-4:
        print("Parity Check PASSED")
    else:
        print("Parity Check FAILED")

if __name__ == "__main__":
    test_parity()
