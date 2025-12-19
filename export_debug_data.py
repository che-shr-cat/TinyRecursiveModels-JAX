
import torch
import numpy as np
import os
import hydra
from omegaconf import DictConfig
from pretrain import create_model, init_train_state, PretrainConfig, PuzzleDatasetMetadata

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def export(hydra_config: DictConfig):
    print("Exporting debug data...")
    
    # Config
    config = PretrainConfig(**hydra_config)
    
    # Dummy metadata
    train_metadata = PuzzleDatasetMetadata(
        seq_len=256,
        vocab_size=100,
        pad_id=0,
        ignore_label_id=-100,
        blank_identifier_id=0,
        num_puzzle_identifiers=10,
        total_groups=1,
        mean_puzzle_examples=1,
        total_puzzles=1,
        sets=["train"]
    )
    
    # Create Model
    model, _, _ = create_model(config, train_metadata, rank=0, world_size=1)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_len = 256
    inputs = torch.randint(0, 100, (batch_size, seq_len)).cuda()
    puzzle_identifiers = torch.zeros((batch_size,), dtype=torch.long).cuda()
    
    batch = {
        "inputs": inputs,
        "puzzle_identifiers": puzzle_identifiers
    }
    
    # Forward
    with torch.no_grad():
        carry = model.initial_carry(batch)
        # Run one step
        new_carry, loss, metrics, preds, all_finish = model(carry=carry, batch=batch, return_keys=[])
        
    # Save
    os.makedirs("debug_data", exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), "debug_data/torch_weights.pth")
    
    # Save inputs
    np.save("debug_data/inputs.npy", inputs.cpu().numpy())
    np.save("debug_data/puzzle_identifiers.npy", puzzle_identifiers.cpu().numpy())
    
    # Save outputs (from carry)
    # Inner carry has z_H, z_L
    np.save("debug_data/z_H.npy", new_carry.inner_carry.z_H.cpu().numpy())
    np.save("debug_data/z_L.npy", new_carry.inner_carry.z_L.cpu().numpy())
    
    print("Export done.")

if __name__ == "__main__":
    export()
