
import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import pydantic
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from models.trm_jax import TinyRecursiveReasoningModel_ACTV1, TinyRecursiveReasoningModel_ACTV1Config, TinyRecursiveReasoningModel_ACTV1Carry

# --- Config ---
# Reusing Pydantic models from pretrain.py where possible, or redefining for simplicity

class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str

class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig

class PretrainConfig(pydantic.BaseModel):
    arch: ArchConfig
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int
    epochs: int
    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None
    seed: int = 0
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0
    
    halt_max_steps: int = 12 # Default from typical TRM
    
    # JAX specific
    jax_seed: int = 42

# --- Data Loading ---

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)

def create_dataloader(config: PretrainConfig, split: str, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=0,
        num_replicas=1,
        test_set_mode=split=="test",
        **kwargs
    ), split=split)
    
    # Use custom collate to return numpy arrays
    dataloader = DataLoader(
        dataset,
        batch_size=None, # Dataset yields batches
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False,
        collate_fn=numpy_collate
    )
    return dataloader, dataset.metadata

# --- Training Step ---

@nnx.jit
def train_step(model: TinyRecursiveReasoningModel_ACTV1, optimizer: nnx.Optimizer, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, jax.Array], rngs: nnx.Rngs):
    
    def loss_fn(model, carry, batch, rngs):
        # Forward
        new_carry, outputs = model(carry, batch, rngs=rngs)
        
        logits = outputs["logits"] # [B, L, V]
        targets = batch["labels"] # [B, L]
        batch_size = logits.shape[0]
        
        # Masks
        mask = targets != -100
        loss_counts = jnp.sum(mask, axis=-1)
        loss_divisor = jnp.maximum(loss_counts, 1.0)[:, None]
        
        # LM Loss
        loss_per_token = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.maximum(targets, 0))
        loss_per_token = jnp.where(mask, loss_per_token, 0.0)
        lm_loss = jnp.sum(loss_per_token) / jnp.maximum(jnp.sum(mask), 1.0) # Mean over all valid tokens in batch
        
        # Metrics Calculation matches ACTLossHead
        preds = jnp.argmax(logits, axis=-1)
        is_correct = mask & (preds == targets)
        seq_is_correct = jnp.sum(is_correct, axis=-1) == loss_counts
        
        # Q Losses
        # q_halt_logits: [B], q_continue_logits: [B]
        # target: seq_is_correct (float)
        q_halt_logits = outputs["q_halt_logits"]
        q_continue_logits = outputs["q_continue_logits"]
        
        q_halt_loss = optax.sigmoid_binary_cross_entropy(q_halt_logits, seq_is_correct.astype(jnp.float32))
        q_halt_loss = jnp.sum(q_halt_loss) # Sum reduction as in PyTorch
        
        q_continue_loss = 0.0
        if "target_q_continue" in outputs:
            q_continue_loss = optax.sigmoid_binary_cross_entropy(q_continue_logits, outputs["target_q_continue"])
            q_continue_loss = jnp.sum(q_continue_loss)

        # PyTorch Loss: lm_loss (sum / divisor) + 0.5 * (q_halt + q_cont)
        # Note: PyTorch lm_loss div is per-sequence counts, then summed.
        # My lm_loss above is global mean. 
        # PyTorch: (loss / loss_divisor).sum() -> Sum of mean-sequence-losses.
        # Let's match PyTorch:
        lm_loss_sum = jnp.sum(jnp.sum(loss_per_token, axis=-1) / loss_divisor.squeeze(-1))
        
        total_loss = lm_loss_sum + 0.5 * (q_halt_loss + q_continue_loss)
        
        # Metrics dict
        # valid_metrics = new_carry.halted & (loss_counts > 0)
        valid_metrics = new_carry.halted & (loss_counts > 0)
        valid_count = jnp.sum(valid_metrics)
        
        # Acuracy: (is_correct / loss_divisor).sum(-1) masked by valid
        seq_acc = jnp.sum(is_correct, axis=-1) / loss_divisor.squeeze(-1)
        accuracy = jnp.sum(jnp.where(valid_metrics, seq_acc, 0.0))
        
        exact_accuracy = jnp.sum(valid_metrics & seq_is_correct)
        
        q_halt_accuracy = jnp.sum(valid_metrics & ((q_halt_logits >= 0) == seq_is_correct))
        
        steps = jnp.sum(jnp.where(valid_metrics, new_carry.steps, 0))

        metrics = {
            "loss": total_loss,
            "lm_loss": lm_loss_sum,
            "q_halt_loss": q_halt_loss,
            "q_continue_loss": q_continue_loss,
            "count": valid_count,
            "accuracy": accuracy,
            "exact_accuracy": exact_accuracy,
            "q_halt_accuracy": q_halt_accuracy,
            "steps": steps
        }
        
        # PyTorch minimizes (1/B) * SumLoss
        return total_loss / batch_size, (new_carry, metrics)

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_carry, metrics)), grads = grad_fn(model, carry, batch, rngs)
    
    optimizer.update(grads)
    
    return loss, new_carry, metrics

# --- Evaluation ---

@nnx.jit
def eval_step(model: TinyRecursiveReasoningModel_ACTV1, batch: Dict[str, jax.Array]):
    # Init carry for this batch
    # We can't use model.initial_carry inside jit if it uses python control flow?
    # initial_carry uses jnp.zeros, so it should be fine.
    carry = model.initial_carry(batch)
    
    # Forward
    # We don't update carry across batches in eval usually, unless it's stateful RNN style.
    # TRM resets carry for new puzzles.
    # So we just run one pass?
    # Wait, TRM is an ACT model. It runs until halt.
    # The `model.__call__` does ONE step of ACT.
    # We need to run until all halt?
    # Or just run N steps?
    # In training we run 1 step per optimizer step (Online ACT).
    # In eval, we usually want to solve the puzzle.
    # The PyTorch `evaluate` loop runs:
    # while True: model(..., return_keys=...)
    
    # Implementing full inference loop inside JIT might be complex due to dynamic shapes/halting.
    # But `scan` or `while_loop` can handle it.
    
    # For now, let's just compute the LOSS on the first step (Teacher Forcing / Training objective).
    # This gives us a validation loss comparable to training loss.
    # Full generation/solving accuracy is a separate metric.
    
    new_carry, outputs = model(carry, batch)
    
    logits = outputs["logits"]
    targets = batch["labels"]
    mask = targets != -100
    
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, jnp.maximum(targets, 0))
    loss = jnp.where(mask, loss, 0.0)
    loss = jnp.sum(loss) / jnp.maximum(jnp.sum(mask), 1.0)
    
    # Accuracy
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets) & mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)
    
    return loss, accuracy

def evaluate(model, dataloader, step):
    print(f"Running evaluation at step {step}...")
    
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    
    # We need to iterate the dataloader.
    # Since it's an IterableDataset, we just iterate.
    # We should limit the number of batches if it's infinite, 
    # but "test" split usually has fixed size?
    # PuzzleDataset is infinite by default unless configured otherwise?
    # In PyTorch `create_dataloader` for test: `epochs_per_iter=1`.
    # It should yield the whole dataset once.
    
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        # Collate is already done by dataloader
        # Convert to JAX
        batch = {k: jnp.array(v) for k, v in batch.items()}
        
        loss, acc = eval_step(model, batch)
        
        total_loss += loss.item()
        total_acc += acc.item()
        count += 1
        
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_acc = total_acc / count if count > 0 else 0.0
    
    print(f"Eval Step {step}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
    
    if wandb.run is not None:
        wandb.log({"eval/loss": avg_loss, "eval/accuracy": avg_acc, "step": step})
        
    return avg_loss

# --- Main ---

@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    # Convert Hydra config to Pydantic
    # We need to handle nested configs manually or just use the dict
    # For simplicity, let's extract what we need.
    
    # Hack: Create a dummy config object to parse
    # config = PretrainConfig(**hydra_config) # This might fail due to nesting
    
    print("Starting JAX Pretraining...")
    
    # Manual Config Extraction (Simplified)
    class ConfigWrapper:
        def __init__(self, cfg):
            self.cfg = cfg
        def __getattr__(self, name):
            return self.cfg.get(name)
            
    # We assume the structure matches.
    # Let's try to instantiate the model config.
    
    arch_cfg = hydra_config.arch
    
    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=hydra_config.global_batch_size,
        seq_len=256, # Placeholder, will update from metadata
        num_puzzle_identifiers=10, # Placeholder
        vocab_size=100, # Placeholder
        H_cycles=arch_cfg.H_cycles,
        L_cycles=arch_cfg.L_cycles,
        H_layers=arch_cfg.H_layers,
        L_layers=arch_cfg.L_layers,
        hidden_size=arch_cfg.hidden_size,
        expansion=arch_cfg.expansion,
        num_heads=arch_cfg.num_heads,
        pos_encodings=arch_cfg.pos_encodings,
        halt_max_steps=arch_cfg.halt_max_steps,
        halt_exploration_prob=arch_cfg.halt_exploration_prob,
        puzzle_emb_ndim=arch_cfg.puzzle_emb_ndim
    )
    
    # Data
    train_loader, train_metadata = create_dataloader(
        PretrainConfig(**hydra_config), "train", 
        epochs_per_iter=1, 
        global_batch_size=hydra_config.global_batch_size
    )
    
    # Test Data
    test_loader, _ = create_dataloader(
        PretrainConfig(**hydra_config), "test",
        epochs_per_iter=1,
        global_batch_size=hydra_config.global_batch_size
    )
    
    # Update config with metadata
    model_config.vocab_size = train_metadata.vocab_size
    model_config.seq_len = train_metadata.seq_len
    model_config.num_puzzle_identifiers = train_metadata.num_puzzle_identifiers
    
    print(f"Model Config: {model_config}")
    
    # Init Model
    rngs = nnx.Rngs(hydra_config.seed)
    model = TinyRecursiveReasoningModel_ACTV1(model_config, rngs=rngs)
    
    # Init WandB
    if hydra_config.get("project_name"):
        wandb_config = OmegaConf.to_container(hydra_config, resolve=True, throw_on_missing=True)
        wandb.init(project=hydra_config.project_name, name=hydra_config.get("run_name"), config=wandb_config)
    
    # Differential Learning Rates & Weight Decay
    # We partition parameters into 'puzzle_emb' and 'common'.
    params = nnx.state(model, nnx.Param)
    
    def map_param_to_group(path, param):
        # path is a tuple of strings/ints
        # Check if 'puzzle_emb' is in the path
        if "puzzle_emb" in map(str, path):
             return "puzzle_emb"
        return "common"
        
    param_labels = jax.tree_util.tree_map_with_path(map_param_to_group, params)
    
    # Schedulers
    # Total Steps
    # Schedulers
    # Total Steps
    total_steps = hydra_config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples // hydra_config.global_batch_size
    total_steps = int(total_steps)

    print(f"DEBUG: Total Steps Calculation:")
    print(f"  Epochs: {hydra_config.epochs}")
    print(f"  Total Groups: {train_metadata.total_groups}")
    print(f"  Mean Puzzle Examples: {train_metadata.mean_puzzle_examples}")
    print(f"  Global Batch Size: {hydra_config.global_batch_size}")
    print(f"  Calculated Total Steps: {total_steps}")

    config_seed = hydra_config.seed

    # Main scheduler
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hydra_config.lr,
        warmup_steps=hydra_config.lr_warmup_steps,
        decay_steps=total_steps,
        end_value=hydra_config.lr * hydra_config.lr_min_ratio
    )
    
    # Puzzle Emb Scheduler
    # Assuming same schedule shape but potentially different peak
    puzzle_emb_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hydra_config.puzzle_emb_lr,
        warmup_steps=hydra_config.lr_warmup_steps, # Assuming same warmup
        decay_steps=total_steps,
        end_value=hydra_config.puzzle_emb_lr * hydra_config.lr_min_ratio
    )

    # Optimizer Chain
    # 1. Clip global norm (applies to gradients of all params)
    # 2. Multi-transform for optimizer updates
    
    optimizer_def = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {
                "common": optax.adamw(
                    learning_rate=scheduler,
                    weight_decay=hydra_config.weight_decay,
                    b1=hydra_config.beta1,
                    b2=hydra_config.beta2
                ),
                "puzzle_emb": optax.adamw(
                    learning_rate=puzzle_emb_scheduler,
                    weight_decay=hydra_config.puzzle_emb_weight_decay,
                    b1=hydra_config.beta1,
                    b2=hydra_config.beta2
                ),
            },
            param_labels
        )
    )

    optimizer = nnx.Optimizer(model, optimizer_def)
    
    # EMA
    ema = None
    if hydra_config.ema:
        # We use Optax's EMA transformation, but we need to apply it separately or wrap the optimizer.
        # Since nnx.Optimizer wraps the update, we can't easily inject it into the chain if we want the *params* to be EMA'd for eval.
        # Standard practice: keep a separate set of EMA params.
        # Or use optax.ema() which returns a gradient transformation.
        # But we want the WEIGHTS to be averaged, not the gradients.
        # Optax has `optax.ema` which is a gradient transform (momentum).
        # We want Polyak averaging.
        # Flax has no built-in helper for this in NNX yet?
        # Let's implement a simple manual EMA update.
        
        ema_params = nnx.state(model, nnx.Param)
        
    def update_ema(ema_params, current_params, rate):
        return jax.tree_util.tree_map(
            lambda e, p: rate * e + (1.0 - rate) * p,
            ema_params,
            current_params
        )
    
    # Initial Carry
    # We need a dummy batch to init carry?
    # Or just use shapes.
    # The `initial_carry` method needs a batch to know keys and shapes.
    # The dataloader yields (set_name, batch, size)
    _, dummy_batch, _ = next(iter(train_loader))
    # Convert to JAX array
    dummy_batch = {k: jnp.array(v) for k, v in dummy_batch.items()}
    
    carry = model.initial_carry(dummy_batch)
    
    # Training Loop
    step = 0

    
    pbar = tqdm.tqdm(total=total_steps)
    
    # We need an infinite iterator for the dataloader since we step continuously
    def infinite_dataloader(loader):
        while True:
            for batch in loader:
                yield batch
                
    data_iter = infinite_dataloader(train_loader)
    
    # Checkpointer
    checkpointer = ocp.StandardCheckpointer()
    checkpoint_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=hydra_config.eval_interval)
    checkpoint_manager = ocp.CheckpointManager(
        os.path.abspath(hydra_config.get("checkpoint_path")) if hydra_config.get("checkpoint_path") else "checkpoints",
        checkpointer,
        checkpoint_options
    )
    
    # Restore if exists
    if checkpoint_manager.latest_step() is not None:
        print(f"Restoring from step {checkpoint_manager.latest_step()}")
        # We need to restore model params and optimizer state
        # NNX makes this a bit tricky as we need to map back to the object
        # For now, let's assume we save the graph state.
        # Actually, NNX has state_dict support.
        
        abstract_state = nnx.state(model, optimizer)
        restored_state = checkpoint_manager.restore(checkpoint_manager.latest_step(), args=ocp.args.StandardRestore(abstract_state))
        nnx.update(model, optimizer, restored_state)
        step = checkpoint_manager.latest_step()
        pbar.update(step)

    # Sharding Utils
    mesh = jax.sharding.Mesh(jax.devices(), ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    
    def shard_batch(batch):
        return jax.tree_util.tree_map(
            lambda x: jax.device_put(x, sharding),
            batch
        )

    for _ in range(step, total_steps):
        _, batch, _ = next(data_iter)
        batch = {k: jnp.array(v) for k, v in batch.items()}
        batch = shard_batch(batch)
        
        # Per-step RNG
        step_rng = nnx.Rngs(config_seed + step)
        
        loss, carry, metrics = train_step(model, optimizer, carry, batch, rngs=step_rng)
        
        if ema_params is not None:
             current_params = nnx.state(model, nnx.Param)
             ema_params = update_ema(ema_params, current_params, hydra_config.ema_rate)
        
        if step % 10 == 0:  
            # Normalize metrics for logging
            # metrics contains Sums.
            # PyTorch logic:
            # loss-like -> divide by global_batch_size
            # count-dependent -> divide by count
            
            count = metrics["count"]
            count_safe = jnp.maximum(count, 1.0)
            
            # Helper to convert to python scalar for logging
            def to_scalar(x): return x.item() if hasattr(x, "item") else float(x)
            
            log_dict = {
                "loss": to_scalar(loss), # Already normalized in loss_fn return
                "step": step,
                "lr": to_scalar(scheduler(step))
            }
            
            for k, v in metrics.items():
                v_scalar = to_scalar(v)
                if k.endswith("loss"):
                    # Losses are sums in metrics dict
                    log_dict[f"train/{k}"] = v_scalar / hydra_config.global_batch_size
                elif k == "count":
                    log_dict[f"train/{k}"] = v_scalar
                else:
                    # Accuracy, steps, etc are sums over 'count' items
                    log_dict[f"train/{k}"] = v_scalar / to_scalar(count_safe)
            
            pbar.set_description(f"Loss: {log_dict['loss']:.4f} | Acc: {log_dict['train/accuracy']:.4f} | Cnt: {metrics['count']}")
            if wandb.run is not None:
                wandb.log(log_dict)
            
        # Checkpointing
        if step % hydra_config.eval_interval == 0:
             # Eval
             if ema_params is not None:
                 print("Evaluating with EMA parameters...")
                 eval_model = nnx.merge(nnx.split(model)[0], ema_params)
                 evaluate(eval_model, test_loader, step)
             else:
                 evaluate(model, test_loader, step)

             # Checkpoint
             if hydra_config.get("checkpoint_path"):
                 save_args = ocp.args.StandardSave(nnx.state(model, optimizer))
                 checkpoint_manager.save(step, args=save_args)
             
        pbar.update(1)
        step += 1
        
    print("Training finished.")
    if hydra_config.get("checkpoint_path"):
         save_args = ocp.args.StandardSave(nnx.state(model, optimizer))
         checkpoint_manager.save(step, args=save_args)
         checkpoint_manager.wait_until_finished()
         
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    launch()
