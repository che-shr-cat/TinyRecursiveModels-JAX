
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from pydantic import BaseModel
import math

from models.layers_jax import RMSNorm, SwiGLU, Attention, RotaryEmbedding, CastedEmbedding, CastedLinear

# --- Config ---

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

from flax import struct

# --- State Dataclasses ---

@struct.dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: jax.Array
    z_L: jax.Array

@struct.dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: jax.Array
    halted: jax.Array
    
    current_data: Dict[str, jax.Array]

# --- Model Components ---

class TinyRecursiveReasoningModel_ACTV1Block(nnx.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, rngs: nnx.Rngs):
        self.config = config
        
        if self.config.mlp_t:
             # Logic for MLP-T: Use a SwiGLU instead of Attention
             # hidden_size calculation might differ? Check PyTorch
             # PyTorch: mlp_t = SwiGLU(hidden_size=seq_len + puzzle_emb_len, expansion=expansion)
             # Wait, in PyTorch it operates on the SEQUENCE dimension?
             # "hidden_size=self.config.seq_len + self.puzzle_emb_len"
             # Yes, it mixes across the sequence length.
             
             # We need L?
             self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
             self.seq_len_total = self.config.seq_len + self.puzzle_emb_len
             
             self.mlp_t = SwiGLU(
                 hidden_size=self.seq_len_total,
                 expansion=config.expansion,
                 rngs=rngs,
                 name="mlp_t"
             )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False,
                rngs=rngs
            )
            
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
            rngs=rngs,
            name="mlp"
        )
        self.norm_eps = config.rms_norm_eps

    @nnx.remat
    def __call__(self, hidden_states: jax.Array, cos_sin: Optional[tuple[jax.Array, jax.Array]] = None) -> jax.Array:
        if self.config.mlp_t:
            # MLP-T operates on Transposed [B, D, L] effectively?
            # PyTorch:
            # hidden_states = hidden_states.transpose(1,2) # [B, D, L]
            # out = self.mlp_t(hidden_states)
            # hidden_states = rms_norm(hidden_states + out)
            # hidden_states = hidden_states.transpose(1,2) # [B, L, D]
            
            # JAX SwiGLU normally expects [..., Features].
            # Here "Features" is L (seq_len + puzzle_len).
            # So we move L to the end.
            # Input: [B, L, D]
            
            x_T = hidden_states.transpose(0, 2, 1) # [B, D, L]
            
            # SwiGLU acts on last dim (L)
            out = self.mlp_t(x_T)
            
            # Residual + Norm
            x_T = self._rms_norm(x_T + out)
            
            # Transpose back
            hidden_states = x_T.transpose(0, 2, 1) # [B, L, D]
            
        else:
            # Self Attention
            attn_out = self.self_attn(hidden_states, cos_sin=cos_sin)
            hidden_states = self._rms_norm(hidden_states + attn_out)
        
        # MLP (Token-wise)
        # [B, L, D]
        mlp_out = self.mlp(hidden_states)
        hidden_states = self._rms_norm(hidden_states + mlp_out)
        
        return hidden_states

    def _rms_norm(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.norm_eps)


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nnx.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, rngs: nnx.Rngs):
        self.layers = [
            TinyRecursiveReasoningModel_ACTV1Block(config, rngs=rngs) 
            for _ in range(config.L_layers)
        ]

    def __call__(self, hidden_states: jax.Array, input_injection: jax.Array, cos_sin: Optional[tuple[jax.Array, jax.Array]] = None) -> jax.Array:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos_sin=cos_sin)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nnx.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, rngs: nnx.Rngs):
        self.config = config
        self.forward_dtype = jnp.bfloat16 if config.forward_dtype == "bfloat16" else jnp.float32
        
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype, rngs=rngs)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False, cast_to=jnp.float32, rngs=rngs) # Logits usually float32
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True, cast_to=jnp.float32, rngs=rngs)
        
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        
        if self.config.puzzle_emb_ndim > 0:
             # TODO: Implement SparseEmbedding if needed, for now assuming standard embedding or skipping
             # The PyTorch code uses CastedSparseEmbedding. For JAX, standard Embed might be fine for small vocab, 
             # but "sparse" implies high cardinality.
             # For now, using standard Embed for simplicity, can optimize later.
             self.puzzle_emb = nnx.Embed(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim, rngs=rngs)
             # Init to 0
             self.puzzle_emb.embedding.value = jnp.zeros_like(self.puzzle_emb.embedding.value)
        
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads, 
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype, rngs=rngs)
            
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(config, rngs=rngs)
        
        # Initial states
        # In PyTorch these are Buffers, meaning they are state but not parameters (or fixed parameters).
        # We can treat them as Params that we don't optimize, or just constants.
        # PyTorch uses trunc_normal_init_ with std=1.
        self.H_init = nnx.Param(jax.random.truncated_normal(rngs.params(), -2, 2, (self.config.hidden_size,), dtype=self.forward_dtype))
        self.L_init = nnx.Param(jax.random.truncated_normal(rngs.params(), -2, 2, (self.config.hidden_size,), dtype=self.forward_dtype))
        
        # Q head special init
        self.q_head.linear.kernel.value = jnp.zeros_like(self.q_head.linear.kernel.value)
        self.q_head.linear.bias.value = jnp.full_like(self.q_head.linear.bias.value, -5.0)

    def _input_embeddings(self, input_ids: jax.Array, puzzle_identifiers: jax.Array):
        embedding = self.embed_tokens(input_ids.astype(jnp.int32))
        
        if self.config.puzzle_emb_ndim > 0:
            print(f"puzzle_identifiers shape: {puzzle_identifiers.shape}")
            if puzzle_identifiers.ndim == 2 and puzzle_identifiers.shape[-1] == 1:
                puzzle_identifiers = puzzle_identifiers.squeeze(-1)
                
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers).astype(self.forward_dtype)
            print(f"puzzle_embedding shape: {puzzle_embedding.shape}")
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                # Ensure 2D [B, D]
                if puzzle_embedding.ndim == 3 and puzzle_embedding.shape[1] == 1:
                     puzzle_embedding = puzzle_embedding.squeeze(1)
                
                print(f"puzzle_embedding shape before pad: {puzzle_embedding.shape}")
                puzzle_embedding = jnp.pad(puzzle_embedding, ((0, 0), (0, pad_count)))
                
            embedding = jnp.concatenate((puzzle_embedding.reshape(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), axis=-2)
            
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding(jnp.arange(embedding.shape[1])).astype(self.forward_dtype))
            
        return self.embed_scale * embedding

    def __call__(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, jax.Array]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, jax.Array, Tuple[jax.Array, jax.Array]]:
        
        cos_sin = self.rotary_emb(self.config.seq_len + self.puzzle_emb_len) if hasattr(self, "rotary_emb") else None
        
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        
        z_H, z_L = carry.z_H, carry.z_L
        
        # Recurrence Loop
        # PyTorch code has H_cycles-1 without grad, then 1 with grad.
        # JAX: We can use stop_gradient for the first H_cycles-1 steps.
        
        # Total H steps = H_cycles
        # For the first H_cycles - 1 steps, we detach z_H and z_L between steps?
        # PyTorch:
        # with torch.no_grad():
        #     for _H_step in range(self.config.H_cycles-1):
        #         for _L_step in range(self.config.L_cycles):
        #             z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        #         z_H = self.L_level(z_H, z_L, **seq_info)
        # for _L_step in range(self.config.L_cycles): ...
        
        # This means gradients only flow through the LAST H-cycle.
        # This is "Truncated BPTT" effectively.
        
        def body_fn(z_H, z_L):
             for _L_step in range(self.config.L_cycles):
                 z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
             z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)
             return z_H, z_L

        # No-grad steps
        for _ in range(self.config.H_cycles - 1):
            z_H = jax.lax.stop_gradient(z_H)
            z_L = jax.lax.stop_gradient(z_L)
            z_H, z_L = body_fn(z_H, z_L)
            
        # Final step with grad
        z_H, z_L = body_fn(z_H, z_L)
        
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=jax.lax.stop_gradient(z_H), z_L=jax.lax.stop_gradient(z_L))
        
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0])
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    def empty_carry(self, batch_size: int):
        shape = (batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size)
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=jnp.empty(shape, dtype=self.forward_dtype),
            z_L=jnp.empty(shape, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: jax.Array, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        # reset_flag: [B]
        reset_flag = reset_flag[:, None, None]
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=jnp.where(reset_flag, self.H_init[None, None, :], carry.z_H),
            z_L=jnp.where(reset_flag, self.L_init[None, None, :], carry.z_L),
        )

class TinyRecursiveReasoningModel_ACTV1(nnx.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, rngs: nnx.Rngs):
        self.config = config
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(config, rngs=rngs)

    def __call__(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, jax.Array], rngs: nnx.Rngs = None) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, jax.Array]]:
        
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = jnp.where(carry.halted, 0, carry.steps)
        
        # Update current data (if halted, replace with new batch data)
        # Note: In JAX, we usually pass the whole batch and mask.
        # Here we assume 'batch' contains the data for the *current* step for all slots.
        # But wait, the PyTorch code says:
        # new_current_data = {k: torch.where(carry.halted..., batch[k], v) ...}
        # This implies 'batch' is the NEW data coming in, and 'carry.current_data' is the OLD data.
        # If a slot halted, we take from 'batch'. If not, we keep 'carry.current_data'.
        
        def update_data(k, v_old, v_new):
             # Broadcast halted to match v shape
             ndim = v_old.ndim
             h = carry.halted.reshape((-1,) + (1,) * (ndim - 1))
             return jnp.where(h, v_new, v_old)

        new_current_data = {k: update_data(k, carry.current_data[k], batch[k]) for k in batch.keys()}
        
        # Forward inner
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # ACT Logic
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step
        
        # If training and ACT enabled
        # We need to handle randomness (exploration)
        # JAX requires explicit RNG
        
        if self.config.halt_max_steps > 1:
             if self.config.no_ACT_continue:
                 should_halt = q_halt_logits > 0
             else:
                 should_halt = q_halt_logits > q_continue_logits
                 
             halted = halted | should_halt
             
             # Exploration
             if rngs is not None:
                 exploration_mask = jax.random.uniform(rngs.exploration()) < self.config.halt_exploration_prob
                 min_halt_steps = jax.random.randint(rngs.exploration(), new_steps.shape, 2, self.config.halt_max_steps + 1)
                 # If exploring, force run at least min_halt_steps
                 # halted = halted & (new_steps >= min_halt_steps)
                 # Logic in PyTorch:
                 # min_halt_steps = (rand < prob) * randint(...)
                 # halted = halted & (new_steps >= min_halt_steps)
                 # If rand >= prob, min_halt_steps is 0. new_steps >= 0 is True. So halted is unchanged.
                 # If rand < prob, min_halt_steps is e.g. 5. If new_steps < 5, halted becomes False (forced continue).
                 
                 # JAX equivalent:
                 # We need to be careful with shapes.
                 min_halt_steps = jnp.where(exploration_mask, min_halt_steps, 0)
                 halted = halted & (new_steps >= min_halt_steps)

             # Target Q (Deep Supervision for ACT)
             if not self.config.no_ACT_continue:
                 # We need next step Q values. This requires another forward pass?
                 # PyTorch code does: self.inner(new_inner_carry, new_current_data) AGAIN.
                 # This is expensive but correct for Q-learning target.
                 _, _, (next_q_halt, next_q_cont) = self.inner(new_inner_carry, new_current_data)
                 target_q = jax.nn.sigmoid(jnp.where(is_last_step, next_q_halt, jnp.maximum(next_q_halt, next_q_cont)))
                 outputs["target_q_continue"] = target_q

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs

    def initial_carry(self, batch: Dict[str, jax.Array]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            halted=jnp.ones((batch_size,), dtype=jnp.bool),
            current_data={k: jnp.zeros_like(v) for k, v in batch.items()} # Initialize with zeros, will be replaced
        )
