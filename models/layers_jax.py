
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Callable, Any
import math

# --- Layers ---

class RMSNorm(nnx.Module):
    def __init__(self, dim: int, epsilon: float = 1e-6, rngs: nnx.Rngs = None):
        self.epsilon = epsilon
        self.weight = nnx.Param(jnp.ones((dim,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        hidden_states = x * jax.lax.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states

class LinearSwish(nnx.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, rngs: nnx.Rngs = None):
        self.linear = nnx.Linear(in_features, out_features, use_bias=bias, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return nnx.swish(self.linear(x))

class SwiGLU(nnx.Module):
    def __init__(self, hidden_size: int, expansion: float, rngs: nnx.Rngs):
        intermediate_size = int(hidden_size * expansion)
        self.w1 = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(intermediate_size, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w3(nnx.swish(self.w1(x)) * self.w2(x))

class RotaryEmbedding(nnx.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute cos/sin
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim))
        t = jnp.arange(self.max_position_embeddings, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos = jnp.cos(emb)
        self.sin = jnp.sin(emb)

    def __call__(self, seq_len: int) -> tuple[jax.Array, jax.Array]:
        return self.cos[:seq_len], self.sin[:seq_len]

def rotate_half(x: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array) -> tuple[jax.Array, jax.Array]:
    # q, k: [B, H, L, D] or [B, L, H, D] depending on implementation. Assuming [B, L, H, D] for now to match typical JAX
    # cos, sin: [L, D]
    
    # Reshape cos/sin for broadcasting
    # Assuming q, k are [Batch, SeqLen, NumHeads, HeadDim]
    # cos, sin are [SeqLen, HeadDim] -> [1, SeqLen, 1, HeadDim]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nnx.Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, num_key_value_heads: int, causal: bool = False, rngs: nnx.Rngs = None):
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal
        
        self.q_proj = nnx.Linear(hidden_size, num_heads * head_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, num_key_value_heads * head_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, num_key_value_heads * head_dim, use_bias=False, rngs=rngs)
        self.o_proj = nnx.Linear(num_heads * head_dim, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, cos_sin: Optional[tuple[jax.Array, jax.Array]] = None, mask: Optional[jax.Array] = None) -> jax.Array:
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, L, self.num_key_value_heads, self.head_dim)
        
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            
        # GQA / MQA handling (repeat k/v if needed)
        if self.num_key_value_heads != self.num_heads:
             k = jnp.repeat(k, self.num_heads // self.num_key_value_heads, axis=2)
             v = jnp.repeat(v, self.num_heads // self.num_key_value_heads, axis=2)

        # Attention
        # [B, L, H, D] -> [B, H, L, D]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn_weights = attn_weights + mask
            
        if self.causal:
            # Create causal mask
            i, j = jnp.indices((L, L))
            causal_mask = jnp.where(i >= j, 0.0, -jnp.inf)
            attn_weights = attn_weights + causal_mask[None, None, :, :]

        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_output = jnp.matmul(attn_weights, v)
        
        # [B, H, L, D] -> [B, L, H, D] -> [B, L, D_total]
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        
        return self.o_proj(attn_output)

class CastedEmbedding(nnx.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 0.02, cast_to: Any = jnp.float32, rngs: nnx.Rngs = None):
        self.embedding = nnx.Embed(num_embeddings, embedding_dim, rngs=rngs)
        self.cast_to = cast_to
        # Custom init
        self.embedding.embedding.value = jax.random.normal(rngs.params(), (num_embeddings, embedding_dim)) * init_std

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.embedding(x).astype(self.cast_to)

class CastedLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, cast_to: Any = jnp.float32, rngs: nnx.Rngs = None):
        self.linear = nnx.Linear(in_features, out_features, use_bias=bias, rngs=rngs)
        self.cast_to = cast_to

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x).astype(self.cast_to)
