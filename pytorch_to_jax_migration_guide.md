# PyTorch to JAX Migration Manual: A Boilerplate for Future Projects

This document serves as a comprehensive guide for porting Deep Learning models from PyTorch to JAX (specifically Flax/NNX). It is divided into **General Requirements** (applicable to any project) and **Project-Specific Requirements** (lessons learned from the TRM/ALBERT adaptation).

---

## Part 1: General Requirements (Universal)

These requirements apply to *any* PyTorch-to-JAX migration project. Failure to follow these will result in subtle bugs, numerical mismatches, or training instability.

### 1. Pre-Flight Analysis
Before writing code, analyze the PyTorch source to understand its "Hidden Contract".

*   **Statefulness:** Does the model maintain state between calls (RNNs) or is it purely functional?
    *   *Test:* Run the model twice on the same input. If outputs differ (with dropout disabled), it has hidden state.
*   **Randomness:** Identify all sources of randomness (`Dropout`, `Gumbel-Softmax`, `Random Sampling`).
    *   *JAX Requirement:* You must pass explicit `rngs` to these layers.
*   **Normalization:** Identify `LayerNorm` vs `RMSNorm` and their exact configurations (`epsilon`, `bias`).
    *   *Trap:* PyTorch `RMSNorm` often uses `eps=1e-5`, while JAX defaults might differ. Mismatches here cause NaNs.

### 2. The "Golden Copy" Testing Strategy
Do not rely on "it looks correct". You must generate ground-truth data from PyTorch.

**Step 1: Generate Artifacts (PyTorch Side)**
Create a script `export_debug_data.py` that saves:
1.  **Model Weights:** `state_dict`.
2.  **Fixed Input:** A batch of random data (fixed seed).
3.  **Intermediate Outputs:** Output of *each* major block (Embeddings, Layer 1, Layer N, Head).
4.  **Final Output:** Logits/Loss.
5.  **Gradients:** Gradients of the first layer weights after one backward pass.

**Step 2: The "One-Step" Parity Test (JAX Side)**
Create a test `test_parity.py` that:
1.  Loads the PyTorch weights into the JAX model.
2.  Runs the *exact same* fixed input.
3.  Asserts `abs(jax_out - torch_out) < 1e-4` (float32) or `1e-2` (bfloat16).
4.  **Crucial:** Test *layer by layer*. If the final output mismatches, check Embeddings first, then Layer 1, etc.

### 3. Debugging Implementation Mismatches
*   **Linear Layers:** PyTorch `Linear` initializes weights from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$. JAX defaults vary. *Always* load PyTorch weights for parity checks.
*   **Embeddings:** PyTorch `Embedding` initializes from $\mathcal{N}(0, 1)$. JAX `Embed` defaults to Uniform.
*   **Broadcasting:** PyTorch broadcasts automatically and silently. JAX is stricter. Verify shapes explicitly.
*   **Einsum:** Use `jnp.einsum` to match PyTorch's `torch.einsum` logic exactly. It's safer than `reshape` + `transpose`.

### 4. JAX/Flax Best Practices
*   **`nnx.remat` (Gradient Checkpointing):** Use this for any model deeper than 12-24 layers to save memory.
*   **`nnx.scan` vs Python Loop:**
    *   Use **Python Loop** (unrolled) for short sequences (< 20 steps) or when debugging. It compiles faster.
    *   Use **`nnx.scan`** for long sequences to reduce compilation time and binary size.
*   **Optimizer Mapping:**
    *   PyTorch `AdamW` $\neq$ Optax `adamw` by default. Check `weight_decay` handling (PyTorch decouples it, Optax might not depending on version/flags).
    *   **Clipping:** PyTorch often clips *after* computing norms. Optax requires explicit chaining: `optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(...))`.

---

## Part 2: Project-Specific Requirements (TRM/ALBERT Lessons)

These are specific lessons learned from porting the **Tiny Recursive Model (TRM)** and **ALBERT**, which involve **Recurrence**, **Adaptive Computation Time (ACT)**, and **Deep Supervision**.

### 1. The "Recurrence" Trap
**Issue:** Many "ACT" models in PyTorch are actually **Stateless**. They run the same layer $N$ times on the *same input* without passing a hidden state.
**Symptom:** Porting them as "Recurrent" (passing `h_t` to `h_{t+1}`) creates a feedback loop that explodes gradients.
**Solution:**
*   **Verify Recurrence:** Run the "Idempotency Test" (Part 1).
*   **If Recurrent:** You **MUST** implement **Input Injection** (`h_t = h_{t-1} + Input`). Without this, the model "forgets" the input and stagnates (e.g., at 39% accuracy).
*   **If Stateless:** Do not pass `prev_hidden_states`. Just loop over the layer.

### 2. Stability on TPU (The "Valley of NaNs")
**Issue:** TPUs are sensitive to unstable gradients, especially in recurrent loops.
**Symptom:** `grad_norm = inf` or `NaN` loss after a few steps.
**Checklist:**
1.  **Zero Input Fix:** In `scan` loops, the "previous state" for Step 0 is often initialized to zeros. Ensure Step 0 logic explicitly uses the *input embeddings*, not the zero state.
2.  **Truncated BPTT:** If Full BPTT explodes, use `jax.lax.stop_gradient(prev_state)` to cut the graph. This stabilizes training at the cost of long-term planning.
3.  **RMSNorm Epsilon:** Ensure `epsilon=1e-6` (or match PyTorch). `1e-5` can be too large/small depending on the scale.
4.  **Gradient Clipping:** **Mandatory**. Clip by global norm (1.0) *before* the optimizer update.

### 3. Deep Supervision & Loss
**Issue:** TRM/ALBERT uses "Deep Supervision" (summing loss from every ACT step).
**Implementation:**
*   **JAX:** Collect outputs in a list `[out_0, out_1, ...]`.
*   **Loss:** `total_loss = sum(loss_fn(out) for out in outputs)`.
*   **Normalization:** Remember to divide by `num_steps` if the PyTorch code does (or if using `mean`).

### 4. Performance Optimization
**Issue:** Unrolling a 12-step loop with a 6-layer ALBERT creates a 72-layer network, causing OOM.
**Solution:**
*   **`nnx.remat`:** Apply `nnx.remat` to the *inner* ALBERT layer. This reduces memory complexity from $O(L \times T)$ to $O(L + T)$.
*   **Scan:** Use `nnx.scan` if the loop size is dynamic or very large.

### 5. Summary of TRM-Specific Bugs
| Bug | Symptom | Fix |
| :--- | :--- | :--- |
| **Zero Input Bug** | NaNs at Step 0 | Handle `is_first_step` explicitly. |
| **Stagnation (39%)** | Model predicts trivial baseline | Add **Input Injection** (`h + emb`). |
| **Exploding Gradients** | `gnorm=inf` | Use **Truncated BPTT** (`stop_gradient`) or fix Zero Input. |
| **OOM** | TPU runs out of memory | Use `nnx.remat` on the inner block. |
