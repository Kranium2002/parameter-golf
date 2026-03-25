# Parameter Golf — Battle Plan v3

**Goal:** Beat current SOTA 1.1194 bpb. Need < 1.1165 bpb (0.005 nats improvement threshold).
**Hardware:** 1x H100 80GB for smoke tests, 8x H100 SXM for submission runs.
**Deadline:** April 30, 2026.

---

## Current SOTA Breakdown (1.1194 bpb, Mar 23)

Pre-TTT: **1.1218** bpb | Post-TTT: **1.1194** bpb | Artifact: ~15.95 MB

| Component | Setting |
|---|---|
| Architecture | 11L, 512d, 8H/4KV, U-Net skip connections |
| MLP | 3x expansion, LeakyReLU(0.5)² |
| Attention tricks | XSA last 4 layers, Partial RoPE (16/64), logit softcap 30 |
| Embeddings | BigramHash(1536), SmearGate, VE128 on layers 9-10, tied |
| Normalization | RMSNorm with LN Scale 1/sqrt(layer+1) |
| Weight averaging | EMA(0.997) + Tight SWA(every 50 steps) |
| Optimizer | Parallel Muon (parameter banking) + AdamW for non-bank params |
| Training | seq_len=2048, batch=786K tokens, warmdown 3500, late QAT at 0.15 |
| Quantization | GPTQ-lite int6 (5 clip percentiles) + lzma |
| Eval | Sliding window stride=64, Legal TTT (SGD, 3ep, 32K chunks, all blocks) |
| Timing | 83.4ms/step, ~7185 steps in 600s train, ~530s eval (120s standard + 410s TTT) |

### SOTA Ablation Stack (from submission README)

| Change | Cumulative bpb | Delta |
|---|---|---|
| PR #414 base (relu²) | 1.1234 | — |
| + Parameter Banking | 1.1234 | ±0.0000 |
| + Legal TTT (3ep, freeze=2) | 1.1217 | -0.0017 |
| + TTT freeze=0 (all blocks) | 1.1213 | -0.0004 |
| + BigramHash 2048→1536 | 1.1204 | -0.0009 |
| + LeakyReLU(0.5)² | **1.1183** | **-0.0021** |

**Key insight:** LeakyReLU(0.5)² was the single biggest win (-0.0021). TTT total was -0.0021. BigramHash tuning was -0.0009.

---

## What We Already Tried (1x H100 Smoke Tests, 500 steps)

All smoke tests use 131K tokens/batch, seq_len=2048, warmdown=100 steps.
**Absolute bpb values are not comparable to 8xH100 runs** — only relative ordering matters.

### Results Table

| Experiment | val_bpb@500 | Δ vs baseline | Step time | Verdict |
|---|---|---|---|---|
| **Baseline** (11L, SOTA arch, no TTT) | **1.5042** | reference | 230ms | ✅ Reference |
| 13 layers (no int4) | **1.4978** | **-0.0064** | 246ms | ✅ **Best result** |
| 13L + int4 MLP QAT | **1.4987** | **-0.0055** | 242ms | ✅ Best with budget fit |
| VE on layers 7-10 (vs 9-10) | 1.5004 | -0.0038 | 229ms | ✅ Small win |
| XSA last 6 (vs 4) | 1.5055 | +0.0013 | 230ms | ❌ Worse |
| BigramHash(4096) (vs 2048) | 1.5078 | +0.0036 | 230ms | ❌ Worse |
| seq_len=3072 | 1.5097 | +0.0055 | 231ms | ❌ Worse (fewer tokens/batch) |
| 13L + VE layers 9-12 | 1.5015 | -0.0027 | 246ms | ⚠️ Worse than 13L alone |
| 4 Conv blocks + 7 Transformer | 1.8689 | +0.3647 | 208ms | ❌ **Much worse** |
| 2 Conv blocks + 9 Transformer | 1.8866 | +0.3824 | 215ms | ❌ **Much worse** |

### What Failed and Why

#### ❌ Mamba / SSM Hybrid (Vector 3 — ABANDONED)

**What we tried:**
1. Full Mamba S6 selective scan (PyTorch implementation)
2. Chunked sequential scan with `@torch.compiler.disable`
3. Parallel cumsum-based scan (log-space trick)
4. Simplified gated causal convolution blocks (multi-scale and single-scale)

**Why it failed:**
- **Sequential scan is too slow without CUDA kernels.** The T=2048 step loop takes ~127ms per block in PyTorch. With 4 Mamba layers, that's 500ms+ per forward pass — slower than the entire baseline step (230ms). `mamba_ssm` and `causal_conv1d` CUDA packages won't compile on our torch 2.4.1+cu124 environment.
- **Parallel scan has numerical instability.** The log-space cumsum trick (`exp(cumsum(log_decay)) * cumsum(exp(-cumsum(log_decay)) * x)`) overflows when cumulative log-decay is large (which it is at T=2048). Results in NaN.
- **Simple causal convolutions don't learn as well as attention.** Replacing even 2 of 11 attention layers with depthwise conv blocks degraded bpb by +0.38 — a massive quality loss. Convolutions simply can't capture the token-to-token dependencies that attention provides at this scale.
- **torch.compile interaction.** Sequential scan breaks `fullgraph=True`. With `fullgraph=False` + `@torch.compiler.disable`, the scan isn't compiled and runs at Python speed. No way to make it fast without custom CUDA kernels.

**Conclusion:** SSM/Mamba is not viable for this challenge without writing custom CUDA kernels, which is a multi-day effort with uncertain payoff. The compile+kernel situation makes it impractical. Pure transformer with more layers is strictly better.

#### ❌ XSA on More Layers

XSA (exclusive self-attention) on last 6 layers was slightly worse than last 4. The SOTA already found the optimal setting.

#### ❌ BigramHash Expansion

BigramHash(4096) was worse than the SOTA's 1536. More hash buckets didn't help — the model already captures enough bigram patterns.

#### ❌ Longer Training Sequence (3072)

seq_len=3072 was worse because it reduces tokens per batch (must be divisible by seq_len × grad_accum). The model gets fewer total tokens per step, hurting convergence. The SOTA's seq_len=2048 with stride=64 sliding window eval already captures long-range dependencies at eval time.

### What Worked

#### ✅ 13 Layers (-0.006 bpb in smoke tests)

Adding 2 more layers (11→13) was the biggest single improvement. The U-Net structure scales naturally (6 encoder + 7 decoder). Step time increases by ~7% (230→246ms), which means ~6% fewer steps in 600s, but the per-step quality more than compensates.

**Budget problem:** 13L at int6 doesn't fit in 16MB (~22MB compressed). Solved with int4 MLP QAT.

#### ✅ Int4 MLP QAT (enables 13L in 16MB)

Int4 STE fake-quantization for MLP bank weights (clip_range=7, 4-bit signed). Applied during warmdown via `LATE_QAT_THRESHOLD`. Frees ~3MB from MLP weights, bringing 13L down to ~15MB compressed. Quality impact is minimal since QAT only activates during warmdown (last ~500 steps of 7000).

#### ✅ VE on More Decoder Layers (-0.004 bpb)

Value Embedding on layers 7-10 instead of 9-10 gave a small but consistent improvement. Cost is negligible (~200KB compressed).

---

## What TTT Testing Showed

TTT (Test-Time Training) could not be properly validated on 1x H100:
- Sliding window eval at stride=64 takes **~670 seconds** on 1 GPU (vs ~120s on 8 GPUs)
- TTT scoring phase is I/O bound by the number of windows (969K windows for the full val set)
- TTT training is only 5% of wall-clock — the bottleneck is scoring

**The enhanced TTT code is implemented** (`train_ttt.py`) but must be validated on 8xH100:
- AdamW optimizer (vs SGD)
- 8 epochs per chunk (vs 3)
- Per-layer LR scaling (deeper layers get higher LR)
- Warmup within each chunk (first epochs at 0.3x LR)
- Configurable chunk size (64K vs 32K)

---

## Revised Strategy: What Can Beat 1.1194

### Tier 1: High Confidence (implement first)

#### 1A. 13 Layers + Int4 MLP (~-0.003 to -0.005 bpb)

**Status: Code ready in `train_ttt.py`. Needs 8xH100 validation.**

The smoke tests show 13L is strictly better than 11L. With int4 MLP QAT, the artifact fits in 16MB. On 8xH100, the step time penalty (~7%) means ~6700 steps vs ~7185, but the quality gain should dominate.

**Estimated pre-TTT bpb:** 1.1218 × (1.4978/1.5042) ≈ **1.117** (very rough extrapolation)

This alone may not beat the threshold, but combined with enhanced TTT it should.

#### 1B. Enhanced TTT (~-0.003 to -0.005 additional bpb)

**Status: Code ready in `train_ttt.py`. Needs 8xH100 validation.**

Current TTT provides -0.0025 bpb with basic SGD/3 epochs. Training is only 5% of TTT wall-clock. We can:

| Change | Expected impact | Time cost |
|---|---|---|
| AdamW instead of SGD | Better per-step adaptation | +0s (same FLOPs) |
| 8 epochs instead of 3 | 2.7x more adaptation | +10s (<3% budget) |
| Per-layer LR (1x→2x) | Better deep-layer adaptation | +0s |
| Warmup within chunk | More stable optimization | +0s |

**Conservative estimate:** TTT improves from -0.0025 to -0.005 bpb (2x improvement).

#### Combined Tier 1 estimate: 1.1218 - 0.004 (depth) - 0.005 (TTT) ≈ **1.113 bpb** ✅ beats threshold

### Tier 2: Medium Confidence (implement if Tier 1 isn't enough)

#### 2A. Warmdown Tuning

Current: 3500 steps. Test 4000 or 4500. Longer warmdown → tighter weight distributions → better quantization. The SOTA's warmdown is already good but may not be optimal for 13L.

#### 2B. Activation Sweep on 13L

LeakyReLU(0.5)² was tuned for 11L. The optimal α might differ for 13L. Quick sweep of α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} on the 13L architecture.

#### 2C. VE Layer Expansion

Test VE on layers 7-12 (all decoder layers) with 13L architecture. Our smoke test showed VE 7-10 helped on 11L.

#### 2D. SwiGLU Activation

Replace LeakyReLU(0.5)² with SwiGLU: `silu(W_gate·x) * W_up·x`. Requires adding a gate projection (doubles MLP input). Can offset by reducing `mlp_mult` from 3 to 2 (gated 2x ≈ ungated 3x in expressiveness). This changes the MLP budget math and needs careful int4 QAT tuning.

### Tier 3: Low Confidence (only if we have time)

#### 3A. LoRA for TTT

Add rank-8 LoRA to Q/K/V/O projections during TTT. Only adapt LoRA params, freeze base weights. Merge at end of each chunk. This adds implicit regularization and may prevent overfitting during extended TTT epochs.

#### 3B. Hessian-Weighted GPTQ

Replace the clip percentile search with Hessian-weighted optimal rounding. Compute diagonal Hessian from 1 calibration batch. Use Hessian to weight the rounding error, preferring to quantize low-sensitivity weights more aggressively.

#### 3C. Trigram Hash Embedding

Add a separate trigram hash embedding table alongside bigram. Captures 3-token patterns at ~200KB compressed cost.

---

## What NOT to Try (Confirmed Dead Ends)

| Idea | Why it's dead |
|---|---|
| **Mamba/SSM hybrid** | Too slow without CUDA kernels. PyTorch scan = 127ms/layer. Conv fallback loses 0.38 bpb. |
| **RWKV** | Same problem as Mamba — sequential scan without kernels is impractical |
| **Kronecker factorization** | Not needed when int4 MLP fits 13L in 16MB |
| **Layer recurrence/looping** | Explicitly failed in prior submission (-0.051 bpb) |
| **Custom CUDA kernels** | Multi-day effort, uncertain payoff, fragile across environments |
| **Novel tokenizer** | Rules scrutinize this heavily, not worth the risk |
| **seq_len > 2048 for training** | Reduces tokens/batch, hurts convergence. Eval already uses sliding window. |
| **More BigramHash buckets** | 4096 was worse than 1536 |
| **XSA on more than 4 layers** | 6 layers was worse than 4 |

---

## Implementation Checklist

### Already Done ✅

- [x] 1x H100 compatible base script (`train_experiment.py`)
- [x] SDPA fallback for flash_attn_3
- [x] SMOKE_TEST mode for fast iteration
- [x] 13L architecture support
- [x] Int4 MLP STE QAT (`_int4_ste` in MLP forward)
- [x] Int4 per-row quantization with GPTQ-lite clip search
- [x] Mixed quantization pipeline (int4 MLP + int6 attn)
- [x] Enhanced TTT: AdamW optimizer option
- [x] Enhanced TTT: per-layer LR scaling
- [x] Enhanced TTT: warmup within chunks
- [x] Enhanced TTT: configurable chunk limit for testing
- [x] Mamba/conv block experiments (concluded: not viable)
- [x] Micro-optimization sweep (XSA, VE, BigramHash, seq_len)

### Next Steps (Need 8xH100)

- [ ] Validate 13L + int4 MLP on 8xH100 (full 600s training)
- [ ] Verify artifact fits in 16MB after full quantization
- [ ] Test enhanced TTT variants (AdamW/8ep vs SGD/3ep)
- [ ] Run 3-seed validation (1337, 42, 2025)
- [ ] Tune warmdown for 13L (may need 4000+ steps)
- [ ] Activation α sweep on 13L
- [ ] Prepare submission PR

---

## Files

| File | Purpose |
|---|---|
| `train_experiment.py` | Experiment harness with SDPA fallback, smoke test mode, Mamba blocks |
| `train_ttt.py` | Submission candidate: 13L + int4 MLP + enhanced TTT |
| `plan.md` | This file |

---

*Plan v3 — March 25, 2026. Updated with experimental results. Supersedes v2.*
