# Parameter Golf — Battle Plan v2

**Goal:** Beat current SOTA 1.1194 bpb. Need < 1.1165 bpb (0.005 nats improvement threshold).
**Hardware:** 1x RTX 5090 for smoke tests, 8x H100 SXM for submission runs.
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

---

## Strategy: Three Independent Attack Vectors

Each vector is independently testable on the 5090 and can be combined.

### Vector 1: Enhanced TTT (~-0.003 additional bpb)

**The biggest low-hanging fruit.** Current TTT is basic SGD with 3 epochs.
Training is only **5% of TTT wall-clock** — scoring (inference) is 95%.
We can make training 3x more expensive with <15s additional wall-clock cost.

| Change | Why | Implementation |
|---|---|---|
| **AdamW** instead of SGD | Better per-step adaptation, handles scale differences across layers | Replace `torch.optim.SGD` with `torch.optim.AdamW(ttt_params, lr=0.001, betas=(0.9, 0.99))` in `eval_val_sliding_ttt` |
| **8 epochs** instead of 3 | 2.7x more adaptation per chunk. Cost: +10s total (<3% of eval budget) | Change `args.ttt_epochs` from 3 to 8 |
| **LoRA adaptation** instead of full weights | Lower effective dimensionality → more stable optimization, less overfitting | Add rank-8 LoRA to Q/K/V/O projections. Only adapt LoRA params during TTT, freeze base weights. Merge at end of each chunk. |
| **Per-layer LR** | Deep layers adapt faster and benefit more from TTT | Scale LR by `(1 + layer_idx / num_layers)` — 2x LR for last layer vs first |
| **Larger chunks** (64K) | More context per adaptation step. Halves chunk count → half the scoring passes | Change `ttt_chunk_tokens` from 32768 to 65536 |
| **Warmup within chunk** | First epoch at 0.3x LR, ramp to full by epoch 3 | Linear warmup across epochs within each chunk |

**Expected gain:** -0.003 to -0.005 bpb on top of current -0.0025 (total -0.006 to -0.008 from TTT).

**Risks:** Overfitting per chunk. Mitigated by LoRA (implicit regularization) and larger chunks.

**5090 test protocol:**
1. Train a small model (500 steps), save checkpoint
2. Run TTT with SGD/3ep on first 100 val chunks → measure bpb
3. Run TTT with AdamW/8ep on same 100 chunks → measure bpb
4. Run TTT with LoRA/AdamW/8ep → measure bpb
5. Compare. If AdamW/8ep beats SGD/3ep by >0.001, the vector is confirmed.

---

### Vector 2: Mixed Quantization → More Layers (~-0.002 to -0.004 bpb)

**Core insight:** MLP weights are the largest tensors and most tolerant of aggressive quantization.
Going from int6 to int4 for MLPs frees ~3-4MB, enough for 2 more layers.

**Budget math:**

Current 11L model:
- 4 parameter banks: qo_bank (22×512×512), kv_bank (22×256×512), mlp_up (11×1536×512), mlp_down (11×512×1536)
- MLP banks alone: 11 × (1536×512 + 512×1536) = 17.3M params = ~69MB FP32
- At int6+lzma: MLP contribution is ~8MB of the 15.95MB artifact
- At int4+lzma: MLP contribution would be ~5MB
- **Freed: ~3MB → enough for 2 more layers**

13L model with int4 MLPs + int6 attention:
- Extra attention: 2 × (512×512 + 256×512 + 256×512 + 512×512) = 1.8M params → ~0.9MB int6
- Extra MLP: 2 × (1536×512 + 512×1536) = 3.1M params → ~1.3MB int4
- Extra control params: ~0.01MB
- **Total extra: ~2.2MB → fits in the 3MB freed**

| Change | Implementation |
|---|---|
| **Int4 QAT for MLP weights** | New `quantize_int4_per_row` function: clip_range=7, 4-bit signed. STE fake-quant during training with `LATE_QAT_THRESHOLD`. |
| **Keep int6 for attention** | Attention is more precision-sensitive. No change for Q/K/V/O banks. |
| **Keep int8 for embeddings** | Embeddings serve dual purpose (input + output). Most critical. |
| **13 layers** | `NUM_LAYERS=13`. U-Net: 6 encoder + 7 decoder. Add 2 more banks. |
| **Full GPTQ with Hessian** | Post-training: compute diagonal Hessian from 1 calibration batch. Use Hessian-weighted rounding instead of just clip percentile search. |

**Expected gain:** -0.002 to -0.004 bpb from depth increase.

**Risks:** Int4 MLP quality loss exceeds depth gain. Mitigated by:
- QAT during training ensures model adapts to quantization
- GPTQ with Hessian improves rounding decisions
- Ablation: test int4 quality on the existing 11L model first

**5090 test protocol:**
1. Train 11L baseline 500 steps with int6 QAT → record val_loss
2. Train 11L with int4 MLP QAT → record val_loss (quality degradation measurement)
3. Train 13L with int4 MLP QAT → record val_loss (depth gain vs quality loss)
4. If 13L-int4 val_loss < 11L-int6 val_loss → vector confirmed

---

### Vector 3: Mamba Hybrid (~-0.001 to -0.003 bpb, novel)

**The creative bet.** Replace the first 3-4 layers with Mamba (S6) blocks.
Nobody in the competition has tried this.

**Rationale:**
- Early layers primarily learn **local patterns** (n-grams, syntax, morphology)
- Mamba's selective state space mechanism is specifically designed for sequential local processing
- Mamba layers have **no KV projections, no attention matrix** → smaller per-layer footprint
- O(T) compute → can train at longer seq_len without cost increase
- Different inductive bias from attention → captures different patterns → complementary

**Architecture:**

```
Layer 0-3:   MambaBlock (selective scan, D=512, state_dim=16, conv_width=4)
Layer 4-12:  TransformerBlock (existing architecture with XSA on last 4)
Total:       13 layers (4 Mamba + 9 Transformer)
```

**Mamba block implementation:**

```python
class MambaBlock(nn.Module):
    """Simplified Mamba (S6) block for hybrid architecture."""
    def __init__(self, dim, state_dim=16, conv_width=4, expand=2):
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)  # x and z branches
        self.conv1d = nn.Conv1d(inner, inner, conv_width, padding=conv_width-1, groups=inner)
        self.x_proj = nn.Linear(inner, state_dim * 2, bias=False)  # dt, B
        self.dt_proj = nn.Linear(state_dim, inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim+1).float().repeat(inner, 1)))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.norm = RMSNorm()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :x.size(1)].transpose(1, 2)
        x_branch = F.silu(x_branch)
        # Selective scan (simplified — use mamba_ssm package for real kernel)
        dt_B = self.x_proj(x_branch)
        dt, B = dt_B.split([self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        # ... selective scan computation ...
        y = self.out_proj(y * F.silu(z))
        return residual + y
```

**Per-layer budget comparison:**

| | Transformer layer | Mamba layer (expand=2) |
|---|---|---|
| Q/K/V/O projections | 512² + 256×512×2 + 512² = 786K | 0 |
| MLP up+down | 1536×512 × 2 = 1.57M | 0 (absorbed into gate) |
| In/out projection | 0 | 512×1024×2 + 1024×512 = 1.57M |
| Conv1d | 0 | 1024×4 = 4K |
| SSM params | 0 | 1024×16×2 + 16×1024 = 49K |
| **Total** | **~2.36M** | **~1.62M** |

Mamba layers are **31% smaller** → 4 Mamba layers free ~3M params = ~12MB FP32 → ~0.9MB compressed.

**Combined with Vector 2:** 4 Mamba + 9 Transformer + int4 MLP could potentially reach 14-15 total layers.

**Expected gain:** -0.001 to -0.003 bpb from better local pattern capture + freed budget.

**Risks:**
- Mamba may not converge as well as attention for this task/scale
- `mamba_ssm` CUDA package dependency (may not be in RunPod image)
- Integration with parameter banking and Parallel Muon needs work

**5090 test protocol:**
1. Install `mamba-ssm` package: `pip install mamba-ssm`
2. Train hybrid (4 Mamba + 7 Transformer = 11L equivalent) 500 steps → record val_loss
3. Compare vs pure 11L transformer at same step count
4. If within 0.01 val_loss → Mamba hybrid is viable for further tuning

---

## Additional Micro-Optimizations (Stack on Top)

These are small independent improvements that can each contribute -0.0005 to -0.002 bpb:

### 3a. Activation Search

LeakyReLU(0.5)² was found by experimentation. Test alternatives:

| Activation | Formula | Why it might help |
|---|---|---|
| **SwiGLU²** | `(silu(W_gate·x) * W_up·x)²` | Gated activation captures non-linearities better. Used in Llama/Gemma. |
| **GEGLU²** | `(gelu(W_gate·x) * W_up·x)²` | Smoother than SiLU gate. |
| **LeakyReLU(α)²** sweep | α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} | 0.5 may not be optimal |

Note: SwiGLU/GEGLU need a gate projection → doubles MLP input size. Must verify budget still works.
Can offset by reducing `mlp_mult` from 3 to 2 (gated MLP with 2x expansion ≈ ungated 3x in expressiveness).

### 3b. XSA on More Layers

Currently XSA on last 4. Test last 5 or 6. Cost is ~0.5ms/step per additional layer.

### 3c. Value Embedding on More Layers

Currently VE128 on layers 9-10 only. Test layers 7-10 or all decoder layers.
VE is cheap (~128 × vocab per layer = 131K params → ~50KB compressed).

### 3d. Warmdown Tuning

Test warmdown 4000 or 4500 (vs current 3500). Longer warmdown → tighter weight distributions → better quantization.

### 3e. Sequence Length During Training

Current: 2048. Test 3072 or 4096. Longer sequences improve loss but reduce steps/minute.
The eval at stride=64 already benefits from long context — but training at long context
could improve the model's ability to use that context.

Break-even analysis: if step time goes from 83ms to 125ms (50% increase for seq_len 3072),
we lose ~2400 steps (7185→4790). Need the per-step improvement to exceed
`(7185-4790) × bpb_per_step_rate`. Test on 5090 to measure the actual tradeoff.

### 3f. BigramHash Expansion

Test BigramHash(3072) or BigramHash(4096). Current 1536 buckets may have too many collisions.
Cost: (4096-1536) × 128 = 328K extra params → ~100KB compressed. Negligible.

### 3g. Trigram or Skip-gram Hash

Extend the hash embedding to capture 3-token patterns or skip-1 bigrams:
```python
# Trigram: XOR of three consecutive tokens
out[..., 2:] = bitwise_xor(t[..., 2:], bitwise_xor(36313 * t[..., 1:-1], 27191 * t[..., :-2])) % mod
```
Separate embedding table, added alongside bigram. Cost: another ~200KB compressed.

---

## 5090 Smoke Test Protocol

### Environment Setup

```bash
cd /root/parameter-golf

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece numpy huggingface-hub datasets

# Download 1 train shard + full val set
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Copy SOTA script as our base
cp records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py train_experiment.py
```

### Adapting SOTA Script for 1x 5090

Key changes needed in `train_experiment.py`:
1. Remove `flash_attn_interface` import → use `torch.nn.functional.scaled_dot_product_attention`
2. Reduce `train_batch_tokens` to fit 32GB VRAM (131072 or 65536)
3. Force `grad_accum_steps = 1` for single GPU
4. Remove DDP-specific code paths (or let it run with `nproc_per_node=1`)
5. Disable TTT for architecture comparison tests (enable only for TTT tests)

### Test Matrix (Priority Order)

All tests: 500 steps, val at step 250 and 500, `MAX_WALLCLOCK_SECONDS=0`.
**Only relative ordering matters** — absolute bpb will differ from 8×H100.

```
Phase 1: Baseline + TTT improvements (Day 1 morning)
═══════════════════════════════════════════════════════
T1.0  Baseline SOTA stack (no TTT)              → reference val_loss
T1.1  TTT with SGD/3ep on 100 val chunks        → TTT reference
T1.2  TTT with AdamW/8ep on 100 val chunks      → measure improvement
T1.3  TTT with LoRA-8/AdamW/8ep                 → measure improvement + stability
T1.4  TTT with 64K chunks + AdamW/8ep           → measure chunk size effect

Phase 2: Quantization + Depth (Day 1 afternoon)
═══════════════════════════════════════════════════════
T2.0  11L with int4 MLP QAT (keep int6 attn)    → measure int4 quality loss
T2.1  12L with int4 MLP QAT                     → is 12L-int4 > 11L-int6?
T2.2  13L with int4 MLP QAT                     → is 13L-int4 > 11L-int6?
T2.3  11L with full GPTQ (Hessian calibration)  → measure GPTQ vs GPTQ-lite

Phase 3: Mamba Hybrid (Day 2 morning)
═══════════════════════════════════════════════════════
T3.0  4 Mamba + 7 Transformer (11 total)        → Mamba viability check
T3.1  4 Mamba + 9 Transformer (13 total)        → full hybrid
T3.2  2 Mamba + 9 Transformer (11 total)        → lighter hybrid

Phase 4: Micro-optimizations (Day 2 afternoon)
═══════════════════════════════════════════════════════
T4.0  SwiGLU² activation (mlp_mult=2, gated)    → activation comparison
T4.1  GEGLU² activation                         → activation comparison
T4.2  LeakyReLU(0.3)² and (0.7)²                → alpha sweep
T4.3  XSA last 6 layers (vs 4)                  → marginal XSA gain
T4.4  VE on layers 7-10 (vs 9-10)               → marginal VE gain
T4.5  BigramHash(4096)                           → hash bucket count
T4.6  TrigramHash addition                       → n-gram extension
T4.7  seq_len=3072 training                      → long context tradeoff

Phase 5: Combine winners (Day 3)
═══════════════════════════════════════════════════════
T5.0  Best TTT + best quantization + best micro  → combined run, 2000 steps
T5.1  Add Mamba if T3.x was positive             → full stack, 2000 steps
T5.2  Full 10-min simulated run on 5090          → extrapolate to 8×H100
```

### Decision Logic After Each Phase

```
After Phase 1:
  IF T1.2 or T1.3 beats T1.1 by > 0.001 bpb → adopt enhanced TTT
  BEST_TTT = argmin(T1.1, T1.2, T1.3, T1.4)

After Phase 2:
  IF T2.1 or T2.2 beats T1.0 → adopt int4 MLP + deeper model
  IF T2.0 degrades by > 0.005 vs T1.0 → abandon int4, keep int6
  BEST_DEPTH = argmin(T2.0, T2.1, T2.2) if beats T1.0, else T1.0

After Phase 3:
  IF T3.x within 0.005 of T1.0 → Mamba hybrid is viable, tune further
  IF T3.x > 0.01 worse → abandon Mamba, stick with pure transformer

After Phase 4:
  Collect all improvements that individually beat the control by > 0.0005.
  Stack them (they should be roughly additive since they're orthogonal).

Phase 5: combine everything and validate on a longer run.
```

---

## Implementation Priority

### Must Build (before any testing)

1. **5090-compatible base script** — strip DDP, replace flash_attn_3, reduce batch size
2. **Metrics logging harness** — CSV output for val_loss, val_bpb, step_time, artifact_size per run
3. **TTT variants** — AdamW path, LoRA path, configurable epochs/chunks in existing TTT code

### Build If Phase Confirms

4. **Int4 QAT** — `quantize_int4_per_row`, STE fake-quant, mixed int4/int6 serialization
5. **MambaBlock** — either from `mamba_ssm` package or hand-written selective scan
6. **Gated MLP** — SwiGLU/GEGLU activation variant with gate projection
7. **GPTQ with Hessian** — calibration pass + Hessian-weighted optimal rounding

### Do Not Build (low expected value)

- ~~RWKV~~ — Full RWKV replacement is too risky given leaderboard evolution
- ~~Kronecker factorization~~ — Extreme compression not needed when int4 MLP works
- ~~Layer recurrence/looping~~ — Explicitly tested and failed (-0.051 bpb in prior submission)
- ~~Custom CUDA kernels~~ — Time-consuming, only needed if Mamba hybrid wins and needs speed
- ~~Reservoir / echo state~~ — Creative but unlikely to beat SOTA
- ~~Novel tokenizer~~ — Rules make this very scrutinized, not worth the risk

---

## Submission Preparation (Once We Have a Winning Config)

### Validation Protocol

1. Run 3 seeds (1337, 42, 2025) on 8×H100 with full 600s training + eval
2. Compute mean ± std of val_bpb
3. Verify `mean_bpb < 1.1194 - 0.005/ln(2) × tokens_per_byte` (≈ 1.1165)
4. Verify `p < 0.01` via one-sided t-test against 1.1194
5. Verify all 3 artifacts < 16,000,000 bytes
6. Verify eval time < 600s for each seed

### Submission Structure

```
records/track_10min_16mb/2026-MM-DD_OurSubmissionName/
├── train_gpt.py          # Self-contained training + eval script
├── README.md             # Architecture, techniques, ablations
├── submission.json       # {"name": "...", "val_bpb": X.XXXX, ...}
├── train_seed1337.log    # Full training log
├── train_seed42.log
└── train_seed2025.log
```

---

## Risk Assessment

| Vector | Expected gain | Probability of working | Risk if it fails |
|---|---|---|---|
| Enhanced TTT | -0.003 to -0.005 | **80%** — TTT is clearly underoptimized | Low — fall back to current TTT |
| Int4 MLP + depth | -0.002 to -0.004 | **60%** — int4 quality is uncertain | Low — keep 11L int6 |
| Mamba hybrid | -0.001 to -0.003 | **30%** — novel, untested at this scale | Medium — wasted implementation time |
| Micro-optimizations | -0.001 to -0.003 combined | **70%** — incremental, low-risk | None — each is independent |

**Expected total if all work:** -0.007 to -0.015 bpb → target 1.105 to 1.113 bpb.
**Conservative (TTT + micro only):** -0.004 to -0.008 → target 1.112 to 1.116 bpb.
**Minimum viable (just beats threshold):** -0.003 → target 1.1165 bpb.

The conservative path (Vector 1 + micro-optimizations) has **~70% probability** of producing
a submittable result. Adding Vector 2 raises the ceiling significantly.

---

## Timeline

| Day | Focus | Deliverable |
|---|---|---|
| **Day 1** | Setup + Phase 1-2 | 5090-compatible script, TTT comparison, int4 test |
| **Day 2** | Phase 3-4 | Mamba viability, micro-optimization sweep |
| **Day 3** | Phase 5 | Combined stack, 2000-step validation |
| **Day 4** | Refinement | Tune hyperparameters of winning components |
| **Day 5** | H100 submission run | 3-seed validation on 8×H100 |
| **Day 6** | PR preparation | README, submission.json, clean logs |

---

*Plan v2 — March 25, 2026. Supersedes original RWKV+Kronecker plan.*
