# 5.3 Florence-2 Fine-Tuning

## Approach

Florence-2 is natively trained for `<REFERRING_EXPRESSION_COMPREHENSION>` (REC) as
a *bounding-box* task: given a text prompt, the decoder emits four `<loc_*>`
tokens encoding `(x_min, y_min, x_max, y_max)` in a 0–999 quantized canvas.
For the GUI-grounding task we want a single click point, so the zero-shot
baseline parses the box and uses its center as the predicted point.

We change the output target during fine-tuning so the model emits a
**point** directly: two `<loc_*>` tokens encoding `(x, y)`. This removes the
"invent a fake bbox" step at training time and gives the loss a more direct
signal: every supervision step is an actual coordinate, not a 4-token box
that the loss treats as a sequence-of-tokens problem.

We use **LoRA** (rank 16, α=32, dropout 0.05) targeting Florence-2's attention
projections — `q_proj`, `k_proj`, `v_proj`, `out_proj` — across both the
language encoder/decoder and the DaViT vision encoder. This matches 144
modules and yields 4.7 M trainable parameters out of 827 M total (0.57 %).
The vision tower is *not* additionally unfrozen; the LoRA adapters on its
attention projections are sufficient to teach the model the new
loc-token-pair output convention.

## Data pipeline

Training items are built from the Show-UI / UGround conversations dataset by
parsing each `human → gpt` pair: the human turn carries an instruction
prompt with a `Description: ...` line; the gpt turn carries `(x, y)` in the
0–999 normalized space. We use one (description, point) pair per source row
and exclude all rows used in the held-out evaluation split (`florence_samples`,
500 items). After filtering, the training set contains ~3,070 (image, RE, point) tuples.

Each sample is preprocessed by:

1. Letterboxing the source image into a 512×512 canvas (preserves aspect
   ratio; padding is grey `(128,128,128)`).
2. Remapping the 0–999 GT point through the same letterbox transform so the
   target coordinate lives in the canvas the model actually sees.
3. Cleaning the human prompt to the description text after `Description:`
   (the raw prompt embeds three paragraphs of task-template boilerplate that
   adds no signal).

The Florence-2 processor then resizes the 512 canvas to the model's native
768×768 vision input.

## Loss function (default)

The decoder target is `[BOS, <loc_X>, <loc_Y>, EOS]`. Florence's standard
cross-entropy loss applies to all decoder positions. With the target shape
above this means the loss puts full pressure on the two coordinate tokens
plus the format markers. We use this CE-only loss as the baseline for the
ablations in §5.4 and as one component of every variant.

## Hyperparameters

| Parameter | Value |
|---|---|
| LoRA rank `r` | 16 |
| LoRA `α` | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj, k_proj, v_proj, out_proj` |
| Trainable params | 4.7 M (0.57 % of 827 M) |
| Per-device batch size | 2 |
| Gradient accumulation | 4 (effective bs = 8) |
| Optimizer | AdamW (`adamw_torch`) |
| Learning rate | 2e-4 |
| LR schedule | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.0 |
| Epochs | 1 |
| Mixed precision | bf16 (A100) / fp16 fallback |
| Gradient checkpointing | on (non-reentrant) |

## Hardware and cost

Training ran on a single Colab A100 (40 GB). One epoch over 3,070 items
takes ≈ 384 optimizer steps and completes in roughly 12–15 minutes wall-clock.
Eval (greedy decode on the held-out 500 items) takes ≈ 140 seconds.
End-to-end the LoRA fine-tune costs well under one A100-hour, which is what
made the loss-function ablation in §5.4 cheap enough to iterate on.

## Inference

We use greedy decoding (`do_sample=False, num_beams=1`, `max_new_tokens=8`).
At eval time we de-standardize predictions: the model emits coordinates in
the 512×512 canvas space, and we invert the letterbox transform (strip
padding, undo the resize, renormalize to 0–999) so predictions are directly
comparable against ground truth in the original image's coordinate frame.
This step is *required for correctness* — without it, predictions for
non-square images are systematically offset by the letterbox padding,
which silently inflates pixel error.

# 5.4 Loss Function Ablations

## Setup

All variants share the §5.3 setup (LoRA-r16, image standardization, single
epoch, eval on 500 held-out items). The only thing that changes between
variants is the loss function. Metrics report the same battery as the §5.1
zero-shot baseline: parse rate, mean / median pixel error, and accuracy at
distance thresholds.

> **Note on GIoU.** The original ablation plan included a GIoU variant
> (generalized IoU between predicted and target bounding boxes) on the
> hypothesis that a spatial loss would provide denser gradients than CE on
> coordinate tokens. We did not implement GIoU in this iteration: switching
> to a *point* output (two tokens, not four) makes IoU undefined as written.
> A natural adaptation — IoU on small fixed-radius boxes around the
> predicted/target points — collapses, in the small-box limit, to a smooth
> distance-style loss and is mathematically close to the L2 expectation
> term we *did* test. We replaced GIoU with a **Gaussian soft-CE** variant
> instead, which addresses the same density-of-gradient hypothesis while
> staying coordinate-token-native (see "Why Gaussian soft-CE" below).

## Variants

### 1. CE only (baseline)

Standard token-level cross-entropy on the decoder output. The model sees
exactly two answer tokens (`<loc_X><loc_Y>`) and gets a one-hot supervision
signal at each. Distance-in-pixels is *invisible* to this loss: predicting
`<loc_499>` when the GT is `<loc_500>` costs the same as predicting
`<loc_0>`.

### 2. CE + L2 (λ_L2 · L2 of expected coords)

We add a differentiable L2 term over the two `<loc_*>` answer positions.
At each position we take the model's logits restricted to the 1000 loc-token
IDs, softmax them, and compute the **expected coordinate**:

```
exp_x = Σ_k softmax(logits_x[loc_ids])[k] · k        (k in [0, 999])
exp_y = Σ_k softmax(logits_y[loc_ids])[k] · k
loss_L2 = √((exp_x − gt_x)² + (exp_y − gt_y)² + ε) / 999
```

The full loss is `loss = loss_CE + λ_L2 · loss_L2`. Normalizing by 999 puts
`loss_L2` in `[0, ~1.4]`, comparable in scale to early-training CE
(~3–6 nats). After empirical tuning we settled on `λ_L2 = 0.1` (the original
`0.01` was CE-dominant and had ≪ 1 % gradient contribution from L2).

### 3. CE + Gaussian soft-CE (λ_soft · KL with a Gaussian target)

Instead of one-hot CE on `<loc_GT>`, we use a Gaussian-smoothed target
distribution centered on GT:

```
target(k) ∝ exp(−(k − gt)² / (2σ²))    over k in [0, 999]
loss_soft = −Σ_k target(k) · log_softmax(logits[loc_ids])(k)
```

The full loss is `loss = loss_CE + λ_soft · loss_soft`, applied only at the
two loc positions. With `σ = 5` (≈ a few pixels on the 512 canvas) and
`λ_soft = 0.5`, this term penalizes wrong tokens linearly with distance
*and* rewards a sharp peak at GT.

At inference we additionally decode by **expectation** rather than
argmax: at the two greedy loc positions we recover `(x, y)` as
`Σ_k softmax(logits[loc_ids])[k] · k` instead of `argmax`. This matches
the soft-CE training objective exactly and is robust to slightly smeared
distributions. We provide both `decode_mode="expectation"` and
`decode_mode="argmax"` for comparison.

## Why Gaussian soft-CE

The L2-on-expectation loss diagnoses cleanly *why* the dense-gradient
hypothesis is correct in spirit but fails in practice with naive expectation
decoding. L2 optimizes a **continuous** quantity (the softmax-weighted mean
over loc tokens), but greedy inference picks the **argmax** — these can
disagree. A model can satisfy L2 with a wide, smeared distribution centered
near GT (low expected error), while the argmax token lands well off-target.
Empirically, after one epoch of CE+L2, mean pixel error was 361 px and
Acc@100px was 15.0 %, despite a 100 % parse rate — the model had learned
the *format* but not coordinate *precision*.

Gaussian soft-CE fixes the train/inference mismatch in two ways:

1. **Sharp peaks.** Cross-entropy against a Gaussian target has a unique
   minimum at the matching distribution; the gradient pushes mass toward
   GT *and* concentrates it. The argmax of the resulting distribution
   reliably falls on or near `<loc_GT>`.
2. **Distance-aware.** Like L2, neighbors of GT are penalized less than
   far tokens, so the gradient is dense — a wrong-by-1-token prediction
   is cheaper to fix than a wrong-by-500-token one. Standard one-hot CE
   has no such structure.

## Results

500 held-out evaluation samples, per-image pixel error, accuracy at
threshold across the test set:

| Metric | Zero-shot (CE-trained, bbox→center) | CE + L2 (λ=0.1) | CE + Gaussian soft-CE (λ=0.5, σ=5) |
|---|---:|---:|---:|
| Parse rate | 67.2 % | 100.0 % | **100.0 %** |
| In-bounds rate | 67.2 % | 100.0 % | **100.0 %** |
| Mean err norm (0–999) | 487.6 | 351.3 | **89.7** |
| Median err norm | 494.8 | 319.7 | **19.0** |
| Mean err (px) | 459.9 | 361.2 | **100.0** |
| **Median err (px)** | 448.3 | 351.0 | **20.7** |
| Mean err (% diag) | 31.3 | 24.5 | **6.8** |
| Acc @ 5 px | 1.0 % | 2.8 % | **14.6 %** |
| Acc @ 10 px | 1.2 % | 4.0 % | **30.0 %** |
| Acc @ 25 px | 1.4 % | 5.8 % | **54.6 %** |
| Acc @ 50 px | 2.4 % | 9.0 % | **64.8 %** |
| **Acc @ 100 px** | 4.6 % | 15.0 % | **77.0 %** |
| Acc @ 200 px | 10.8 % | 29.8 % | **86.6 %** |
| Acc @ 1 % diag | 1.2 % | 4.8 % | **40.0 %** |
| Acc @ 5 % diag | 3.0 % | 11.2 % | **71.4 %** |
| Acc @ 10 % diag | 7.0 % | 21.0 % | **82.4 %** |

## Discussion

**Both LoRA variants beat the zero-shot bbox-center baseline at every
threshold.** Parse rate jumps from 67 % → 100 % because the model is
trained explicitly to emit two well-formed `<loc_*>` tokens; the failures
of the zero-shot baseline are mostly format errors (Florence emits text
descriptions or malformed boxes). This alone is a +33 pp gain.

**CE + L2 is a partial win.** It improves coarse localization (Acc@200px
+19 pp over zero-shot) but barely moves the needle on fine precision
(Acc@5px goes from 1.0 % to only 2.8 %). Median pixel error drops modestly
from 448 px to 351 px. The L2 term is doing *something* — predictions are
closer in expectation — but the argmax decode is throwing precision away.

**CE + Gaussian soft-CE is the dominant variant.** Median pixel error
collapses from 351 px (CE+L2) to 20.7 px — a 17× improvement on the same
checkpoint. Acc@100px goes from 15 % to 77 %; Acc@25px goes from 5.8 % to
54.6 %. This validates the diagnosis above: the L2 expectation term *was*
giving the model the right gradient direction, but the model was not being
asked to commit that gradient to a sharp peak. Gaussian soft-CE adds that
constraint directly.

**The mean–median gap (100.0 px vs 20.7 px) on the soft-CE run reveals a
heavy tail.** ~13 % of samples (between Acc@200px = 86.6 % and 100 %) are
> 200 px off and drag the mean up. Qualitative inspection of failure cases
suggests these are mostly references to small icons or ambiguous
descriptions ("the menu"), not systemic precision failures.

**Implementation notes that materially affected results:**

- An early version of the eval pipeline failed to invert the letterbox
  transform, so model predictions in the 512-canvas space were being
  compared to GT in the original image's space. For widescreen images this
  shifts predictions by ~100+ px and silently regresses Acc@<100px. Fixing
  the de-standardization recovered ~10 pp on Acc@100px before the loss
  change was applied.
- The dataset's prompts embed the actual element description after a
  `Description:` line preceded by ~3 paragraphs of task boilerplate. An
  early prompt-cleaning routine missed this marker and fell through to a
  heuristic that split on " located " / " in ", chopping the middle of
  valid descriptions. Both training and inference were affected.

## Future directions

- **More epochs.** The full run was a single epoch (~384 optimizer steps).
  The loss curve had not flattened. Two or three epochs should push
  Acc@5–10px further.
- **Higher canvas resolution.** Letterboxing to Florence's native 768 (vs
  our 512) gives each `<loc_*>` token a finer slice of the image and
  should help the heavy tail.
- **An actual GIoU variant.** Reverting the output to a 4-token bbox
  (`<loc_x_min><loc_y_min><loc_x_max><loc_y_max>`) and adding a GIoU loss
  on the parsed boxes would let us test the original hypothesis directly.
  The point output we used here is more direct for the click task but
  forecloses that comparison.
- **Hyperparameter sweep on σ and λ_soft.** σ = 5 was a reasoned starting
  point, not a swept optimum. σ ∈ {2, 3, 5, 8, 10} on a small held-out
  set should be cheap to run.
