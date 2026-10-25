# G-Loss under batch-size constraints

Reproduction and stress-test of **G-Loss** — [paper](https://arxiv.org/abs/2604.25853) · [original code](https://github.com/saditya13/G-Loss-LoG)

## What this is

While reproducing G-Loss on a 4GB GPU, I found a checkpoint-loading bug in the
public code, fixed it, reproduced the paper's baseline within normal seed
variance, then tested whether G-Loss's benefit survives at a much smaller
batch size than the paper's default. Short answer: it doesn't clearly hold up,
and the likely reason points at a specific, testable mechanism.

## Bug found and fixed

`main.py` saves the best-validation-F1 checkpoint to `checkpoint.pth` during
training, but at test time it evaluates whatever state `train_model` returns
— the final epoch (the early-stopping point), not the best one. On my first
run this cost ~1.4 points of test F1. Patch: `patches/checkpoint-fix-and-seed-flag.diff`
(also adds a `--seed` CLI flag, hardcoded to 42 in the original).

## Setup

Hardware forced batch_size=16 (DistilBERT + Adam on a 4GB Quadro P2000),
vs. the paper's default of 128. Dataset: R8. Model: distilbert-base-uncased.
3 seeds (42, 7, 123) — not the paper's full seed count, GPU-time constrained.

## Results

| Loss | Test Acc | Test F1 |
|---|---|---|
| CE | 97.53 ± 0.17% | 93.54 ± 0.21% |
| G-Loss (sigmafn=root) | 97.32 ± 0.48% | 92.91 ± 0.74% |
| *Paper (CE+G-Loss-SQRT, batch=128)* | *~97.46 ± 0.17%* | *~93.62 ± 0.69%* |

Per-seed breakdown (CE − G-Loss, F1):

| seed | CE F1 | G-Loss F1 | Δ |
|---|---|---|---|
| 42 | 93.70 | 92.35 | +1.35 |
| 7 | 93.61 | 92.63 | +0.98 |
| 123 | 93.30 | **93.75** | **−0.45** |

## Interpretation

G-Loss loses on 2/3 seeds but wins on the third — not uniformly worse, just
noisier, and a paired comparison across seeds doesn't clear significance at
n=3. What *is* consistent: G-Loss's variance (±0.74 F1) is over 3x CE's
(±0.21 F1).

**Hypothesis:** `predict_lpa` builds its label-propagation graph over the
mini-batch itself, not the full dataset. At batch_size=16 vs. the paper's
128, G-Loss gets roughly 8x fewer points to propagate labels across — which
would explain both the accuracy gap and the elevated variance, independent
of whether the method is sound at full batch size.

**Open question, not yet tested:** does G-Loss's benefit return at larger
batch sizes, and is there a way to decouple graph size from the
memory-bound training batch size (e.g. a cached embedding pool rather than
only the current batch)? Gradient accumulation doesn't work here — the
method backprops through a `torch.linalg.solve` coupling every point in the
batch, so it isn't decomposable into independent per-example terms the way
CE is.

## Reproduce

```bash
git clone https://github.com/saditya13/G-Loss-LoG.git
cd G-Loss-LoG
git apply /path/to/patches/checkpoint-fix-and-seed-flag.diff
cd "Combined with CE"
# then run run_remaining_seeds.sh, or main.py directly per its own README
```
