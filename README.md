# M2O: Many-to-One Self-Distillation for MeanFlow

**When More Calls Help: Dataset-Free Low-Compute Self-Distillation for MeanFlow**  
Zhicheng Jiang (MIT, 2077 AI), Ningshan Ma (MIT)  
Contact: jzc_2007@mit.edu

M2O distills a better multi-step sampler of an iMF checkpoint into its one-step
mode. Adaptation uses random noise, class labels, a frozen self-teacher and a
frozen VAE/ConvNeXt feature map, without loading real training images.

[Code](https://github.com/jzc-2007/M2O) · [Checkpoints](https://github.com/jzc-2007/M2O/releases/tag/v1.0.0)

This release provides the L/2 training and evaluation recipe in `configs/l2.yml`.
At fixed CFG 5, ImageNet-256 FID-50K improves from 2.31 to 1.72 in 0.93 H200
GPU-hours of adaptation. The separately calibrated one-step baseline is 1.88.
The fresh A100 L/2 reproduction and its hardware-specific timing are recorded
separately in `results/l2_a100_results.json`.

The additional XL/2 result is **1.62 FID-50K**; its checkpoint is listed in
[CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Installation

The validated environment uses Linux, Python 3.10, CUDA 12 and an 80GB GPU.
Torch is CPU-only; JAX performs training. No ImageNet training set is needed.

```bash
git clone https://github.com/jzc-2007/M2O.git
cd M2O
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-h200.txt
python -m pip install -r requirements-eval.txt
python -m pip install -e .
git clone https://github.com/Lyy-iiis/imeanflow vendor/imeanflow
git -C vendor/imeanflow checkout bf60cd7cb653f6628e59d48034b333c5eba445e2
python scripts/download_assets.py --model L
```

Assets are revision-pinned and SHA256-verified. Set `HF_ENDPOINT` if a mirror
is needed. Pretrained weights retain their upstream licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md).

## L/2 training and evaluation

Choose an authorized idle GPU and inspect `nvidia-smi`, compute processes,
`tmux ls`, training processes and disk space before launching. Set
`CUDA_VISIBLE_DEVICES` to that device and disable JAX preallocation.
Run commands from the repository root.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python train_imeanflow_b2_convnext_kid.py --config configs/l2.yml --workdir runs/l2 --max-steps 16000 --resume auto
python eval_imeanflow_b2_student.py --config configs/l2.yml --evaluation-config configs/l2.yml --checkpoint-dir runs/l2/checkpoint_16000 --workdir runs/l2_eval50k --samples 50000 --batch-size 4 --seed 20260831 --id-offset 700000 --noise-scale 1 --output-scale 1 --params-source raw --inception-backend cpu --feature-export-dtype float32
```

The historical filenames include `b2`, but these entry points support the L/2
endpoint architecture. The four-step teacher and one-step student use CFG 5,
interval [0.40, 0.60], and unit noise/output scales. Adaptation uses batch 4,
16K updates and learning rate 1e-6; only the configured u-head/readout leaves
are updated. The backbone, teacher, VAE and feature encoder remain frozen.
Training seed is 42 and generated-pair seed is 20260831.

`--resume auto` restores the complete optimizer, EMA and running RMS state.
Keep a run's saved configuration unchanged. For baseline evaluation, replace
`--checkpoint-dir ...` with `--warmstart-initial`. The calibrated 1.88 baseline
uses the separate guidance setting described in the paper. Teacher evaluation
uses `eval_imeanflow_b2_teacher.py`.

FID-50K uses balanced labels and the seed-20260831 / offset-700000 bank,
float32 saved Inception features, and ImageNet-256 real-reference statistics.
The 5K screening bank has offset 500000. Pretraining, selection and final
evaluation are excluded from the reported adaptation cost.

## Checkpoints, validation and manuscript

See [checkpoint formats](docs/CHECKPOINTS.md) and [validation](docs/VALIDATION.md).
Model weights are available from the [v1.0.0 release](https://github.com/jzc-2007/M2O/releases/tag/v1.0.0).
Download and verify a complete archive with:

```bash
python scripts/download_checkpoint.py --model L
# Optional additional checkpoint:
python scripts/download_checkpoint.py --model XL
```

The downloader resumes interrupted transfers, verifies every shard and the
merged archive, and outputs one checkpoint archive per model. The L/2 scientific trainer and evaluator retain their
archived implementation; provenance is recorded in `docs/CODE_PROVENANCE.json`.

```bash
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONPATH=src:. python -m pytest tests -q
python scripts/build_paper.py
```

The manuscript source in `paper/` retains the ICLR template and named authors.
The additional result changes only one row of Table 4; see
[arXiv packaging](docs/ARXIV_UPLOAD.md). The arXiv submission is handled separately by the authors.
