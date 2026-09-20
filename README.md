# Flow2Surf: Continuous-Flow Point Cloud Denoising

Flow2Surf is a Jittor implementation of continuous-flow point cloud denoising.
It predicts alpha-conditioned velocities on local patches and aggregates them
while integrating from noisy to clean.

Flow2Surf was developed for the 3D Point Cloud Denoising track of the Sixth
Jittor Artificial Intelligence Challenge. It ranked first on both the A and B
leaderboards, scoring 85.16 and 82.10, respectively.

## Features

- Continuous-alpha velocity training independent of inference discretization.
- Shared XYZ neighborhoods combining content and HPE positional edges through
  an exact factorized projection.
- Alpha-conditioned AdaLN-Zero transformer trunk with offset attention and
  SwiGLU feed-forward networks.
- Center-weighted fusion of overlapping patch velocities at each reverse-flow
  integration step.
- Rotationally invariant Gaussian and Laplace noise with matched covariance.

## Installation

See [docs/setup.md](docs/setup.md) for environment and dataset setup. Create the
single maintained environment with:

```bash
conda env create -f environment.yaml
conda activate flow2surf
```

## Data

Dataset descriptors and manifests define exact splits. The included descriptors
expect:

```text
data/
  A/
    dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
    dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
  B/
    dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
    dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

`datasets/A.yaml` preserves the original A split. `datasets/B.yaml` uses the
official B manifests.

## Training

Train with the default config:

```bash
python train.py --config configs/default.yaml
```

Select B without changing the model config:

```bash
python train.py --config configs/default.yaml --dataset datasets/B.yaml
```

Resume from a checkpoint:

```bash
python train.py --config configs/default.yaml --resume <checkpoint.pkl>
```

Each run writes logs and timestamped checkpoints:

```text
logs/train_<time>.log
<checkpoint_dir>_<time>/epoch_*.pkl
<checkpoint_dir>_<time>/epoch_*_optim.pkl
<checkpoint_dir>_<time>/epoch_*_ema.pkl     # when EMA is enabled
<checkpoint_dir>_<time>/best_ep*.pkl
<checkpoint_dir>_<time>/best_mini_ep*.pkl
<checkpoint_dir>_<time>/train_state.json
```

When EMA is enabled, validation, mini-validation, and best checkpoints use
averaged weights. Periodic resume checkpoints pair the raw model and optimizer
with the corresponding EMA state.

Training and inference options are documented inline in
[configs/default.yaml](configs/default.yaml).

## Prediction

Run inference on the test set. `--model` is required:

```bash
python predict.py \
  --config configs/default.yaml \
  --dataset datasets/A.yaml \
  --model <checkpoint.pkl> \
  --out results
```

Package a submission zip. The config defaults to `configs/default.yaml` and
must match the checkpoint architecture; the output name is optional:

```bash
bash scripts/make_submission.sh <checkpoint.pkl> [config.yaml] [name] [dataset.yaml]
```

Predictions are written as:

```text
<out>/shapenet/<synset_id>/<model_id>/denoised.npy
```

The submission helper also creates:

```text
result_<name>.zip
```

## Evaluation

Score a checkpoint on the fixed family-scale validation grid:

```bash
python eval.py \
  --config configs/default.yaml \
  --dataset datasets/A.yaml \
  --model <checkpoint.pkl> \
  --meshes 50
```

This writes `eval_<config>_<time>.json` with protocol, architecture, inference,
and aggregate metrics by family and noise level.

## Project Structure

```text
train.py                    training entrypoint
predict.py                  prediction entrypoint
eval.py                     fixed-grid validation scorer
flow2surf/flow.py           conditional-flow mathematics
flow2surf/dataset.py        mesh sampling, noise, and patch datasets
flow2surf/inference.py      shared patch-wise reverse integration
flow2surf/evaluation.py     full-cloud metrics and mini-validation
flow2surf/runtime.py        logging, reproducibility, and run metadata
flow2surf/models/           encoder, transformer, and velocity decoder
flow2surf/training/         losses, optimization, EMA, and checkpoints
configs/                    training and inference configs
datasets/                   A/B descriptors and exact split manifests
scripts/make_submission.sh  inference and zip packaging helper
docs/                       setup and design notes
environment.yaml            conda environment spec
```
