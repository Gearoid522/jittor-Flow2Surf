# Flow2Surf: Flow-Guided Point Cloud Denoising

Flow2Surf is a Jittor-based project for denoising noisy point clouds. It
predicts a per-point residual field on local patches and aggregates overlapping
patch predictions to refine the full-cloud output.

## Features

- Patch-based denoising for large point clouds.
- Dynamic kNN geometry embedding with EdgeConv-style local features.
- Time-conditioned attention blocks with AdaLN-Zero modulation.
- Residual, Chamfer, and optional density-aware Chamfer training losses.

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate jittor
```

## Data

Expected layout:

```text
data/
  dataset_train/
    shapenet/<synset_id>/<model_id>/models/model_normalized.obj
  dataset_test_noisy/
    shapenet/<synset_id>/<model_id>/noisy.npy
```

## Training

Train with the default config:

```bash
python train.py --config configs/default.yaml
```

Resume from a checkpoint:

```bash
python train.py --config configs/default.yaml --resume <checkpoint.pkl>
```

Each run writes logs and timestamped checkpoints:

```text
logs/train_<time>.log
<checkpoint_dir>_<time>/epoch_*.pkl
<checkpoint_dir>_<time>/best_info.json
<checkpoint_dir>_<time>/best_mini_info.json
```

Training and inference options are documented inline in
[configs/default.yaml](configs/default.yaml).

## Prediction

Run inference on the test set. `--model` is required:

```bash
python predict.py \
  --config configs/default.yaml \
  --model <checkpoint.pkl> \
  --out results
```

Package a submission zip. Only the checkpoint is required; config and name are
optional:

```bash
bash scripts/make_submission.sh <checkpoint.pkl> [config.yaml] [name]
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

Score a checkpoint on the held-out validation split (config-driven, reproducible):

```bash
python eval.py --config configs/default.yaml --model <checkpoint.pkl>
```

This writes `eval_<config>_<time>.json` with the setting and per-mesh scores.

## Project Structure

```text
train.py                    training entrypoint
predict.py                  inference and submission generation
eval.py                     validation-split scorer (per-mesh JSON)
src/dataset.py              data loading, sampling, noise, patch extraction
src/embedding.py            dynamic kNN geometry encoder
src/model.py                Flow2Surf network
configs/                    training and inference configs
scripts/make_submission.sh  inference and zip packaging helper
docs/                       setup and design notes
environment.yml             conda environment spec
```
