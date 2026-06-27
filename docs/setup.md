# Flow2Surf Setup

Flow2Surf runs on a CUDA GPU. Jittor compiles CUDA kernels at runtime, so the
machine needs a matching CUDA toolkit and a C++ compiler (g++). Tested on an
RTX 4090, Ubuntu 22.04.

## 1. Clone and create the environment

```bash
git clone git@git.tsinghua.edu.cn:srw24/flow2surf.git
cd flow2surf
conda env create -f environment.yml
conda activate jittor
```

## 2. Set up CUDA for Jittor

```bash
python -m jittor_utils.install_cuda
python -m jittor.test.test_cuda      # should finish and report CUDA support
```

## 3. Data

Place or symlink the dataset under `data/`:

```text
data/
  dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
  dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

## 4. Train

```bash
python train.py --config configs/default.yaml      # writes logs/train_<time>.log
```

Use `tmux` (or similar) for long runs; resume with `--resume <checkpoint.pkl>`.

## 5. Predict and package

```bash
python predict.py --config configs/default.yaml --model <checkpoint.pkl> --out results
bash scripts/make_submission.sh <checkpoint.pkl> [config.yaml] [name]
```
