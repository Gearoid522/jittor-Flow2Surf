# Flow2Surf Setup

## Requirements

- Linux with Conda, `g++`, and `zip`
- CUDA-capable NVIDIA GPU and compatible driver/toolkit

Jittor compiles CUDA kernels at runtime. The project has been tested on Ubuntu
22.04 with an RTX 4090.

## Installation

```bash
git clone git@git.tsinghua.edu.cn:srw24/flow2surf.git
cd flow2surf
conda env create -f environment.yaml
conda activate flow2surf
```

## CUDA Verification

```bash
python -m jittor_utils.install_cuda
python -m jittor.test.test_cuda
```

## Dataset

Place or symlink the datasets under `data/`:

```text
data/
  A/dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
  A/dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
  B/dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
  B/dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

Use `datasets/A.yaml` or `datasets/B.yaml` to select the corresponding exact
split.
