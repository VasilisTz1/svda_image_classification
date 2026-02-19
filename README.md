# svda_image_classification

SVDA-based transformer image classification experiments and profiling artifacts for:
- `FashionMNIST`
- `CIFAR-10`
- `CIFAR-100`
- `ImageNet-100`

This repository contains training notebooks, checkpoints, profiling scripts, and analysis artifacts.

## What is included

Each dataset folder contains:
- Training notebook (`svda_*.ipynb`)
- Baseline and SVDA checkpoints (`baseline_results_*.pth`, `svda_results_*.pth`)
- Inference profiling script (`profile_inference_*.py`)
- Inference profile output JSON (`*_inference_profile.json`)
- Metric logs (`logs_baseline_*`, `logs_svda_*`)
- Epoch summary metrics (`metric_summary_over_epochs.csv`)
- Model summaries (`model_summary_baseline.txt`, `model_summary_svda.txt`)
- Analysis notebooks (`*_graphs.ipynb`, `*graphs3.ipynb`)

Top-level dataset folders:
- `FashionMNIST/`
- `cifar10/`
- `cifar100/`
- `imagenet100/`

## Quick start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch torchvision numpy ptflops jupyter matplotlib pandas
```

## Run inference profiling

Important: run each script from inside its dataset folder because checkpoint paths are relative.

### FashionMNIST

```bash
cd FashionMNIST
python profile_inference_fashionmnist.py
```

### CIFAR-10

```bash
cd cifar10
python profile_inference_cifar10.py
```

### CIFAR-100

```bash
cd cifar100
python profile_inference_cifar100.py
```

### ImageNet-100

```bash
cd imagenet100
python profile_inference_imagenet100.py
```

Each profiling script compares baseline vs SVDA and reports:
- MACs
- Parameter count
- Inference latency stats (`mean`, `p50`, `p90`, `p99`)
- Peak inference memory on CUDA

## Training notebooks

Primary training notebooks:
- `FashionMNIST/svda_FashionMNIST.ipynb`
- `cifar10/svda_cifar10.ipynb`
- `cifar100/svda_cifar100.ipynb`
- `imagenet100/svda_imagenet100.ipynb`

To run locally:

```bash
jupyter notebook
```

Then open the dataset notebook above and run cells in order.

## Analyze results

Open dataset notebooks to reproduce plots and comparisons:
- `FashionMNIST/FashionMNIST_graphs.ipynb`
- `FashionMNIST/svda_FashionMNIST_graphs3.ipynb`
- `cifar10/cifar10_graphs.ipynb`
- `cifar10/svda_cifar10_graphs3.ipynb`
- `cifar100/cifar100_graphs.ipynb`
- `cifar100/svda_cifar100_graphs3.ipynb`
- `imagenet100/imagenet100_graphs.ipynb`
- `imagenet100/svda_imagenet100_graphs3.ipynb`

## Notes

- The profiling scripts auto-select device in this order: CUDA, MPS, CPU.
- Checkpoint files are required and are expected to be present in the same folder as the profiling script.
