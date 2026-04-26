# svda_image_classification

Experiments comparing baseline Vision Transformer attention with SVDA attention
for image classification datasets.

The repository currently contains single-file training and reporting scripts for:

- FashionMNIST
- CIFAR-10
- CIFAR-100
- ImageNet-100
- ImageNet-100 with a pretrained ViT backbone and SVDA applied to the final
  transformer blocks

Each script trains baseline and SVDA variants across the configured seeds,
computes interpretability and efficiency metrics, saves checkpoints and figures
under a local run directory, and writes summary CSV files.

## Repository Layout

```text
.
├── fashionmnist/
│   ├── train_fashionmnist.py
│   ├── fashionmnist_experiment_summary_mean_std.csv
│   └── fashionmnist_faithfulness_comparison_mean_std.csv
├── cifar10/
│   ├── train_cifar10.py
│   ├── cifar10_experiment_summary_mean_std.csv
│   └── cifar10_faithfulness_comparison_mean_std.csv
├── cifar100/
│   ├── train_cifar100.py
│   ├── cifar100_experiment_summary_mean_std.csv
│   └── cifar100_faithfulness_comparison_mean_std.csv
└── imagenet100/
    ├── train_imagenet100.py
    ├── train_imagenet100_pretrained_late-block.py
    ├── imagenet100_experiment_summary_mean_std.csv
    ├── imagenet100_faithfulness_comparison_mean_std.csv
    ├── late-block_experiment_summary_mean_std.csv
    └── late-block_faithfulness_comparison_mean_std.csv
```

Note: the current pretrained late-block script filename in the working tree has
a trailing space. If your shell cannot find it, run `ls -lb imagenet100` to see
the exact filename, or rename it locally.

## Setup

Create and activate a Python environment, then install the Python packages used
by the scripts:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scipy ptflops torchinfo datasets pillow timm
```

The FashionMNIST, CIFAR-10, and CIFAR-100 scripts download data through
`torchvision`. The ImageNet-100 scripts load `clane9/imagenet-100` from Hugging
Face Datasets, so they require network access and enough local disk cache.

## Running Experiments

Run scripts from inside their dataset directory. The scripts use relative paths
for downloaded data and generated outputs.

### FashionMNIST

```bash
cd fashionmnist
python train_fashionmnist.py
```

### CIFAR-10

```bash
cd cifar10
python train_cifar10.py
```

### CIFAR-100

```bash
cd cifar100
python train_cifar100.py
```

### ImageNet-100

```bash
cd imagenet100
python train_imagenet100.py
```

### ImageNet-100 Pretrained Late-Block Experiment

```bash
cd imagenet100
python 'train_imagenet100_pretrained_late-block.py '
```

This experiment starts from `vit_tiny_patch16_224` via `timm`, replaces the
last configured transformer blocks with baseline or SVDA attention variants,
and trains both models for the configured seeds.

## Configuration

Each training script has a top-level `params` dictionary. Edit that dictionary
to change dataset settings, model size, seeds, training length, reporting, and
profiling behavior.

Common fields include:

- `seeds`: seeds used for repeated baseline and SVDA runs
- `epochs`, `batch_size`, `lr`: training hyperparameters
- `use_svda`: toggled internally by the pipeline for baseline and SVDA runs
- `compute_interpretability`: enables attention and spectral metrics
- `compute_efficiency`: enables latency, memory, MAC, and parameter profiling
- `profile_batch_sizes`: batch sizes used for efficiency profiling
- `run_training` and `run_reporting`: both are enabled by default

The scripts do not currently expose command-line arguments.

## Outputs

During a run, each script creates a run root such as:

- `runs_fashionmnist/`
- `runs_cifar10/`
- `runs_cifar100/`
- `runs_imagenet100/`
- `runs_imagenet100_pretrained/`

Per-run outputs are organized into:

- `checkpoints/`: final model checkpoint files
- `metrics/`: per-seed NumPy metric arrays
- `figures/`: publication-style plots and attention overlays
- `profiles/`: efficiency profiling outputs
- `run_summary.json`: metrics, checkpoint paths, and profile summaries
- `model_summary.txt`: architecture summary from `torchinfo`

The reporting step also writes aggregate CSV files under the run root:

- `experiment_summary.csv`
- `experiment_summary_mean_std.csv`
- `faithfulness_comparison.csv`
- `faithfulness_comparison_mean_std.csv`
- `metric_summary_over_epochs_seed*.csv`

The CSV files committed in each dataset folder are compact mean/std summaries
from completed runs.

## Metrics

The scripts report:

- Best and final test accuracy
- Mean epoch time
- Attention entropy, effective rank, sparsity, selectivity, robustness, and
  angular alignment where available
- Faithfulness metrics using deletion and insertion at top-k fractions
- Forward and backward latency
- Throughput
- Peak CUDA memory when CUDA is available
- MACs and parameter counts

## Device Selection

The scripts select the first available backend in this order:

1. Apple MPS
2. CUDA
3. CPU

CUDA-specific peak memory values are reported only when running on CUDA.
