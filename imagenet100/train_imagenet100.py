import os
import time
import random
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ptflops import get_model_complexity_info
from datasets import load_dataset
from PIL import Image

# ===== CONFIGURABLE PARAMETERS =====
params = {
    "dataset": "ImageNet100",
    "image_size": 224,
    "patch_size": 8,
    "embed_dim": 96,
    "num_heads": 2,
    "num_layers": 8,
    "num_classes": 100,
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
    "lambda_ortho": 3e-4,
    "use_svda": True,  # ← Toggle this for baseline vs SVDA
    "seed": 42,
    "seeds": [42, 43, 44],
    "log_root": "runs_imagenet100",
    "compute_interpretability": True,
    "compute_efficiency": True,
    "faithfulness_eval_samples": 128,
    "faithfulness_topk_fracs": [0.1, 0.2, 0.3],
    "attention_layer_aggregation": "last",
    "attention_head_aggregation": "mean",
    "profile_batch_sizes": [1, 64],
    "save_publication_figures": True,
    "save_plot_dpi": 300,
    "max_overlay_examples": 8,
    "run_training": True,
    "run_reporting": True,
    "in_channels": 3
}
def get_latest_run_dirs(model_name):
    latest_seed = params["seeds"][-1]
    run_name = get_run_name(model_name, latest_seed)
    run_root = os.path.join(params["log_root"], run_name)
    return {
        "root": run_root,
        "checkpoints": os.path.join(run_root, "checkpoints"),
        "metrics": os.path.join(run_root, "metrics"),
        "figures": os.path.join(run_root, "figures"),
        "profiles": os.path.join(run_root, "profiles"),
    }


def save_current_figure(path):
    plt.savefig(path, dpi=params["save_plot_dpi"], bbox_inches="tight")


def save_publication_figure(fig, path):
    fig.savefig(path, dpi=params["save_plot_dpi"], bbox_inches="tight")
    plt.close(fig)
def build_faithfulness_table(all_summaries):
    rows = []
    for item in all_summaries:
        deletion_summary = item.get("deletion_summary") or {}
        insertion_summary = item.get("insertion_summary") or {}
        for frac in params["faithfulness_topk_fracs"]:
            frac_key = f"{frac:.1f}"
            rows.append({
                "model": item["model"],
                "seed": item["seed"],
                "topk_frac": frac,
                "deletion_top": deletion_summary.get(f"top_{frac_key}", np.nan),
                "deletion_random": deletion_summary.get(f"random_{frac_key}", np.nan),
                "insertion_top": insertion_summary.get(f"top_{frac_key}", np.nan),
                "insertion_random": insertion_summary.get(f"random_{frac_key}", np.nan),
            })
    return pd.DataFrame(rows)


def save_faithfulness_outputs(all_summaries):
    faithfulness_df = build_faithfulness_table(all_summaries)
    faithfulness_csv = os.path.join(params["log_root"], "faithfulness_comparison.csv")
    faithfulness_df.to_csv(faithfulness_csv, index=False)

    mean_df = faithfulness_df.groupby(["model", "topk_frac"])[[
        "deletion_top", "deletion_random", "insertion_top", "insertion_random"
    ]].mean().reset_index()
    std_df = faithfulness_df.groupby(["model", "topk_frac"])[[
        "deletion_top", "deletion_random", "insertion_top", "insertion_random"
    ]].std().reset_index()
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in ["deletion_top", "deletion_random", "insertion_top", "insertion_random"]})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in ["deletion_top", "deletion_random", "insertion_top", "insertion_random"]})
    merged = mean_df.merge(std_df, on=["model", "topk_frac"], how="left")
    merged.to_csv(os.path.join(params["log_root"], "faithfulness_comparison_mean_std.csv"), index=False)
    return faithfulness_df, merged


def create_attention_overlay_figure(attn_map, image, patch_size, title=""):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis('off')

    N = attn_map.shape[-1] - 1
    heatmap = attn_map[1:, 0].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
    heatmap = heatmap.numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    from scipy.ndimage import zoom
    heatmap = zoom(heatmap, patch_size)
    ax.imshow(heatmap, cmap='jet', alpha=0.4)
    return fig


def save_attention_overlay_comparison(base_model, svda_model, loader, current_device, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sample_img, sample_label = next(iter(loader))
    sample_img = sample_img[:1].to(current_device)

    baseline_agg = aggregate_attention_maps(
        base_model,
        sample_img,
        layer_aggregation=params["attention_layer_aggregation"],
        head_aggregation="mean",
    )
    svda_agg = aggregate_attention_maps(
        svda_model,
        sample_img,
        layer_aggregation=params["attention_layer_aggregation"],
        head_aggregation="mean",
    )

    if baseline_agg is None or svda_agg is None:
        print("Skipping attention overlay comparison because attention weights were not available.")
        return

    baseline_map = baseline_agg[0].cpu()
    svda_map = svda_agg[0].cpu()

    baseline_fig = create_attention_overlay_figure(
        baseline_map,
        sample_img[0].cpu(),
        patch_size=params["patch_size"],
        title="Baseline Attention Overlay"
    )
    save_publication_figure(baseline_fig, os.path.join(save_dir, "baseline_attention_overlay.png"))

    svda_fig = create_attention_overlay_figure(
        svda_map,
        sample_img[0].cpu(),
        patch_size=params["patch_size"],
        title="SVDA Attention Overlay"
    )
    save_publication_figure(svda_fig, os.path.join(save_dir, "svda_attention_overlay.png"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    image_np = sample_img[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis('off')

    for ax, amap, ttl in zip(axes[1:], [baseline_map, svda_map], ["Baseline", "SVDA"]):
        ax.imshow(image_np)
        N = amap.shape[-1] - 1
        heatmap = amap[1:, 0].reshape(int(np.sqrt(N)), int(np.sqrt(N))).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        from scipy.ndimage import zoom
        heatmap = zoom(heatmap, params["patch_size"])
        ax.imshow(heatmap, cmap='jet', alpha=0.4)
        ax.set_title(ttl)
        ax.axis('off')

    save_publication_figure(fig, os.path.join(save_dir, "baseline_vs_svda_attention_overlay_comparison.png"))



def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# Reusable function to measure model complexity
def measure_model_complexity(model, input_res=None):
    if input_res is None:
        input_res = (params["in_channels"], params["image_size"], params["image_size"])
    model = model.cpu()
    model.eval()
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            model, input_res,
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
    return macs, params

def sigma_entropy_loss(sigmas):
    p = (sigmas ** 2)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # (L, H)
    return entropy.mean()

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

set_seed(params["seed"])
device = get_device()
print("Using device:", device)


train_transform = transforms.Compose([
    transforms.Resize((params["image_size"], params["image_size"]), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((params["image_size"], params["image_size"]), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label

def build_dataloaders(current_params):
    hf_ds = load_dataset("clane9/imagenet-100")

    train_split_name = "train"
    test_split_name = "validation" if "validation" in hf_ds else "test"

    train_data = HFDatasetWrapper(hf_ds[train_split_name], transform=train_transform)
    test_data = HFDatasetWrapper(hf_ds[test_split_name], transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=current_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=current_params["batch_size"], shuffle=False)
    return train_loader, test_loader


train_loader, test_loader = build_dataloaders(params)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x
    
class StandardMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_weights = None

    def forward(self, x):
        attn_output, attn_weights = self.mha(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )
        self.attn_weights = attn_weights.detach()   # shape: (B, H, N, N)
        return attn_output
    
class SVDMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, lambda_ortho=0.0):
        super().__init__()
        self.dim = dim
        self.h = num_heads
        self.dk = dim // num_heads
        self.lambda_ortho = lambda_ortho

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)

        self.sigmas = nn.Parameter(torch.empty(self.h, self.dk))
        nn.init.normal_(self.sigmas, mean=1.0, std=0.1)  # Or try mean=0.5 for sharper contrast

        self.attn_weights = None  # store A
        self.angular_alignments = None  # buffer for angular alignment

    def forward(self, x):
        B, N, _ = x.size()
        Q = self.Wq(x).view(B, N, self.h, self.dk).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.h, self.dk).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.h, self.dk).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        # Angular alignment: cosine similarity between Q and K (mean per batch, per head)
        cos_sim = torch.sum(Q * K, dim=-1).mean(dim=-1).mean(dim=0)  # (H,)
        self.angular_alignments = cos_sim.detach()

        # Element-wise scaling of Q with sigmas: shape (1, H, 1, Dk)
        weighted_Q = Q * self.sigmas.unsqueeze(0).unsqueeze(2)
        scores = torch.matmul(weighted_Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        A = F.softmax(scores, dim=-1)

        self.attn_weights = A.detach()  # <=== this line is essential

        Z = torch.matmul(A, V).transpose(1, 2).reshape(B, N, self.dim)
        return self.Wo(Z)

    def orthogonality_loss(self):
        q_gram = self.Wq.weight.T @ self.Wq.weight
        k_gram = self.Wk.weight.T @ self.Wk.weight
        eye_q = torch.eye(q_gram.size(0), device=self.Wq.weight.device)
        eye_k = torch.eye(k_gram.size(0), device=self.Wk.weight.device)
        loss_q = ((q_gram - eye_q) ** 2).mean()
        loss_k = ((k_gram - eye_k) ** 2).mean()
        return self.lambda_ortho * (loss_q + loss_k)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_svda, lambda_ortho):
        super().__init__()
        self.attn = SVDMultiHeadAttention(dim, num_heads, lambda_ortho) if use_svda else StandardMultiHeadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x

    def get_ortho_loss(self):
        return self.attn.orthogonality_loss() if hasattr(self.attn, "orthogonality_loss") else 0.0


class TransformerClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embed = PatchEmbedding(params["in_channels"], params["patch_size"], params["embed_dim"], params["image_size"])
        self.blocks = nn.ModuleList([
            TransformerBlock(params["embed_dim"], params["num_heads"], params["use_svda"], params["lambda_ortho"])
            for _ in range(params["num_layers"])
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, params["embed_dim"]))
        self.norm = nn.LayerNorm(params["embed_dim"])
        self.head = nn.Linear(params["embed_dim"], params["num_classes"])

    def forward(self, x):
        B = x.size(0)
        x = self.embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, 0]))

    def orthogonality_loss(self):
        return sum([b.get_ortho_loss() for b in self.blocks])

def get_run_name(model_name, seed):
    return f"imagenet100_{model_name}_seed{seed}"

def get_run_dirs(current_params, model_name, seed):
    run_name = get_run_name(model_name, seed)
    run_root = os.path.join(current_params["log_root"], run_name)
    dirs = {
        "root": run_root,
        "checkpoints": os.path.join(run_root, "checkpoints"),
        "metrics": os.path.join(run_root, "metrics"),
        "figures": os.path.join(run_root, "figures"),
        "profiles": os.path.join(run_root, "profiles"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def summarize_attention_metrics(model, current_params, eval_loader, current_device, epsilon=1e-2):
    model.eval()
    summary = {
        "entropy": np.nan,
        "effective_rank": np.nan,
        "sparsity": np.nan,
        "selectivity": np.nan,
        "robustness": np.nan,
        "angular_alignment": np.nan,
    }
    if not current_params["use_svda"]:
        return summary

    with torch.no_grad():
        all_sigmas = torch.stack([block.attn.sigmas.detach().cpu() for block in model.blocks])
        ent = spectral_entropy(all_sigmas)
        summary["entropy"] = ent.mean().item()
        summary["effective_rank"] = torch.exp(ent).mean().item()
        summary["sparsity"] = spectral_sparsity(all_sigmas)

        selectivity_vals = []
        for block in model.blocks:
            if hasattr(block.attn, "attn_weights") and block.attn.attn_weights is not None:
                A = block.attn.attn_weights
                sel = 1 - (A ** 2).sum(dim=-1) / (A.sum(dim=-1) ** 2 + 1e-8)
                selectivity_vals.append(sel.mean().item())
        if selectivity_vals:
            summary["selectivity"] = float(np.mean(selectivity_vals))

        sample_x, _ = next(iter(eval_loader))
        sample_x = sample_x[:1].to(current_device)
        delta = torch.randn_like(sample_x) * epsilon
        perturbed_x = torch.clamp(sample_x + delta, 0, 1)
        _ = model(sample_x)
        A_clean_all = [block.attn.attn_weights for block in model.blocks]
        _ = model(perturbed_x)
        A_perturbed_all = [block.attn.attn_weights for block in model.blocks]
        robustness_vals = []
        for A_clean, A_perturbed in zip(A_clean_all, A_perturbed_all):
            delta_A = (A_clean - A_perturbed).norm(p='fro') / A_clean.numel()
            robustness_vals.append(delta_A.item())
        if robustness_vals:
            summary["robustness"] = float(np.mean(robustness_vals))

        alignments = []
        for block in model.blocks:
            if hasattr(block.attn, "angular_alignments") and block.attn.angular_alignments is not None:
                alignments.append(block.attn.angular_alignments.mean().item())
        if alignments:
            summary["angular_alignment"] = float(np.mean(alignments))

    return summary


def aggregate_attention_maps(model, x, layer_aggregation="last", head_aggregation="mean"):
    model.eval()
    with torch.no_grad():
        _ = model(x)

    maps = []
    for block in model.blocks:
        if hasattr(block.attn, "attn_weights") and block.attn.attn_weights is not None:
            maps.append(block.attn.attn_weights)

    if not maps:
        return None

    if layer_aggregation == "last":
        A = maps[-1]
    elif layer_aggregation == "mean":
        A = torch.stack(maps, dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unsupported layer_aggregation: {layer_aggregation}")

    if head_aggregation == "mean":
        A = A.mean(dim=1)
    elif head_aggregation == "max":
        A = A.max(dim=1).values
    elif head_aggregation == "none":
        return A
    else:
        raise ValueError(f"Unsupported head_aggregation: {head_aggregation}")

    return A


def extract_cls_attention_scores(model, x, layer_aggregation="last", head_aggregation="mean"):
    A = aggregate_attention_maps(model, x, layer_aggregation=layer_aggregation, head_aggregation=head_aggregation)
    if A is None:
        return None
    if A.dim() == 4:
        A = A.mean(dim=1)
    cls_to_patches = A[:, 0, 1:]
    return cls_to_patches


def build_patch_mask(attention_scores, patch_size, image_shape, topk_frac, mode="top"):
    B, num_patches = attention_scores.shape
    _, _, H, W = image_shape
    patches_per_side = int(np.sqrt(num_patches))
    k = max(1, int(num_patches * topk_frac))
    if mode == "top":
        selected_indices = torch.topk(attention_scores, k=k, dim=-1).indices
    elif mode == "bottom":
        selected_indices = torch.topk(-attention_scores, k=k, dim=-1).indices
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    patch_mask = torch.zeros((B, 1, H, W), device=attention_scores.device)
    for b in range(B):
        for idx in selected_indices[b]:
            patch_idx = idx.item()
            row = patch_idx // patches_per_side
            col = patch_idx % patches_per_side
            h_start = row * patch_size
            h_end = h_start + patch_size
            w_start = col * patch_size
            w_end = w_start + patch_size
            patch_mask[b, :, h_start:h_end, w_start:w_end] = 1.0
    return patch_mask


def mask_topk_patches(images, attention_scores, patch_size, topk_frac):
    keep_mask = 1.0 - build_patch_mask(attention_scores, patch_size, images.shape, topk_frac, mode="top")
    return images * keep_mask


def restore_topk_patches(images, attention_scores, patch_size, topk_frac, baseline_value=0.0):
    base = torch.full_like(images, fill_value=baseline_value)
    restore_mask = build_patch_mask(attention_scores, patch_size, images.shape, topk_frac, mode="top")
    return base + images * restore_mask


def compute_attention_faithfulness(model, loader, current_device, current_params, max_samples=128, topk_fracs=None):
    if topk_fracs is None:
        topk_fracs = [0.1, 0.2, 0.3]

    model.eval()
    deletion_stats = {f"top_{frac:.1f}": [] for frac in topk_fracs}
    deletion_stats.update({f"random_{frac:.1f}": [] for frac in topk_fracs})
    insertion_stats = {f"top_{frac:.1f}": [] for frac in topk_fracs}
    insertion_stats.update({f"random_{frac:.1f}": [] for frac in topk_fracs})
    processed = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(current_device), y.to(current_device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
            base_conf = probs.gather(1, pred.unsqueeze(1)).squeeze(1)

            attn_scores = extract_cls_attention_scores(
                model,
                x,
                layer_aggregation=current_params["attention_layer_aggregation"],
                head_aggregation=current_params["attention_head_aggregation"],
            )
            if attn_scores is None:
                print("Skipping faithfulness computation because attention scores were not available.")
                return None, None

            for frac in topk_fracs:
                masked_top = mask_topk_patches(x, attn_scores, current_params["patch_size"], frac)
                masked_logits = model(masked_top)
                masked_probs = torch.softmax(masked_logits, dim=-1)
                masked_conf = masked_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                deletion_stats[f"top_{frac:.1f}"].append((base_conf - masked_conf).mean().item())

                restored_top = restore_topk_patches(x, attn_scores, current_params["patch_size"], frac)
                restored_logits = model(restored_top)
                restored_probs = torch.softmax(restored_logits, dim=-1)
                restored_conf = restored_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                insertion_stats[f"top_{frac:.1f}"].append(restored_conf.mean().item())

                rand_scores = torch.rand_like(attn_scores)
                masked_rand = mask_topk_patches(x, rand_scores, current_params["patch_size"], frac)
                rand_logits = model(masked_rand)
                rand_probs = torch.softmax(rand_logits, dim=-1)
                rand_conf = rand_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                deletion_stats[f"random_{frac:.1f}"].append((base_conf - rand_conf).mean().item())

                restored_rand = restore_topk_patches(x, rand_scores, current_params["patch_size"], frac)
                restored_rand_logits = model(restored_rand)
                restored_rand_probs = torch.softmax(restored_rand_logits, dim=-1)
                restored_rand_conf = restored_rand_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
                insertion_stats[f"random_{frac:.1f}"].append(restored_rand_conf.mean().item())

            processed += x.size(0)
            if processed >= max_samples:
                break

    deletion_summary = {k: float(np.mean(v)) if len(v) > 0 else np.nan for k, v in deletion_stats.items()}
    insertion_summary = {k: float(np.mean(v)) if len(v) > 0 else np.nan for k, v in insertion_stats.items()}
    return deletion_summary, insertion_summary


def profile_model_efficiency(model, current_device, input_res=None, batch_sizes=(1, 64), num_warmup=5, num_iters=20):
    results = []
    if input_res is None:
        input_res = (params["in_channels"], params["image_size"], params["image_size"])
    model = model.to(current_device)

    for batch_size in batch_sizes:
        dummy_input = torch.randn((batch_size, *input_res), device=current_device)
        dummy_target = torch.randint(0, params["num_classes"], (batch_size,), device=current_device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        model.eval()
        for _ in range(num_warmup):
            _ = model(dummy_input)
        if current_device.type == "cuda":
            torch.cuda.synchronize()

        forward_times = []
        if current_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(current_device)
        for _ in range(num_iters):
            start = time.time()
            _ = model(dummy_input)
            if current_device.type == "cuda":
                torch.cuda.synchronize()
            forward_times.append(time.time() - start)

        peak_mem_mb = None
        if current_device.type == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)

        model.train()
        backward_times = []
        for _ in range(num_warmup):
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        if current_device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(num_iters):
            optimizer.zero_grad()
            start = time.time()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            if current_device.type == "cuda":
                torch.cuda.synchronize()
            backward_times.append(time.time() - start)

        with torch.no_grad():
            macs, n_params = get_model_complexity_info(
                model.cpu(), input_res, as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
        model = model.to(current_device)

        avg_forward = float(np.mean(forward_times))
        avg_backward = float(np.mean(backward_times))
        throughput = float(batch_size / avg_forward) if avg_forward > 0 else np.nan

        results.append({
            "batch_size": batch_size,
            "forward_ms": 1000.0 * avg_forward,
            "backward_ms": 1000.0 * avg_backward,
            "throughput_img_s": throughput,
            "peak_mem_mb": peak_mem_mb,
            "macs": macs,
            "params": n_params,
        })

    return results
def train(model, loader, optimizer, criterion, use_svda):
    model.train()
    total_loss, total_correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        if use_svda:
            loss += model.orthogonality_loss()
            # Optional: add entropy regularization
            if hasattr(model, "blocks"):
                all_sigmas = torch.stack([b.attn.sigmas for b in model.blocks])  # (L, H, Dk)
                entropy_loss = sigma_entropy_loss(all_sigmas)
                loss += 1e-3 * entropy_loss  # ← You can tune this weight

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (logits.argmax(1) == y).sum().item()
    acc = total_correct / len(loader.dataset)
    return total_loss / len(loader), acc

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)

def spectral_entropy(sigmas):
    # sigmas shape: (L, H, Dk)
    p = (sigmas ** 2)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)  # normalize over Dk
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)  # shape: (L, H)
    return entropy  # <-- return tensor, NOT entropy.mean().item()

def effective_rank(entropy_tensor):
    return torch.exp(entropy_tensor).mean().item()

def spectral_sparsity(sigmas, threshold=1e-3):
    mask = (sigmas.abs() < threshold).float()
    return mask.mean().item()

def run_experiment(use_svda_flag, current_seed=None):
    if current_seed is None:
        current_seed = params["seed"]

    set_seed(current_seed)
    current_params = {**params, "use_svda": use_svda_flag, "seed": current_seed}
    local_train_loader, local_test_loader = build_dataloaders(current_params)
    model_name = "svda" if use_svda_flag else "baseline"
    run_dirs = get_run_dirs(current_params, model_name, current_seed)

    print(f"\nTraining {'SVDA' if use_svda_flag else 'Baseline'} Transformer | seed={current_seed}...")
    model = TransformerClassifier(current_params).to(device)

    from torchinfo import summary
    model_summary = summary(
        model,
        input_size=(1, current_params["in_channels"], current_params["image_size"], current_params["image_size"]),
        verbose=0,
    )
    with open(os.path.join(run_dirs["root"], "model_summary.txt"), "w") as f:
        f.write(str(model_summary))

    optimizer = torch.optim.Adam(model.parameters(), lr=current_params["lr"])
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []
    entropy_log, rank_log, sparsity_log, time_log = [], [], [], []
    selectivity_log, robustness_log, angular_log = [], [], []
    epsilon = 1e-2

    if use_svda_flag:
        entropy_layerwise = []
        rank_layerwise = []
        sparsity_layerwise = []
        alignment_layerwise = []
        selectivity_layerwise = []
        robustness_layerwise = []

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    best_acc = -np.inf
    best_state_dict = None

    for epoch in range(current_params["epochs"]):
        start_time = time.time()
        train_loss, train_acc = train(
            model, local_train_loader, optimizer, criterion, use_svda=use_svda_flag
        )
        test_acc = evaluate(model, local_test_loader)
        duration = time.time() - start_time
        time_log.append(duration)

        print(
            f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | Time: {duration:.2f}s"
        )

        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if use_svda_flag:
            with torch.no_grad():
                all_sigmas = [block.attn.sigmas.detach().cpu() for block in model.blocks]
                all_sigmas = torch.stack(all_sigmas)

                ent = spectral_entropy(all_sigmas)
                effr = effective_rank(ent)
                spar = spectral_sparsity(all_sigmas)

                entropy_log.append(ent.mean().item())
                entropy_layerwise.append(ent.numpy())
                rank_log.append(effr)
                rank_layerwise.append(torch.exp(ent).numpy())
                sparsity_log.append(spar)
                sparsity_layerwise.append((all_sigmas.abs() < 1e-3).float().mean(dim=-1).numpy())

                selectivity_per_layer = []
                for block in model.blocks:
                    if hasattr(block.attn, "attn_weights") and block.attn.attn_weights is not None:
                        A = block.attn.attn_weights
                        sel = 1 - (A**2).sum(dim=-1) / (A.sum(dim=-1)**2 + 1e-8)
                        selectivity_per_layer.append(sel.mean(dim=(0, 1)).cpu().numpy())
                    else:
                        selectivity_per_layer.append(np.nan * np.ones(current_params["num_heads"]))
                selectivity_layerwise.append(np.stack(selectivity_per_layer))
                selectivity_log.append(np.mean(selectivity_layerwise[-1]))

                robustness_per_layer = []
                sample_x, _ = next(iter(local_test_loader))
                sample_x = sample_x[:1].to(device)
                delta = torch.randn_like(sample_x) * epsilon
                perturbed_x = torch.clamp(sample_x + delta, 0, 1)
                _ = model(sample_x)
                A_clean_all = [block.attn.attn_weights for block in model.blocks]
                _ = model(perturbed_x)
                A_perturbed_all = [block.attn.attn_weights for block in model.blocks]
                for A_clean, A_perturbed in zip(A_clean_all, A_perturbed_all):
                    delta_A = (A_clean - A_perturbed).norm(p="fro") / A_clean.numel()
                    robustness_per_layer.append(delta_A.item())
                robustness_layerwise.append(robustness_per_layer)
                robustness_log.append(np.mean(robustness_per_layer))

                angular_log.append(
                    np.mean([b.attn.angular_alignments.mean().item() for b in model.blocks])
                )
                alignment_layerwise.append(
                    torch.stack([b.attn.angular_alignments for b in model.blocks]).cpu().numpy()
                )

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_model_state_dict": best_state_dict,
        "train_acc": train_accs,
        "test_acc": test_accs,
        "time_log": time_log,
        "seed": current_seed,
        "params": current_params,
        "best_acc": best_acc,
    }

    if use_svda_flag:
        final_checkpoint.update(
            {
                "entropy": entropy_log,
                "effective_rank": rank_log,
                "sparsity": sparsity_log,
                "selectivity": selectivity_log,
                "robustness": robustness_log,
                "angular_alignment": angular_log,
            }
        )

    checkpoint_path = os.path.join(
        run_dirs["checkpoints"], f"{model_name}_results_seed{current_seed}.pth"
    )
    torch.save(final_checkpoint, checkpoint_path)

    np.save(os.path.join(run_dirs["metrics"], f"test_accuracy_seed{current_seed}.npy"), np.array(test_accs))
    np.save(os.path.join(run_dirs["metrics"], f"epoch_time_seed{current_seed}.npy"), np.array(time_log))

    if use_svda_flag:
        np.save(os.path.join(run_dirs["metrics"], f"entropy_seed{current_seed}.npy"), np.array(entropy_log))
        np.save(os.path.join(run_dirs["metrics"], f"effective_rank_seed{current_seed}.npy"), np.array(rank_log))
        np.save(os.path.join(run_dirs["metrics"], f"sparsity_seed{current_seed}.npy"), np.array(sparsity_log))
        np.save(os.path.join(run_dirs["metrics"], f"selectivity_seed{current_seed}.npy"), np.array(selectivity_log))
        np.save(os.path.join(run_dirs["metrics"], f"robustness_seed{current_seed}.npy"), np.array(robustness_log))
        np.save(os.path.join(run_dirs["metrics"], f"angular_alignment_seed{current_seed}.npy"), np.array(angular_log))
        np.save(os.path.join(run_dirs["metrics"], f"entropy_layerwise_seed{current_seed}.npy"), np.array(entropy_layerwise))
        np.save(os.path.join(run_dirs["metrics"], f"rank_layerwise_seed{current_seed}.npy"), np.array(rank_layerwise))
        np.save(os.path.join(run_dirs["metrics"], f"sparsity_layerwise_seed{current_seed}.npy"), np.array(sparsity_layerwise))
        np.save(
            os.path.join(run_dirs["metrics"], f"angular_alignment_layerwise_seed{current_seed}.npy"),
            np.array(alignment_layerwise),
        )
        np.save(
            os.path.join(run_dirs["metrics"], f"selectivity_layerwise_seed{current_seed}.npy"),
            np.array(selectivity_layerwise),
        )
        np.save(
            os.path.join(run_dirs["metrics"], f"robustness_layerwise_seed{current_seed}.npy"),
            np.array(robustness_layerwise),
        )

    attention_summary = summarize_attention_metrics(model, current_params, local_test_loader, device)
    deletion_summary, insertion_summary = compute_attention_faithfulness(
        model,
        local_test_loader,
        device,
        current_params,
        max_samples=current_params["faithfulness_eval_samples"],
        topk_fracs=current_params["faithfulness_topk_fracs"],
    )
    efficiency_summary = (
        profile_model_efficiency(
            model,
            device,
            input_res=(current_params["in_channels"], current_params["image_size"], current_params["image_size"]),
            batch_sizes=tuple(current_params["profile_batch_sizes"]),
        )
        if current_params["compute_efficiency"]
        else None
    )

    run_summary = {
        "run_name": get_run_name(model_name, current_seed),
        "model": model_name,
        "seed": current_seed,
        "best_acc": best_acc,
        "final_acc": test_accs[-1],
        "mean_epoch_time": float(np.mean(time_log)),
        "attention_summary": attention_summary,
        "deletion_summary": deletion_summary,
        "insertion_summary": insertion_summary,
        "efficiency_summary": efficiency_summary,
        "checkpoint_path": checkpoint_path,
    }

    with open(os.path.join(run_dirs["root"], "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    return {
        "model": model,
        "train_acc": train_accs,
        "test_acc": test_accs,
        "time_log": time_log,
        "entropy": entropy_log if use_svda_flag else None,
        "effective_rank": rank_log if use_svda_flag else None,
        "sparsity": sparsity_log if use_svda_flag else None,
        "selectivity": selectivity_log if use_svda_flag else None,
        "robustness": robustness_log if use_svda_flag else None,
        "angular_alignment": angular_log if use_svda_flag else None,
        "summary": run_summary,
    }


def run_training_pipeline():
    all_run_summaries = []
    last_baseline_results = None
    last_svda_results = None

    for seed in params["seeds"]:
        baseline_results = run_experiment(use_svda_flag=False, current_seed=seed)
        svda_results = run_experiment(use_svda_flag=True, current_seed=seed)
        all_run_summaries.append(baseline_results["summary"])
        all_run_summaries.append(svda_results["summary"])
        last_baseline_results = baseline_results
        last_svda_results = svda_results

    os.makedirs(params["log_root"], exist_ok=True)
    with open(os.path.join(params["log_root"], "all_run_summaries.json"), "w") as f:
        json.dump(all_run_summaries, f, indent=2)

    return {
        "all_run_summaries": all_run_summaries,
        "last_baseline_results": last_baseline_results,
        "last_svda_results": last_svda_results,
    }

def print_training_step_stats(model, input_res=None, device=None):
    if input_res is None:
        input_res = (params["in_channels"], params["image_size"], params["image_size"])
    if device is None:
        device = get_device()
    model = model.to(device)
    model.train()

    dummy_input = torch.randn((64, *input_res)).to(device)
    dummy_target = torch.randint(0, params["num_classes"], (64,)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    forward_times, backward_times = [], []

    for _ in range(10):
        optimizer.zero_grad()

        start_fwd = time.time()
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_fwd = time.time()
        forward_times.append(end_fwd - start_fwd)

        start_bwd = time.time()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_bwd = time.time()
        backward_times.append(end_bwd - start_bwd)

    print(f"Model: {model.__class__.__name__}")
    print(f"  Avg forward time per batch: {1000 * sum(forward_times)/len(forward_times):.2f} ms")
    print(f"  Avg backward+update time per batch: {1000 * sum(backward_times)/len(backward_times):.2f} ms")

    with torch.no_grad():
        macs, model_params = get_model_complexity_info(
            model,
            input_res,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )
    print(f"  MACs (inference-only est.): {macs}")
    print(f"  Parameters: {model_params}\n")


def generate_reports(training_outputs):
    all_run_summaries = training_outputs["all_run_summaries"]
    last_baseline_results = training_outputs["last_baseline_results"]
    last_svda_results = training_outputs["last_svda_results"]

    baseline_ckpt_path = last_baseline_results["summary"]["checkpoint_path"]
    svda_ckpt_path = last_svda_results["summary"]["checkpoint_path"]
    baseline_ckpt = torch.load(baseline_ckpt_path, map_location=device)
    svda_ckpt = torch.load(svda_ckpt_path, map_location=device)

    base_model = TransformerClassifier({**params, "use_svda": False}).to(device)
    svda_model = TransformerClassifier({**params, "use_svda": True}).to(device)
    base_model.load_state_dict(baseline_ckpt["model_state_dict"])
    svda_model.load_state_dict(svda_ckpt["model_state_dict"])
    base_model.eval()
    svda_model.eval()

    baseline_test = baseline_ckpt["test_acc"]
    baseline_time = baseline_ckpt["time_log"]
    svda_test = svda_ckpt["test_acc"]
    svda_entropy = svda_ckpt["entropy"]
    svda_rank = svda_ckpt["effective_rank"]
    svda_sparsity = svda_ckpt["sparsity"]
    svda_time = svda_ckpt["time_log"]
    svda_selectivity = svda_ckpt["selectivity"]
    svda_robustness = svda_ckpt["robustness"]
    svda_angular = svda_ckpt["angular_alignment"]

    latest_seed = params["seeds"][-1]
    log_dir = os.path.join(params["log_root"], get_run_name("svda", latest_seed), "metrics")
    seed = latest_seed

    def extract_attention_maps(model, x, layer_idx=-1, head_idx=0):
        model.eval()
        with torch.no_grad():
            _ = model(x)
        attn_block = model.blocks[layer_idx].attn
        A = attn_block.attn_weights
        return A[0, head_idx].cpu()

    def visualize_attention_overlay(attn_map, image, patch_size, title=""):
        image_np = image.permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots()
        ax.imshow(image_np)
        ax.set_title(title)
        ax.axis("off")

        N = attn_map.shape[-1] - 1
        heatmap = attn_map[1:, 0].reshape(int(np.sqrt(N)), int(np.sqrt(N)))
        heatmap = heatmap.numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        from scipy.ndimage import zoom
        heatmap = zoom(heatmap, patch_size)
        ax.imshow(heatmap, cmap="jet", alpha=0.4)
        plt.show()

    def headwise_entropy_colorbars(sigmas):
        p = (sigmas ** 2)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        for l in range(entropy.shape[0]):
            plt.bar(np.arange(entropy.shape[1]), entropy[l].numpy(), color="royalblue")
            plt.title(f"Spectral Entropy per Head - Layer {l}")
            plt.xlabel("Head")
            plt.ylabel("Entropy")
            plt.grid(True)
            plt.show()

    def load_metric_array(name):
        return np.load(os.path.join(log_dir, f"{name}_layerwise_seed{seed}.npy"))

    def plot_metric_by_epoch(df, metric):
        fig = plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="epoch", y=metric, estimator="mean", errorbar="sd")
        plt.ylabel("")
        plt.grid(True)
        plt.tight_layout()
        if params["save_publication_figures"]:
            safe_metric = metric.lower().replace(" ", "_")
            save_current_figure(os.path.join(get_latest_run_dirs("svda")["figures"], f"metric_epoch_{safe_metric}.png"))
        plt.show()

    def plot_metric_by_layer(df, metric):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="layer", y=metric)
        plt.title(f"{metric} Distribution per Layer")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_metric_histogram(df, metric):
        plt.figure(figsize=(8, 4))
        sns.histplot(df[metric], bins=30, kde=True)
        plt.title(f"Histogram of {metric}")
        plt.tight_layout()
        plt.show()

    def plot_head_layer_heatmap(df, metric, epoch=None):
        subset = df[df["epoch"] == epoch] if epoch is not None else df
        pivot = subset.pivot_table(index="layer", columns="head", values=metric)
        fig = plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".2f")
        plt.title(f"{metric} Heatmap per Layer/Head" + (f" (Epoch {epoch})" if epoch is not None else ""))
        plt.tight_layout()
        if params["save_publication_figures"]:
            safe_metric = metric.lower().replace(" ", "_")
            epoch_suffix = f"_epoch_{epoch}" if epoch is not None else ""
            save_current_figure(os.path.join(get_latest_run_dirs("svda")["figures"], f"heatmap_{safe_metric}{epoch_suffix}.png"))
        plt.show()

    def plot_metric_violin_by_layer(df, metric):
        plt.figure(figsize=(10, 5))
        sns.violinplot(data=df, x="layer", y=metric, inner="quartile")
        plt.title(f"{metric} Violin Plot per Layer")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def facet_metric_by_head(df, metric):
        g = sns.FacetGrid(df, col="head", col_wrap=4, height=3.5)
        g.map(sns.lineplot, "epoch", metric)
        g.fig.suptitle(f"{metric} per Head over Epochs", y=1.02)
        plt.tight_layout()
        plt.show()

    fig = plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(baseline_test) + 1), baseline_test, label="Baseline", linewidth=2)
    plt.plot(range(1, len(svda_test) + 1), svda_test, label="SVDA", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("SVDA vs Baseline Accuracy")
    plt.grid(True)
    plt.legend()
    if params["save_publication_figures"]:
        save_current_figure(os.path.join(get_latest_run_dirs("svda")["figures"], "accuracy_comparison.png"))
    plt.show()

    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(baseline_time) + 1), baseline_time, label="Baseline", linestyle="--")
    plt.plot(range(1, len(svda_time) + 1), svda_time, label="SVDA", linestyle="-")
    plt.title("Epoch Duration Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if params["save_publication_figures"]:
        save_current_figure(os.path.join(get_latest_run_dirs("svda")["figures"], "epoch_duration_comparison.png"))
    plt.show()

    svda_epochs = list(range(1, len(svda_entropy) + 1))
    for values, title in [
        (svda_entropy, "Spectral Entropy of $\\Sigma$"),
        (svda_rank, "Effective Rank"),
        (svda_sparsity, "Spectral Sparsity"),
        (svda_selectivity, "Selectivity Index"),
        (svda_robustness, "Attention Perturbation Robustness"),
        (svda_angular, "Angular Alignment (cos θ)"),
    ]:
        plt.figure()
        plt.plot(svda_epochs, values)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    with torch.no_grad():
        last_sigmas = torch.stack([block.attn.sigmas.detach().cpu() for block in svda_model.blocks])
        num_layers = last_sigmas.shape[0]
        num_heads = last_sigmas.shape[1]
        for l in range(num_layers):
            plt.figure(figsize=(12, 3))
            for h in range(num_heads):
                plt.plot(last_sigmas[l, h], label=f"Head {h}")
            plt.title(f"Layer {l} - Per-Head Spectra ($\\Sigma$)")
            plt.xlabel("Dimension")
            plt.ylabel("Singular value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    sample_img, _ = next(iter(test_loader))
    sample_img = sample_img[:1].to(device)
    attn_map = aggregate_attention_maps(
        svda_model,
        sample_img,
        layer_aggregation=params["attention_layer_aggregation"],
        head_aggregation="mean",
    )[0].cpu()
    visualize_attention_overlay(
        attn_map, sample_img[0], patch_size=params["patch_size"], title="Attention Overlay - Aggregated"
    )
    if params["save_publication_figures"]:
        overlay_fig = create_attention_overlay_figure(
            attn_map,
            sample_img[0].cpu(),
            patch_size=params["patch_size"],
            title="SVDA Attention Overlay - Aggregated",
        )
        save_publication_figure(
            overlay_fig,
            os.path.join(get_latest_run_dirs("svda")["figures"], "svda_attention_overlay_aggregated.png"),
        )

    with torch.no_grad():
        sigmas_all = torch.stack([b.attn.sigmas.detach().cpu() for b in svda_model.blocks])
        headwise_entropy_colorbars(sigmas_all)

    print_training_step_stats(base_model)
    print_training_step_stats(svda_model)

    faithfulness_df, faithfulness_mean_std_df = save_faithfulness_outputs(all_run_summaries)
    if params["save_publication_figures"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        deletion_plot_df = faithfulness_df.melt(
            id_vars=["model", "seed", "topk_frac"],
            value_vars=["deletion_top", "deletion_random"],
            var_name="strategy",
            value_name="score",
        )
        insertion_plot_df = faithfulness_df.melt(
            id_vars=["model", "seed", "topk_frac"],
            value_vars=["insertion_top", "insertion_random"],
            var_name="strategy",
            value_name="score",
        )

        sns.lineplot(data=deletion_plot_df, x="topk_frac", y="score", hue="strategy", style="model", marker="o", ax=axes[0])
        axes[0].set_title("Deletion Faithfulness")
        axes[0].set_xlabel("Top-k fraction")
        axes[0].set_ylabel("Confidence drop")
        axes[0].grid(True)

        sns.lineplot(data=insertion_plot_df, x="topk_frac", y="score", hue="strategy", style="model", marker="o", ax=axes[1])
        axes[1].set_title("Insertion Faithfulness")
        axes[1].set_xlabel("Top-k fraction")
        axes[1].set_ylabel("Recovered confidence")
        axes[1].grid(True)

        plt.tight_layout()
        save_publication_figure(
            fig,
            os.path.join(get_latest_run_dirs("svda")["figures"], "faithfulness_comparison_curves.png"),
        )
        save_attention_overlay_comparison(
            base_model, svda_model, test_loader, device, get_latest_run_dirs("svda")["figures"]
        )

    sample_array = np.load(os.path.join(log_dir, f"entropy_layerwise_seed{seed}.npy"))
    num_epochs, num_layers, num_heads = sample_array.shape

    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
        }
    )

    df_records = []
    for epoch in range(num_epochs):
        for layer in range(num_layers):
            for head in range(num_heads):
                df_records.append(
                    {
                        "epoch": epoch,
                        "layer": layer,
                        "head": head,
                        "Spectral entropy": load_metric_array("entropy")[epoch, layer, head],
                        "Effective rank": load_metric_array("rank")[epoch, layer, head],
                        "Angular alignment": load_metric_array("angular_alignment")[epoch, layer, head],
                        "Selectivity index": load_metric_array("selectivity")[epoch, layer, head],
                        "Spectral sparsity": load_metric_array("sparsity")[epoch, layer, head],
                        "Perturbation robustness": load_metric_array("robustness")[epoch, layer],
                    }
                )

    df = pd.DataFrame(df_records)
    df["epoch"] = df["epoch"].astype(int)
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)

    metrics = [
        "Spectral entropy",
        "Effective rank",
        "Angular alignment",
        "Selectivity index",
        "Spectral sparsity",
        "Perturbation robustness",
    ]

    for metric in metrics:
        print(f"\n📊 Plotting {metric}...")
        plot_metric_by_epoch(df, metric)
        plot_metric_by_layer(df, metric)
        plot_metric_histogram(df, metric)
        plot_head_layer_heatmap(df, metric, epoch=df["epoch"].max())
        plot_metric_violin_by_layer(df, metric)
        facet_metric_by_head(df, metric)

    summary = df.groupby("epoch")[metrics].agg(["mean", "std"])
    summary.to_csv(os.path.join(params["log_root"], f"metric_summary_over_epochs_seed{seed}.csv"))
    if params["save_publication_figures"]:
        summary.reset_index().to_csv(
            os.path.join(params["log_root"], f"metric_summary_over_epochs_publication_seed{seed}.csv"),
            index=False,
        )

    summary_rows = []
    for item in all_run_summaries:
        row = {
            "run_name": item["run_name"],
            "model": item["model"],
            "seed": item["seed"],
            "best_acc": item["best_acc"],
            "final_acc": item["final_acc"],
            "mean_epoch_time": item["mean_epoch_time"],
        }
        attention_summary = item.get("attention_summary") or {}
        deletion_summary = item.get("deletion_summary") or {}
        insertion_summary = item.get("insertion_summary") or {}
        efficiency_summary = item.get("efficiency_summary") or []

        for key, value in attention_summary.items():
            row[f"attn_{key}"] = value
        for key, value in deletion_summary.items():
            row[f"deletion_{key}"] = value
        for key, value in insertion_summary.items():
            row[f"insertion_{key}"] = value
        for eff in efficiency_summary:
            batch_size = eff.get("batch_size")
            if batch_size is None:
                continue
            row[f"forward_ms_bs{batch_size}"] = eff.get("forward_ms", np.nan)
            row[f"backward_ms_bs{batch_size}"] = eff.get("backward_ms", np.nan)
            row[f"throughput_img_s_bs{batch_size}"] = eff.get("throughput_img_s", np.nan)
            row[f"peak_mem_mb_bs{batch_size}"] = eff.get("peak_mem_mb", np.nan)
            row[f"macs_bs{batch_size}"] = eff.get("macs", np.nan)
            row[f"params_bs{batch_size}"] = eff.get("params", np.nan)
        summary_rows.append(row)

    experiment_summary_df = pd.DataFrame(summary_rows)
    experiment_summary_df.to_csv(os.path.join(params["log_root"], "experiment_summary.csv"), index=False)

    numeric_cols = [col for col in experiment_summary_df.columns if col not in {"run_name", "model", "seed"}]
    mean_df = experiment_summary_df.groupby("model")[numeric_cols].mean().add_suffix("_mean")
    std_df = experiment_summary_df.groupby("model")[numeric_cols].std().add_suffix("_std")
    summary_mean_std = pd.concat([mean_df, std_df], axis=1).reset_index()
    summary_mean_std.to_csv(
        os.path.join(params["log_root"], "experiment_summary_mean_std.csv"),
        index=False,
    )


if __name__ == "__main__":
    training_outputs = None
    if params["run_training"]:
        training_outputs = run_training_pipeline()
    else:
        raise RuntimeError(
            "run_training=False is not yet supported in this single-file version. Enable run_training or load saved outputs manually."
        )

    if params["run_reporting"]:
        generate_reports(training_outputs)


# === Bulk qualitative visualization export for reviewer-facing figure selection ===

def _to_displayable_image(tensor_img):
    img = tensor_img.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)
    return img


def _attn_map_to_heatmap(attn_map, patch_size):
    N = attn_map.shape[-1] - 1
    side = int(np.sqrt(N))
    heatmap = attn_map[1:, 0].reshape(side, side).detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    from scipy.ndimage import zoom
    return zoom(heatmap, patch_size)


def _save_overlay_panel(image_tensor, attn_map, save_path, title=""):
    image_np = _to_displayable_image(image_tensor)
    heatmap = _attn_map_to_heatmap(attn_map, params["patch_size"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np)
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    ax.set_title(title)
    ax.axis("off")
    save_publication_figure(fig, save_path)


def export_bulk_attention_visualizations(
    base_model,
    svda_model,
    loader,
    current_device,
    save_root,
    max_examples=100,
    require_both_correct=True,
    save_comparison=True,
    save_svda_only=True,
    save_baseline_only=False,
):
    """
    Exports many qualitative examples so the user can manually choose the best ones.
    The reviewer asked for visual-region consistency evidence, so baseline-vs-SVDA
    comparison is exported by default. SVDA-only overlays are also exported for easy figure selection.
    """
    os.makedirs(save_root, exist_ok=True)
    comparison_dir = os.path.join(save_root, "comparison_panels")
    svda_dir = os.path.join(save_root, "svda_only")
    baseline_dir = os.path.join(save_root, "baseline_only")
    raw_dir = os.path.join(save_root, "raw_inputs")
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(svda_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    base_model.eval()
    svda_model.eval()

    exported_rows = []
    exported = 0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(current_device)
            y = y.to(current_device)

            base_logits = base_model(x)
            svda_logits = svda_model(x)
            base_pred = base_logits.argmax(dim=-1)
            svda_pred = svda_logits.argmax(dim=-1)
            base_conf = torch.softmax(base_logits, dim=-1).gather(1, base_pred.unsqueeze(1)).squeeze(1)
            svda_conf = torch.softmax(svda_logits, dim=-1).gather(1, svda_pred.unsqueeze(1)).squeeze(1)

            baseline_agg = aggregate_attention_maps(
                base_model,
                x,
                layer_aggregation=params["attention_layer_aggregation"],
                head_aggregation="mean",
            )
            svda_agg = aggregate_attention_maps(
                svda_model,
                x,
                layer_aggregation=params["attention_layer_aggregation"],
                head_aggregation="mean",
            )
            if baseline_agg is None or svda_agg is None:
                print("Skipping bulk visualization export because aggregated attention maps were not available.")
                return None

            for i in range(x.size(0)):
                if require_both_correct:
                    keep = (base_pred[i].item() == y[i].item()) and (svda_pred[i].item() == y[i].item())
                else:
                    keep = (svda_pred[i].item() == y[i].item())
                if not keep:
                    continue

                idx_str = f"sample_{exported:04d}"
                label_val = int(y[i].item())
                base_pred_val = int(base_pred[i].item())
                svda_pred_val = int(svda_pred[i].item())
                base_conf_val = float(base_conf[i].item())
                svda_conf_val = float(svda_conf[i].item())

                raw_path = os.path.join(raw_dir, f"{idx_str}_input.png")
                plt.imsave(raw_path, _to_displayable_image(x[i]))

                if save_svda_only:
                    svda_path = os.path.join(svda_dir, f"{idx_str}_svda_overlay.png")
                    _save_overlay_panel(
                        x[i].cpu(),
                        svda_agg[i].cpu(),
                        svda_path,
                        title=f"SVDA | y={label_val} pred={svda_pred_val} conf={svda_conf_val:.3f}",
                    )
                else:
                    svda_path = None

                if save_baseline_only:
                    base_path = os.path.join(baseline_dir, f"{idx_str}_baseline_overlay.png")
                    _save_overlay_panel(
                        x[i].cpu(),
                        baseline_agg[i].cpu(),
                        base_path,
                        title=f"Baseline | y={label_val} pred={base_pred_val} conf={base_conf_val:.3f}",
                    )
                else:
                    base_path = None

                if save_comparison:
                    image_np = _to_displayable_image(x[i])
                    base_heatmap = _attn_map_to_heatmap(baseline_agg[i].cpu(), params["patch_size"])
                    svda_heatmap = _attn_map_to_heatmap(svda_agg[i].cpu(), params["patch_size"])
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(image_np)
                    axes[0].set_title(f"Input | y={label_val}")
                    axes[0].axis("off")

                    axes[1].imshow(image_np)
                    axes[1].imshow(base_heatmap, cmap="jet", alpha=0.4)
                    axes[1].set_title(f"Baseline | pred={base_pred_val} conf={base_conf_val:.3f}")
                    axes[1].axis("off")

                    axes[2].imshow(image_np)
                    axes[2].imshow(svda_heatmap, cmap="jet", alpha=0.4)
                    axes[2].set_title(f"SVDA | pred={svda_pred_val} conf={svda_conf_val:.3f}")
                    axes[2].axis("off")

                    comparison_path = os.path.join(comparison_dir, f"{idx_str}_comparison.png")
                    save_publication_figure(fig, comparison_path)
                else:
                    comparison_path = None

                exported_rows.append({
                    "index": exported,
                    "label": label_val,
                    "baseline_pred": base_pred_val,
                    "svda_pred": svda_pred_val,
                    "baseline_conf": base_conf_val,
                    "svda_conf": svda_conf_val,
                    "raw_input_path": raw_path,
                    "baseline_overlay_path": base_path,
                    "svda_overlay_path": svda_path,
                    "comparison_path": comparison_path,
                })
                exported += 1
                if exported >= max_examples:
                    break

            if exported >= max_examples:
                break

    manifest_df = pd.DataFrame(exported_rows)
    manifest_path = os.path.join(save_root, "visualization_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    summary = {
        "export_root": save_root,
        "num_examples": int(exported),
        "require_both_correct": bool(require_both_correct),
        "comparison_dir": comparison_dir if save_comparison else None,
        "svda_only_dir": svda_dir if save_svda_only else None,
        "baseline_only_dir": baseline_dir if save_baseline_only else None,
        "manifest_csv": manifest_path,
    }
    with open(os.path.join(save_root, "visualization_export_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved bulk qualitative visualizations:")
    print(f"  - export root: {save_root}")
    print(f"  - examples exported: {exported}")
    print(f"  - manifest: {manifest_path}")
    if save_comparison:
        print(f"  - comparison panels: {comparison_dir}")
    if save_svda_only:
        print(f"  - SVDA-only overlays: {svda_dir}")
    if save_baseline_only:
        print(f"  - baseline-only overlays: {baseline_dir}")

    return manifest_df


def export_bulk_visualizations_from_saved_runs(
    baseline_summary_path,
    svda_summary_path,
    max_examples=100,
    require_both_correct=True,
    save_comparison=True,
    save_svda_only=True,
    save_baseline_only=False,
    export_subdir="bulk_visualizations",
):
    with open(baseline_summary_path, "r") as f:
        baseline_summary = json.load(f)
    with open(svda_summary_path, "r") as f:
        svda_summary = json.load(f)

    baseline_ckpt = torch.load(baseline_summary["checkpoint_path"], map_location=device)
    svda_ckpt = torch.load(svda_summary["checkpoint_path"], map_location=device)

    base_model = TransformerClassifier({**params, "use_svda": False}).to(device)
    svda_model = TransformerClassifier({**params, "use_svda": True}).to(device)
    base_model.load_state_dict(baseline_ckpt["model_state_dict"])
    svda_model.load_state_dict(svda_ckpt["model_state_dict"])
    base_model.eval()
    svda_model.eval()

    _, local_test_loader = build_dataloaders(params)

    export_root = os.path.join(params["log_root"], export_subdir)
    return export_bulk_attention_visualizations(
        base_model=base_model,
        svda_model=svda_model,
        loader=local_test_loader,
        current_device=device,
        save_root=export_root,
        max_examples=max_examples,
        require_both_correct=require_both_correct,
        save_comparison=save_comparison,
        save_svda_only=save_svda_only,
        save_baseline_only=save_baseline_only,
    )