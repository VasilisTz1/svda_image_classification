import os
import time
import json
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ptflops import get_model_complexity_info

# =========================
# Match your CIFAR-10 setup
# =========================
params = {
    "dataset": "CIFAR10",
    "image_size": 32,
    "patch_size": 4,
    "embed_dim": 256,
    "num_heads": 4,
    "num_layers": 4,
    "num_classes": 10,
    "batch_size": 64,
    "epochs": 100,
    "lr": 3e-4,
    "lambda_ortho": 1e-3,
    "use_svda": True
}

# -------------------------
# Model definition (as in your training script)
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class StandardMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
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
        nn.init.normal_(self.sigmas, mean=1.0, std=0.1)

        self.attn_weights = None
        self.angular_alignments = None

    def forward(self, x):
        B, N, _ = x.size()
        Q = self.Wq(x).view(B, N, self.h, self.dk).transpose(1, 2)
        K = self.Wk(x).view(B, N, self.h, self.dk).transpose(1, 2)
        V = self.Wv(x).view(B, N, self.h, self.dk).transpose(1, 2)

        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)

        # diagonal proxy alignment, as in your training script
        cos_sim = torch.sum(Q * K, dim=-1).mean(dim=-1).mean(dim=0)  # (H,)
        self.angular_alignments = cos_sim.detach()

        weighted_Q = Q * self.sigmas.unsqueeze(0).unsqueeze(2)
        scores = torch.matmul(weighted_Q, K.transpose(-1, -2)) / np.sqrt(self.dk)
        A = F.softmax(scores, dim=-1)
        self.attn_weights = A.detach()

        Z = torch.matmul(A, V).transpose(1, 2).reshape(B, N, self.dim)
        return self.Wo(Z)

    def orthogonality_loss(self):
        loss_q = ((self.Wq.weight @ self.Wq.weight.T - torch.eye(self.dim, device=self.Wq.weight.device)) ** 2).mean()
        loss_k = ((self.Wk.weight @ self.Wk.weight.T - torch.eye(self.dim, device=self.Wk.weight.device)) ** 2).mean()
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
        self.embed = PatchEmbedding(3, params["patch_size"], params["embed_dim"], params["image_size"])
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

# -------------------------
# Device helpers
# -------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        # available on recent torch builds; safe-guard
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

# -------------------------
# Profiling
# -------------------------
@torch.no_grad()
def measure_inference(model, device, batch_size, iters=300, warmup=100):
    model = model.to(device).eval()

    x = torch.randn(batch_size, 3, 32, 32, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    sync_device(device)

    # Peak memory (CUDA only)
    peak_mib = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        sync_device(device)

    # Timing
    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(iters):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            times_ms.append(starter.elapsed_time(ender))
        mean_ms = float(np.mean(times_ms))
        p50_ms = float(np.percentile(times_ms, 50))
        p90_ms = float(np.percentile(times_ms, 90))
        p99_ms = float(np.percentile(times_ms, 99))
        peak_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p90_ms": p90_ms, "p99_ms": p99_ms, "peak_mem_mib": peak_mib}

    # CPU/MPS timing (wall-clock)
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model(x)
        sync_device(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    p50_ms = float(np.percentile(times_ms, 50))
    p90_ms = float(np.percentile(times_ms, 90))
    p99_ms = float(np.percentile(times_ms, 99))
    return {"mean_ms": mean_ms, "p50_ms": p50_ms, "p90_ms": p90_ms, "p99_ms": p99_ms, "peak_mem_mib": peak_mib}

def measure_macs_params(model):
    # ptflops expects CPU model
    model_cpu = model.cpu().eval()
    with torch.no_grad():
        macs, nparams = get_model_complexity_info(
            model_cpu, (3, 32, 32),
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
    # ptflops returns MACs as scalar count
    return int(macs), int(nparams)

def load_model(ckpt_path, use_svda):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = TransformerClassifier({**params, "use_svda": use_svda})
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model

def pretty_macs(macs):
    if macs >= 1e9:
        return f"{macs/1e9:.2f} GMAC"
    if macs >= 1e6:
        return f"{macs/1e6:.2f} MMAC"
    return str(macs)

def main():
    device = pick_device()
    print("Device:", device)

    # Set these filenames as in your folder listing
    baseline_ckpt = "baseline_results_cifar10.pth"
    svda_ckpt = "svda_results_cifar10.pth"

    assert os.path.exists(baseline_ckpt), f"Missing {baseline_ckpt}"
    assert os.path.exists(svda_ckpt), f"Missing {svda_ckpt}"

    baseline = load_model(baseline_ckpt, use_svda=False)
    svda = load_model(svda_ckpt, use_svda=True)

    # MACs/Params
    base_macs, base_params = measure_macs_params(baseline)
    svda_macs, svda_params = measure_macs_params(svda)

    # Inference profiling (report both batch=1 and batch=64 like common practice)
    batches = [1, 64]
    results = {
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "baseline": {"params": base_params, "macs": base_macs, "profiles": {}},
        "svda": {"params": svda_params, "macs": svda_macs, "profiles": {}},
    }

    for bsz in batches:
        results["baseline"]["profiles"][f"batch_{bsz}"] = measure_inference(baseline, device, bsz)
        results["svda"]["profiles"][f"batch_{bsz}"] = measure_inference(svda, device, bsz)

    # Print a compact table
    print("\n=== CIFAR-10 Inference Profiling ===")
    print(f"MACs/Params baseline: {pretty_macs(base_macs)} | {base_params:,} params")
    print(f"MACs/Params SVDA:     {pretty_macs(svda_macs)} | {svda_params:,} params")

    for bsz in batches:
        bkey = f"batch_{bsz}"
        b0 = results["baseline"]["profiles"][bkey]
        s0 = results["svda"]["profiles"][bkey]
        print(f"\nBatch size = {bsz}")
        if device.type == "cuda":
            print(f"Baseline: mean {b0['mean_ms']:.3f} ms | p50 {b0['p50_ms']:.3f} | p90 {b0['p90_ms']:.3f} | p99 {b0['p99_ms']:.3f} | peak {b0['peak_mem_mib']:.1f} MiB")
            print(f"SVDA:     mean {s0['mean_ms']:.3f} ms | p50 {s0['p50_ms']:.3f} | p90 {s0['p90_ms']:.3f} | p99 {s0['p99_ms']:.3f} | peak {s0['peak_mem_mib']:.1f} MiB")
        else:
            print(f"Baseline: mean {b0['mean_ms']:.3f} ms | p50 {b0['p50_ms']:.3f} | p90 {b0['p90_ms']:.3f} | p99 {b0['p99_ms']:.3f}")
            print(f"SVDA:     mean {s0['mean_ms']:.3f} ms | p50 {s0['p50_ms']:.3f} | p90 {s0['p90_ms']:.3f} | p99 {s0['p99_ms']:.3f}")
            print("(Peak inference memory is reported only on CUDA.)")

    # Save JSON for paper / appendix
    with open("cifar10_inference_profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: cifar10_inference_profile.json")

if __name__ == "__main__":
    main()