"""
Kohonen Self-Organizing Map (SOM) — reference implementation.

Includes:
  - SOM class (NumPy)
  - Image compression / decompression via vector quantization
  - CLI: train, compress, decompress, evaluate

Usage:
    python som.py compress input.png --block 4 --codebook 64 --iters 2000 \\
        --out compressed.npz --reconstruct reconstructed.png

    python som.py decompress compressed.npz --out reconstructed.png

    python som.py demo-colors --grid 30 --iters 5000

Dependencies: numpy, pillow (PIL), matplotlib (optional, for demos)
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# SOM core
# ---------------------------------------------------------------------------
@dataclass
class SOMConfig:
    grid_h: int = 10
    grid_w: int = 10
    input_dim: int = 3
    initial_lr: float = 0.5
    initial_radius: float = None  # default = max(grid)/2
    n_iters: int = 1000
    seed: int = 42


class SOM:
    """Kohonen Self-Organizing Map with Gaussian neighborhood and exponential decay."""

    def __init__(self, cfg: SOMConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.weights = rng.random((cfg.grid_h, cfg.grid_w, cfg.input_dim), dtype=np.float32)
        # 2D coordinate grid for vectorized neighborhood
        ys, xs = np.meshgrid(np.arange(cfg.grid_h), np.arange(cfg.grid_w), indexing="ij")
        self.coords = np.stack([ys, xs], axis=-1).astype(np.float32)
        self.r0 = cfg.initial_radius or max(cfg.grid_h, cfg.grid_w) / 2.0
        self.lr0 = cfg.initial_lr
        self.time_const = cfg.n_iters / math.log(self.r0) if self.r0 > 1 else cfg.n_iters

    # -- neighborhood schedules ---------------------------------------------
    def _radius(self, t: int) -> float:
        return self.r0 * math.exp(-t / self.time_const)

    def _lr(self, t: int) -> float:
        return self.lr0 * math.exp(-t / self.cfg.n_iters)

    # -- core ops -----------------------------------------------------------
    def bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """Return (row, col) of Best Matching Unit for input x."""
        diff = self.weights - x  # broadcast
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        idx = np.argmin(d2)
        return divmod(int(idx), self.cfg.grid_w)

    def step(self, x: np.ndarray, t: int) -> Tuple[int, int]:
        """One training step on a single input."""
        bmu_r, bmu_c = self.bmu(x)
        radius = self._radius(t)
        lr = self._lr(t)
        # squared distance in grid space from BMU
        dr = self.coords[..., 0] - bmu_r
        dc = self.coords[..., 1] - bmu_c
        d2 = dr * dr + dc * dc
        influence = np.exp(-d2 / (2.0 * radius * radius))[..., None]  # (h,w,1)
        self.weights += lr * influence * (x - self.weights)
        return bmu_r, bmu_c

    def train(self, data: np.ndarray, verbose: bool = False) -> None:
        rng = np.random.default_rng(self.cfg.seed + 1)
        n = len(data)
        for t in range(self.cfg.n_iters):
            x = data[rng.integers(0, n)]
            self.step(x, t)
            if verbose and (t + 1) % max(1, self.cfg.n_iters // 10) == 0:
                qe = self.quantization_error(data[: min(2000, n)])
                print(f"  iter {t+1}/{self.cfg.n_iters}  "
                      f"lr={self._lr(t):.4f}  r={self._radius(t):.3f}  QE={qe:.4f}")

    # -- evaluation ---------------------------------------------------------
    def quantization_error(self, data: np.ndarray) -> float:
        """Mean Euclidean distance from each input to its BMU weight."""
        flat_w = self.weights.reshape(-1, self.cfg.input_dim)
        # distance matrix in chunks to avoid huge memory
        total = 0.0
        chunk = 1024
        for i in range(0, len(data), chunk):
            x = data[i:i + chunk]
            d = np.linalg.norm(x[:, None, :] - flat_w[None, :, :], axis=-1)
            total += d.min(axis=1).sum()
        return float(total / len(data))

    def topographic_error(self, data: np.ndarray) -> float:
        """Fraction of inputs whose 1st and 2nd BMUs are NOT grid-adjacent."""
        flat_w = self.weights.reshape(-1, self.cfg.input_dim)
        gw = self.cfg.grid_w
        bad = 0
        chunk = 1024
        for i in range(0, len(data), chunk):
            x = data[i:i + chunk]
            d = np.linalg.norm(x[:, None, :] - flat_w[None, :, :], axis=-1)
            top2 = np.argpartition(d, 2, axis=1)[:, :2]
            # ensure ordering
            order = np.argsort(np.take_along_axis(d, top2, axis=1), axis=1)
            top2 = np.take_along_axis(top2, order, axis=1)
            for a, b in top2:
                ar, ac = divmod(int(a), gw)
                br, bc = divmod(int(b), gw)
                if abs(ar - br) > 1 or abs(ac - bc) > 1:
                    bad += 1
        return bad / len(data)

    def u_matrix(self) -> np.ndarray:
        """U-matrix: average distance from each neuron's weights to its neighbors."""
        h, w, _ = self.weights.shape
        u = np.zeros((h, w), dtype=np.float32)
        for r in range(h):
            for c in range(w):
                acc, n = 0.0, 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            acc += float(np.linalg.norm(self.weights[r, c] - self.weights[rr, cc]))
                            n += 1
                u[r, c] = acc / n
        return u


# ---------------------------------------------------------------------------
# Image compression via SOM-as-codebook
# ---------------------------------------------------------------------------
def image_to_blocks(img: np.ndarray, block: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Split HxWxC image into (N, block*block*C) vectors. Pads edges if needed."""
    h, w, c = img.shape
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    H, W, _ = img.shape
    blocks = (img.reshape(H // block, block, W // block, block, c)
                 .swapaxes(1, 2)
                 .reshape(-1, block * block * c))
    return blocks.astype(np.float32) / 255.0, (H, W)


def blocks_to_image(blocks: np.ndarray, shape: Tuple[int, int], block: int, c: int) -> np.ndarray:
    H, W = shape
    arr = (blocks.reshape(H // block, W // block, block, block, c)
                 .swapaxes(1, 2)
                 .reshape(H, W, c))
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def compress_image(path: str, block: int, codebook: int, iters: int, seed: int = 42):
    if Image is None:
        raise RuntimeError("Pillow is required: pip install pillow")
    img = np.array(Image.open(path).convert("RGB"))
    h, w, c = img.shape
    blocks, padded_shape = image_to_blocks(img, block)
    # Choose grid close to square totaling >= codebook
    side = int(math.ceil(math.sqrt(codebook)))
    cfg = SOMConfig(grid_h=side, grid_w=side, input_dim=blocks.shape[1],
                    n_iters=iters, seed=seed)
    som = SOM(cfg)
    print(f"Training SOM {side}x{side} on {len(blocks)} blocks of dim {blocks.shape[1]}...")
    som.train(blocks, verbose=True)
    # Encode: BMU index per block
    flat_w = som.weights.reshape(-1, blocks.shape[1])
    indices = np.empty(len(blocks), dtype=np.int32)
    chunk = 4096
    for i in range(0, len(blocks), chunk):
        d = np.linalg.norm(blocks[i:i+chunk, None, :] - flat_w[None, :, :], axis=-1)
        indices[i:i+chunk] = d.argmin(axis=1)
    return {
        "indices": indices,
        "codebook": flat_w.astype(np.float32),
        "block": block,
        "channels": c,
        "padded_shape": padded_shape,
        "original_shape": (h, w, c),
    }, som


def decompress(payload: dict) -> np.ndarray:
    block = int(payload["block"])
    c = int(payload["channels"])
    codebook = payload["codebook"]
    indices = payload["indices"]
    blocks = codebook[indices]
    img = blocks_to_image(blocks, tuple(payload["padded_shape"]), block, c)
    h, w, _ = payload["original_shape"]
    return img[:h, :w, :]


def metrics(original: np.ndarray, reconstructed: np.ndarray, payload: dict) -> dict:
    mse = float(np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2))
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    h, w, c = original.shape
    original_bits = h * w * c * 8
    bits_per_index = max(1, int(math.ceil(math.log2(len(payload["codebook"])))))
    n_indices = len(payload["indices"])
    codebook_bits = payload["codebook"].size * 8  # 8-bit per channel after quant; conservative
    compressed_bits = n_indices * bits_per_index + codebook_bits
    return {
        "MSE": mse,
        "PSNR_dB": psnr,
        "compression_ratio": original_bits / compressed_bits,
        "bits_per_pixel": compressed_bits / (h * w),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_compress(args):
    payload, som = compress_image(args.input, args.block, args.codebook, args.iters, args.seed)
    np.savez_compressed(args.out, **payload)
    if args.reconstruct:
        rec = decompress(payload)
        Image.fromarray(rec).save(args.reconstruct)
        original = np.array(Image.open(args.input).convert("RGB"))
        m = metrics(original, rec, payload)
        print("\n=== Metrics ===")
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")


def cmd_decompress(args):
    payload = dict(np.load(args.input, allow_pickle=False))
    rec = decompress(payload)
    Image.fromarray(rec).save(args.out)
    print(f"Wrote {args.out}")


def cmd_demo_colors(args):
    """Train a SOM on random RGB samples and dump the resulting color map."""
    rng = np.random.default_rng(0)
    data = rng.random((args.samples, 3), dtype=np.float32)
    cfg = SOMConfig(grid_h=args.grid, grid_w=args.grid, input_dim=3,
                    n_iters=args.iters, seed=0)
    som = SOM(cfg)
    print("Training color SOM...")
    som.train(data, verbose=True)
    img = (np.clip(som.weights, 0, 1) * 255).astype(np.uint8)
    if Image is not None:
        Image.fromarray(img).resize((args.grid * 16, args.grid * 16),
                                    Image.NEAREST).save(args.out)
        print(f"Saved color map to {args.out}")
    qe = som.quantization_error(data)
    te = som.topographic_error(data[:1000])
    print(f"QE={qe:.4f}  TE={te:.4f}")


def main():
    p = argparse.ArgumentParser(description="Kohonen SOM toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compress", help="Compress an image with a SOM codebook")
    pc.add_argument("input")
    pc.add_argument("--block", type=int, default=4)
    pc.add_argument("--codebook", type=int, default=64)
    pc.add_argument("--iters", type=int, default=2000)
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--out", default="compressed.npz")
    pc.add_argument("--reconstruct", default="reconstructed.png")
    pc.set_defaults(func=cmd_compress)

    pd = sub.add_parser("decompress", help="Decompress a .npz produced by `compress`")
    pd.add_argument("input")
    pd.add_argument("--out", default="reconstructed.png")
    pd.set_defaults(func=cmd_decompress)

    pdc = sub.add_parser("demo-colors", help="Train color SOM toy example")
    pdc.add_argument("--grid", type=int, default=30)
    pdc.add_argument("--iters", type=int, default=5000)
    pdc.add_argument("--samples", type=int, default=2000)
    pdc.add_argument("--out", default="color_map.png")
    pdc.set_defaults(func=cmd_demo_colors)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
