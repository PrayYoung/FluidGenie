from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from jaxtyping import Array, Float

import numpy as np


def update_running_mean_var(
    mean: Float[Array, "c"] | None,
    var: Float[Array, "c"] | None,
    count: int,
    batch_mean: Float[Array, "c"],
    batch_var: Float[Array, "c"],
    batch_count: int,
):
    if count == 0:
        return batch_mean, batch_var, batch_count
    delta = batch_mean - mean
    total = count + batch_count
    new_mean = mean + delta * (batch_count / total)
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta * delta * (count * batch_count / total)
    new_var = m2 / total
    return new_mean, new_var, total


def compute_stats(data_dir: str) -> tuple[Float[Array, "c"], Float[Array, "c"]]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    assert files, f"No npy found in {data_dir}"

    mean = None
    var = None
    count = 0

    for f in files:
        fields = np.load(f, mmap_mode="r")  # [T,H,W,C]
        x = fields.astype(np.float64)
        # flatten T,H,W but keep C
        x = x.reshape(-1, x.shape[-1])
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        if mean is None:
            mean = batch_mean
            var = batch_var
            count = batch_count
        else:
            mean, var, count = update_running_mean_var(mean, var, count, batch_mean, batch_var, batch_count)

    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True, help="Output .npz path for mean/std")
    args = ap.parse_args()

    mean, std = compute_stats(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean=mean, std=std)
    print(f"Saved stats to {out}")
    print("mean:", mean)
    print("std:", std)


if __name__ == "__main__":
    main()
