from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(run_dir: Path, out_path: Path | None = None):
    d = np.load(run_dir / "metrics.npz")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(d["epoch"], d["train_loss"], label="train")
    ax1.plot(d["epoch"], d["test_loss"], label="test")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("loss")

    ax2.plot(d["epoch"], d["train_acc"], label="train")
    ax2.plot(d["epoch"], d["test_acc"], label="test")
    ax2.set_xscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.set_title("accuracy")

    fig.suptitle(f"grokking: (a + b) mod p — {run_dir.name}")
    fig.tight_layout()

    out_path = out_path or (run_dir / "training_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
