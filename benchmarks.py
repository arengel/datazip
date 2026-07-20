"""Benchmark DataZip vs pickle: write time, read time, and file size.

Run from the repo root:

    python docs/benchmarks.py

Saves figures to docs/images/ and prints a markdown results table.
"""

from __future__ import annotations

import io
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt  # ty:ignore[unresolved-import]
import numpy as np
import pandas as pd

from datazip.core import DataZip

IMAGES_DIR = Path(__file__).parent / "images"
REPEATS = 7


# ---------------------------------------------------------------------------
# Test-data factories
# ---------------------------------------------------------------------------


def make_nested_dict(depth: int, breadth: int) -> dict:
    """Return a recursively nested dict *depth* levels deep.

    With breadth=2 this creates 2**10 = 1,024 leaf nodes.
    """
    if depth == 0:
        return {
            "values": list(range(50)),
            "label": "leaf",
            "score": 3.14159,
            "tags": ["alpha", "beta", "gamma"],
        }
    return {f"branch_{i}": make_nested_dict(depth - 1, breadth) for i in range(breadth)}


def make_mixed_object() -> dict:
    """Return a dict that mimics a realistic analysis result object.

    Contains:
    - one large DataFrame  (5M rows × 20 cols)
    - two medium DataFrames (50k rows × 10 cols each)
    - a nested config dict
    - a list of category labels
    - a description string
    """  # noqa: RUF002
    rng = np.random.default_rng(42)

    def _df(rows, cols):
        half = cols // 2
        d = {f"float_{i}": rng.standard_normal(rows) for i in range(half)}
        d |= {f"int_{i}": rng.integers(0, 10_000, rows) for i in range(half)}
        return pd.DataFrame(d)

    return {
        "results": _df(5_000_000, 20),
        "train_metrics": _df(50_000, 10),
        "val_metrics": _df(50_000, 10),
        "config": {
            "model": "GradientBoosting",
            "hyperparams": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05},
            "feature_groups": {
                "numeric": [f"float_{i}" for i in range(10)],
                "integer": [f"int_{i}" for i in range(10)],
            },
            "cv_folds": 5,
        },
        "category_labels": [f"class_{i}" for i in range(20)],
        "description": (
            "Gradient boosting model trained on synthetic data. "
            "Metrics recorded at each epoch for train and validation splits."
        ),
    }


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _time(fn, repeats: int) -> float:
    """Return the trimmed mean wall-clock seconds for *fn()* over *repeats* runs."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return float(np.mean(times[:-1]))  # drop the slowest (JIT / GC noise)


def bench_pickle(obj, repeats: int = REPEATS) -> dict[str, float]:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    size_mb = buf.tell() / 1e6
    data = buf.getvalue()

    write_s = _time(lambda: pickle.dump(obj, io.BytesIO()), repeats)
    read_s = _time(lambda: pickle.loads(data), repeats)

    return {"write_s": write_s * 1e3, "read_s": read_s * 1e3, "size_mb": size_mb}


def bench_partial_read(obj: dict, key: str, repeats: int = REPEATS) -> dict[str, float]:
    """Benchmark reading a single key: pickle must deserialize everything; DataZip does not.

    With pickle, accessing any one key requires deserializing the entire blob —
    including every large DataFrame stored in it.  With DataZip, only the
    requested entry is deserialized; the binary files (Parquet, npy, …) for all
    other entries are never read.
    """  # noqa: E501, W505
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    pkl_data = buf.getvalue()

    dz_buf = io.BytesIO()
    with DataZip(dz_buf, "w") as z:
        for k, v in obj.items():
            z[k] = v
    dz_data = dz_buf.getvalue()

    pkl_s = _time(lambda: pickle.loads(pkl_data)[key], repeats)

    def _dz_read():
        with DataZip(io.BytesIO(dz_data), "r") as z:
            _ = z[key]

    dz_s = _time(_dz_read, repeats)
    return {"pickle_s": pkl_s * 1e3, "datazip_s": dz_s * 1e3}


def bench_datazip_dict(obj: dict, repeats: int = REPEATS) -> dict[str, float]:
    """Benchmark DataZip storing each top-level key of *obj* separately.

    This is the natural DataZip usage pattern for mixed objects: each named
    component is stored as its own entry in the archive.
    """
    buf = io.BytesIO()
    with DataZip(buf, "w") as z:
        for k, v in obj.items():
            z[k] = v
    size_mb = buf.tell() / 1e6
    data = buf.getvalue()

    def _write():
        DataZip.dump(obj, io.BytesIO())

    def _read():
        _ = DataZip.load(io.BytesIO(data))

    write_s = _time(_write, repeats)
    read_s = _time(_read, repeats)

    return {"write_s": write_s * 1e3, "read_s": read_s * 1e3, "size_mb": size_mb}


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

_PALETTE = {"pickle": "#4C72B0", "DataZip": "#DD8452"}
_BAR_WIDTH = 0.35


def _bar_chart(
    ax: plt.Axes,
    labels: list[str],
    pickle_vals: list[float],
    dz_vals: list[float],
    ylabel: str,
    title: str,
) -> None:
    x = np.arange(len(labels))
    bars_pkl = ax.bar(
        x - _BAR_WIDTH / 2,
        pickle_vals,
        _BAR_WIDTH,
        label="pickle",
        color=_PALETTE["pickle"],
    )
    bars_dz = ax.bar(
        x + _BAR_WIDTH / 2,
        dz_vals,
        _BAR_WIDTH,
        label="DataZip",
        color=_PALETTE["DataZip"],
    )
    # value labels on bars
    for bar in [*bars_pkl, *bars_dz]:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.02,
            f"{h:.2f}" if h < 10 else f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)  # noqa: FBT003
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def make_partial_figure(result: dict[str, float], key: str) -> None:
    """Save a two-bar chart for the single-item read benchmark."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    labels = ["pickle\n(full deserialize)", "DataZip\n(lazy, 1 entry)"]
    vals = [result["pickle_s"], result["datazip_s"]]
    colors = [_PALETTE["pickle"], _PALETTE["DataZip"]]
    bars = ax.bar(labels, vals, color=colors, width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.02,
            f"{h:.3f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        f"Single-item read (key='{key}')\npickle vs DataZip", fontweight="bold"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)  # noqa: FBT003
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = IMAGES_DIR / "bench_partial_read.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def make_figures(results: dict[str, dict[str, dict[str, float]]]) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    display_labels = list(results.keys())
    metrics = [
        ("write_s", "Time (ms)", "Write Time"),
        ("read_s", "Time (ms)", "Read Time"),
        ("size_mb", "Size (MB)", "File Size"),
    ]
    filenames = ["bench_write.png", "bench_read.png", "bench_size.png"]

    for (key, ylabel, title), fname in zip(metrics, filenames, strict=True):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        pkl_vals = [results[l]["pickle"][key] for l in display_labels]  # noqa: E741
        dz_vals = [results[l]["datazip"][key] for l in display_labels]  # noqa: E741
        _bar_chart(ax, display_labels, pkl_vals, dz_vals, ylabel, title)
        fig.tight_layout()
        path = IMAGES_DIR / fname
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    depth = 10
    breadth = 2
    print(f"\n── Nested Dict (depth={depth}, {breadth**10} leaves) ──")
    nested = make_nested_dict(depth, breadth)

    print("  pickle  ...", end=" ", flush=True)
    nested_pkl = bench_pickle(nested)
    r = nested_pkl
    print(
        f"write={r['write_s']:.3f}s  read={r['read_s']:.3f}s  "
        f"size={r['size_mb']:.2f} MB"
    )

    print("  DataZip ...", end=" ", flush=True)
    nested_dz = bench_datazip_dict(nested)
    r = nested_dz
    print(
        f"write={r['write_s']:.3f}s  read={r['read_s']:.3f}s  "
        f"size={r['size_mb']:.2f} MB"
    )

    print("\n── Mixed Object (large + 2 medium DataFrames, dicts, lists, string) ──")
    mixed = make_mixed_object()

    print("  pickle  ...", end=" ", flush=True)
    mixed_pkl = bench_pickle(mixed)
    r = mixed_pkl
    print(
        f"write={r['write_s']:.3f}ms  read={r['read_s']:.3f}ms  "
        f"size={r['size_mb']:.2f} MB"
    )

    print("  DataZip ...", end=" ", flush=True)
    mixed_dz = bench_datazip_dict(mixed)
    r = mixed_dz
    print(
        f"write={r['write_s']:.3f}ms  read={r['read_s']:.3f}ms  "
        f"size={r['size_mb']:.2f} MB"
    )

    TARGET_KEY = "config"  # noqa: N806
    print(f"\n── Single-item read (key='{TARGET_KEY}' from mixed object) ──")
    print(
        f"  pickle must deserialize the entire blob (all DataFrames included).\n"
        f"  DataZip only decodes the '{TARGET_KEY}' entry; other files are never read."
    )
    partial = bench_partial_read(mixed, TARGET_KEY)
    speedup = partial["pickle_s"] / partial["datazip_s"]
    print(
        f"  pickle={partial['pickle_s']:.0f}ms  "
        f"DataZip={partial['datazip_s'] * 1e3:.0f}us  "
        f"speedup={speedup:.1f}×"  # noqa: RUF001
    )

    results = {
        "Nested Dict\n(depth=10, 1 k leaves)": {
            "pickle": nested_pkl,
            "datazip": nested_dz,
        },
        "Mixed Object\n(3 DataFrames + metadata)": {
            "pickle": mixed_pkl,
            "datazip": mixed_dz,
        },
    }

    print("\nGenerating figures ...")
    make_figures(results)
    make_partial_figure(partial, TARGET_KEY)

    print("\n### Markdown results table\n")
    hdr = (
        "| Benchmark | Write: pickle | Write: DataZip | Read: pickle "
        "| Read: DataZip | Size: pickle | Size: DataZip |"
    )
    sep = "|---|---|---|---|---|---|---|"
    print(hdr)
    print(sep)
    for label, r in results.items():
        name = label.replace("\n", " ")
        p, d = r["pickle"], r["datazip"]
        print(
            f"| {name} "
            f"| {p['write_s']:.1f} ms | {d['write_s']:.1f} ms "
            f"| {p['read_s']:.1f} ms | {d['read_s']:.1f} ms "
            f"| {p['size_mb']:.2f} MB | {d['size_mb']:.2f} MB |"
        )


if __name__ == "__main__":
    main()
