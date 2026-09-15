"""Generate a confusion matrix on the PTB-XL test set (per-record evaluation).

Run from the root of the ecg-ml-stream repository INSIDE the dev container
(`docker run -it ecg-ml-stream-dev`):

    python scripts/gen_confusion_matrix.py \
        --model models/run_20260514_004115/best_model.pt \
        --output ../EE-dyplom/rysunki/macierz_pomylek.pdf

Copyright 2026 Mateusz Golebiewski
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from ecg_ml_stream.config import cfg
from ecg_ml_stream.dataset.ecg_dataset import ECGDataset
from ecg_ml_stream.ml.model import ECGClassifier
from ecg_ml_stream.utils.constants import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=str,
        default="models/run_20260514_004115/best_model.pt",
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../EE-dyplom/rysunki/macierz_pomylek.pdf",
        help="Output PDF path (a PNG is saved alongside it).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu/cuda/mps). Auto-detected by default.",
    )
    return parser.parse_args()


def main() -> None:
    """Run per-record inference with soft voting and save the confusion matrix."""
    args = parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"[i] Device: {device}")

    print(f"[i] Loading model from {args.model}")
    classifier = ECGClassifier(model_path=args.model, device=str(device))

    print(f"[i] Loading test set from {cfg.data.path}")
    test_dataset = ECGDataset(
        data_path=cfg.data.path,
        sampling_rate=cfg.data.sampling_rate,
        split="test",
    )
    record_ids = list(test_dataset.records.index)
    print(f"[i] Test records: {len(record_ids)}")

    print("[i] Running inference (per-record, soft voting)...")
    all_preds: list[int] = []
    all_targets: list[int] = []

    for i, ecg_id in enumerate(record_ids):
        windows_tensor, label = test_dataset.get_record_windows(ecg_id)
        windows_tensor = windows_tensor.to(device)

        result = classifier.predict_windows(windows_tensor)
        all_preds.append(result["class_idx"])
        all_targets.append(label)

        if (i + 1) % 200 == 0:
            print(f"  record {i + 1}/{len(record_ids)}")

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    print(f"[i] Predictions: {len(y_pred)}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    print("\nConfusion matrix (counts):")
    print(cm)
    print("\nConfusion matrix (row-normalized):")
    print(np.round(cm_norm, 3))

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall — row-normalized values")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix — test set (per-record, soft voting)")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            value = cm_norm[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(
                j, i, f"{value:.2f}\n({cm[i, j]})",
                ha="center", va="center", color=color, fontsize=9,
            )

    out_pdf = Path(args.output)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Saved: {out_pdf}")
    print(f"[OK] Saved: {out_png}")


if __name__ == "__main__":
    main()
