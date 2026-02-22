"""Training module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from ecg.dataset.ecg_dataset import ECGAugmentation, ECGDataset
from ecg.ml.metrics import (
    AverageMeter,
    EarlyStopping,
    MetricsCalculator,
    format_metrics,
    save_training_history,
)
from ecg.ml.model import ECGClassifier, ResNet1D
from ecg.utils.constants import CLASS_NAMES
from ecg.utils.helpers import count_parameters, get_device, save_checkpoint, set_seed, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments.

    """
    parser = argparse.ArgumentParser(description="Train ResNet1D on PTX-XL ECG dataset")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ptb-xl-1.0.3",
        help="Path to the PTX-XL dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory for saved models and logs",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=[100, 500],
        help="Sampling rate for ECG signals (Hz)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=2.5,
        help="Window size for training (seconds)",
    )
    parser.add_argument(
        "--window-stride",
        type=float,
        default=1.25,
        help="Stride between windows (seconds)",
    )
    parser.add_argument(
        "--base-filters",
        type=int,
        default=64,
        help="Number of filters in the first ResNet block",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate in the ResNet1D model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2) for the AdamW optimizer",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )

    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Run one training epoch over the full training set.

    Args:
        model (nn.Module): The ResNet1D model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number for logging purposes.

    Returns:
        dict: Dictionary of training metrics for this epoch.

    """
    model.train()
    metrics = MetricsCalculator()
    loss_meter = AverageMeter()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_meter.update(loss.item(), data.size(0))
        metrics.update(output, target, loss.item())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    results = metrics.compute()
    results["loss"] = loss_meter.avg
    return results


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> dict:
    """Evaluate the model on a validation/test DataLoader.

    Args:
        model (nn.Module): The ResNet1D model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation/test set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on (CPU or GPU).
        epoch (int): Current epoch number for logging purposes.

    Returns:
        dict: Dictionary of validation metrics for this epoch.

    """
    model.eval()
    metrics = MetricsCalculator()
    loss_meter = AverageMeter()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        loss_meter.update(loss.item(), data.size(0))
        metrics.update(output, target, loss.item())

        pbar.set_postfix({"loss": f"{loss_meter.avg:.4f}"})

    results = metrics.compute()
    results["loss"] = loss_meter.avg
    return results


@torch.no_grad()
def evaluate_with_voting(
    model: nn.Module,
    dataset: ECGDataset,
    device: torch.device,
    num_samples: int | None = None,
) -> dict:
    """Evaluate accuracy and F1 using 7-window soft voting per record.

    Args:
        model (nn.Module): The ResNet1D model to evaluate.
        dataset (ECGDataset): ECGDataset providing `get_record_windows` method.
        device (torch.device): Device to run the evaluation on.
        num_samples (int | None): Number of records to evaluate. If None, evaluate all records.

    Returns:
        dict: Dictionary containing `voting_accuracy`, `voting_f1_macro`, `total_records`.

    """
    model.eval()
    classifier = ECGClassifier()
    classifier.model = model
    classifier.device = device

    predictions = []
    targets = []

    record_ids = dataset.records.index.tolist()
    if num_samples:
        record_ids = record_ids[:num_samples]

    for ecg_id in tqdm(record_ids, desc="Voting evaluation"):
        windows, label = dataset.get_record_windows(ecg_id)
        result = classifier.predict_windows(windows)
        predictions.append(result["class_idx"])
        targets.append(label)

    n_correct = sum(p == t for p, t in zip(predictions, targets, strict=True))
    accuracy = n_correct / len(targets)
    f1_macro = f1_score(targets, predictions, average="macro")

    return {
        "voting_accuracy": accuracy,
        "voting_f1_macro": f1_macro,
        "total_records": len(targets),
    }


def voting_prediction(
    predictions: np.ndarray,
    mode: str = "soft",
) -> tuple[int, np.ndarray]:
    """Aggregate window-level predictions into a single record prediction.

    Args:
        predictions (np.ndarray): Array of shape (num_windows, num_classes) containing
            per-window class probabilities.
        mode (str): "soft" averages probabilities; "hard" uses majority vote
            on argmax class indices.

    Returns:
        tuple[int, np.ndarray]: Tuple of (predicted_class_index, averaged_probabilities).

    """
    if mode == "soft":
        avg_probs = predictions.mean(axis=0)
        predicted_class = int(avg_probs.argmax())
    else:
        hard_preds = predictions.argmax(axis=1)
        predicted_class = int(np.bincount(hard_preds, minlength=predictions.shape[1]).argmax())
        avg_probs = predictions.mean(axis=0)

    return predicted_class, avg_probs


def main() -> None:
    """Entry point for the training script."""
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005 - No timezone needed
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(run_dir / "logs"), "training")
    logger.info("Arguments: %s", args)

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data path does not exist: %s", data_path)
        return

    logger.info("Loading datasets...")

    augmentation = ECGAugmentation()

    train_dataset = ECGDataset(
        data_path=str(data_path),
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        window_stride=args.window_stride,
        split="train",
        transforms=augmentation,
    )
    val_dataset = ECGDataset(
        data_path=str(data_path),
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        window_stride=args.window_stride,
        split="val",
    )
    test_dataset = ECGDataset(
        data_path=str(data_path),
        sampling_rate=args.sampling_rate,
        window_size=args.window_size,
        window_stride=args.window_stride,
        split="test",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    windows_size = int(args.window_size * args.sampling_rate)
    logger.info(
        "Window size: %d samples (%.1f s @ %d Hz)",
        windows_size,
        args.window_size,
        args.sampling_rate,
    )

    model = ResNet1D(
        input_channels=12,
        num_classes=5,
        base_filters=args.base_filters,
        dropout=args.dropout,
    ).to(device)

    logger.info("Model: %d parameters", count_parameters(model))

    class_weights = train_dataset.get_class_weights().to(device)
    logger.info("Class weights: %s", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )
    early_stopping = EarlyStopping(args.patience, mode="max")

    start_epoch = 0
    best_f1 = 0.0

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint.get("metrics", {}).get("val_f1_macro", 0.0)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
    }

    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info("=" * 60)
        logger.info("Epoch %d/%d | LRL: %.6f", epoch + 1, args.epochs, current_lr)
        logger.info("=" * 60)

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        val_metrics = validate(model, val_loader, criterion, device, epoch + 1)

        scheduler.step(val_metrics["f1_macro"])

        logger.info(
            "Train: Loss=%.4f Acc=%.4f F1=%.4f",
            train_metrics["loss"],
            train_metrics["accuracy"],
            train_metrics["f1_macro"],
        )
        logger.info(
            "Val:   Loss=%.4f Acc=%.4f F1=%.4f",
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
        )

        f1_parts = [f"{name}={val_metrics.get(f'f1_{name}', 0):.3f}" for name in CLASS_NAMES]
        logger.info("Val F1 per-class: %s", ", ".join(f1_parts))

        if (epoch + 1) % 5 == 0:
            val_calc = MetricsCalculator()
            model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_calc.update(output, target)

            voting_metrics = evaluate_with_voting(
                model,
                val_dataset,
                device,
                num_samples=200,
            )
            logger.info(
                "Voting F1 (val subset): acc=%.4f f1=%.4f",
                voting_metrics["voting_accuracy"],
                voting_metrics["voting_f1_macro"],
            )

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["lr"].append(current_lr)

        is_best = val_metrics["f1_macro"] > best_f1
        if is_best:
            best_f1 = val_metrics["f1_macro"]
            logger.info("New best model! F1=%.4f", best_f1)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train": train_metrics, "val": val_metrics},
            path=str(run_dir / "checkpoint.pth"),
            is_best=is_best,
        )

        if early_stopping(val_metrics["f1_macro"]):
            logger.info("Early stopping triggered after %d epochs", epoch + 1)
            break

    save_training_history(history, run_dir / "history.json")

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)

    best_model_path = run_dir / "best_model.pt"
    if best_model_path.exists():
        best_checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    else:
        logger.info("No best_model.pt found - using in-memory model weights")

    test_metrics = validate(model, test_loader, criterion, device, epoch=0)
    logger.info("Per-window metrics:\n%s", format_metrics(test_metrics))

    voting_metrics = evaluate_with_voting(model, test_dataset, device)
    logger.info("Voting metrics (per-record):\n%s", format_metrics(voting_metrics))

    final_model_path = output_dir / "ecg_resnet1d.pt"
    if best_model_path.exists():
        shutil.copy(best_model_path, final_model_path)
    else:
        save_checkpoint(model, optimizer, start_epoch, {}, str(final_model_path))
    logger.info("Best model saved to: %s", final_model_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
