"""Training module for ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

from ecg.dataset import ECGDataset
from ecg.ml.metrics import AverageMeter, MetricsCalculator
from ecg.ml.model import ECGClassifier


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
        device (torch.device): Device to run the evaluation on (CPU or ).
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

    record_ids = dataset.records.idex.tolist()
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
