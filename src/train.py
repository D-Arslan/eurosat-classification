"""Training script for EuroSAT classification with ResNet-18."""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import load_eurosat
from model import build_model
from utils import set_seed, setup_logging, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model, return average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train EuroSAT classifier")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Setup
    logger = setup_logging()
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Config: epochs={args.epochs}, lr={args.lr}, "
                f"batch_size={args.batch_size}, seed={args.seed}, "
                f"patience={args.patience}")
    logger.info(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = load_eurosat(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    # Model
    model = build_model(num_classes=10, pretrained=True).to(device)

    # Loss: CrossEntropyLoss (standard for multi-class classification)
    # L = -sum(y_i * log(p_i)) for i in classes
    # No explicit regularization term; weight_decay in optimizer acts as L2
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam with L2 regularization (weight_decay=1e-4)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Scheduler: reduce LR on validation loss plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss": train_loss,
                "train_acc": train_acc,
            }, checkpoint_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered.")
                break

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} min")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
