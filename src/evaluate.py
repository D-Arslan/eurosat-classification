"""Evaluation script: metrics, confusion matrix, and error analysis."""

import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import load_eurosat, EUROSAT_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from utils import set_seed, get_device


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor.cpu() * std + mean


@torch.no_grad()
def evaluate(model, loader, device):
    """Run evaluation, return predictions and ground truth."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_images = []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_images.extend(images.cpu())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        all_images
    )


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EUROSAT_CLASSES,
        yticklabels=EUROSAT_CLASSES
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — EuroSAT Classification")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_misclassified(images, y_true, y_pred, probs, output_dir, n=9):
    """Show misclassified examples for error analysis."""
    misclassified_idx = np.where(y_true != y_pred)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified examples!")
        return

    # Sort by confidence (most confident mistakes first)
    confidences = [probs[i][y_pred[i]] for i in misclassified_idx]
    sorted_idx = np.argsort(confidences)[::-1]
    selected = misclassified_idx[sorted_idx[:n]]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, idx in zip(axes.flat, selected):
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(
            f"True: {EUROSAT_CLASSES[y_true[idx]]}\n"
            f"Pred: {EUROSAT_CLASSES[y_pred[idx]]} "
            f"({probs[idx][y_pred[idx]]:.2%})",
            fontsize=9
        )
        ax.axis("off")

    plt.suptitle("Most Confident Misclassifications", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "misclassified.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved misclassified examples to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate EuroSAT classifier")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test set
    _, _, test_loader = load_eurosat(
        data_dir=args.data_dir, batch_size=args.batch_size, seed=args.seed
    )

    # Load model
    model = build_model(num_classes=10, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_acc']:.4f}")

    # Evaluate
    y_pred, y_true, probs, images = evaluate(model, test_loader, device)

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=EUROSAT_CLASSES, digits=4
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Save report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, args.output_dir)

    # Error analysis
    plot_misclassified(images, y_true, y_pred, probs, args.output_dir)

    # Summary
    accuracy = np.mean(y_true == y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Total test samples: {len(y_true)}")
    print(f"Misclassified: {np.sum(y_true != y_pred)}")


if __name__ == "__main__":
    main()
