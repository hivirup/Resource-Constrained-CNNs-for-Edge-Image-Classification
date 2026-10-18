import random
import time
import copy
import csv

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modelB import ModelB_LightweightCNN


# ============================================================
# Configuration
# ============================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 17

PLOT_FILE = "optimizer_comparison_modelB.png"
MODEL_FILE = "optimizer_comparison_modelB_adam_best.pth"
ADAM_CSV_FILE = "optimizer_comparison_modelB_adam_history.csv"

EXPERIMENTS = {
    "Adam": {
        "lr": 0.001,
        "momentum": None,
    },
    "SGD": {
        "lr": 0.01,
        "momentum": None,
    },
    "SGD + Momentum": {
        "lr": 0.01,
        "momentum": 0.9,
    },
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Create optimizer
# ============================================================

def make_optimizer(name, model, settings):
    if name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=settings["lr"],
        )

    if name == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=settings["lr"],
        )

    return torch.optim.SGD(
        model.parameters(),
        lr=settings["lr"],
        momentum=settings["momentum"],
    )


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()

        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# Validate one epoch
# ============================================================

def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# Run one optimizer experiment
# ============================================================

def run_experiment(
    name,
    settings,
    device,
    train_loader,
    val_loader,
):
    print("\n" + "=" * 65)
    print(f"Optimizer: {name}")
    print(f"Learning rate: {settings['lr']}")

    if settings["momentum"] is not None:
        print(f"Momentum: {settings['momentum']}")

    print("=" * 65)

    # Start every experiment with the same initial model weights.
    set_seed(SEED)

    model = ModelB_LightweightCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = make_optimizer(
        name,
        model,
        settings,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "epoch_time_seconds": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    best_val_accuracy = 0.0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        epoch_time = time.perf_counter() - start_time

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["epoch_time_seconds"].append(epoch_time)

        # Save the best weights in memory for Adam only.
        if name == "Adam" and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_accuracy = val_accuracy

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    # --------------------------------------------------------
    # Save the best Adam checkpoint only
    # --------------------------------------------------------

    if name == "Adam" and best_model_state is not None:
        torch.save(
            best_model_state,
            MODEL_FILE,
        )

        print("\nBest Adam model saved successfully.")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(
            f"Validation accuracy at best epoch: "
            f"{best_val_accuracy:.2f}%"
        )
        print(f"Checkpoint: {MODEL_FILE}")

        with open(ADAM_CSV_FILE, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "train_accuracy_percent",
                "val_accuracy_percent", "epoch_time_seconds",
            ])
            for epoch in range(EPOCHS):
                writer.writerow([
                    epoch + 1,
                    history["train_loss"][epoch],
                    history["val_loss"][epoch],
                    history["train_accuracy"][epoch],
                    history["val_accuracy"][epoch],
                    history["epoch_time_seconds"][epoch],
                ])
        print(f"Adam training history saved: {ADAM_CSV_FILE}")

    return history


# ============================================================
# Generate optimizer comparison plot
# ============================================================

def generate_plot(histories):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
    )

    epochs = range(1, EPOCHS + 1)

    for name, history in histories.items():
        axes[0].plot(
            epochs,
            history["train_loss"],
            label=name,
            linewidth=2,
        )

        axes[1].plot(
            epochs,
            history["val_loss"],
            label=name,
            linewidth=2,
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(range(1, EPOCHS + 1, 2))

    fig.suptitle(
        "Model B: Adam vs SGD vs SGD with Momentum",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nComparison plot saved: {PLOT_FILE}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Epochs per optimizer: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    # Get the same train/validation split for all experiments.
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=BATCH_SIZE
    )

    histories = {}

    for name, settings in EXPERIMENTS.items():
        histories[name] = run_experiment(
            name=name,
            settings=settings,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
        )

    # Generate the only image file.
    generate_plot(histories)

    print("\n" + "=" * 65)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 65)
    print(f"Plot: {PLOT_FILE}")
    print(f"Best Adam model: {MODEL_FILE}")
    print(f"Adam history CSV: {ADAM_CSV_FILE}")


if __name__ == "__main__":
    main()
