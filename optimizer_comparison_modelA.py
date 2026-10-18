import csv
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from dataset import get_dataloaders
from modelA import ModelA_StandardCNN


# Keep the same training configuration used for Model B's Adam run.
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 17
LEARNING_RATE = 0.001

MODEL_FILE = "optimizer_comparison_modelA_adam_best.pth"
CSV_FILE = "optimizer_comparison_modelA_adam_history.csv"
PLOT_FILE = "optimizer_comparison_modelA.png"

EXPERIMENTS = {
    "Adam": {"lr": LEARNING_RATE, "momentum": None},
    "SGD": {"lr": 0.01, "momentum": None},
    "SGD + Momentum": {"lr": 0.01, "momentum": 0.9},
}


def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, 100.0 * total_correct / total_samples


def make_optimizer(name, model, settings):
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=settings["lr"])
    return torch.optim.SGD(
        model.parameters(),
        lr=settings["lr"],
        **({} if settings["momentum"] is None else {"momentum": settings["momentum"]}),
    )


def run_experiment(name, settings, device, train_loader, val_loader):
    print(f"\nOptimizer: {name} | Learning rate: {settings['lr']}")
    if settings["momentum"] is not None:
        print(f"Momentum: {settings['momentum']}")

    # Reset the seed so all optimizers start from identical Model A weights.
    set_seed(SEED)
    model = ModelA_StandardCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(name, model, settings)

    history = []
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )
        epoch_time = time.perf_counter() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy_percent": train_accuracy,
            "val_accuracy_percent": val_accuracy,
            "epoch_time_seconds": epoch_time,
        })

        if name == "Adam" and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Acc: {val_accuracy:.2f}% | Time: {epoch_time:.2f}s"
        )

    if name == "Adam" and best_model_state is not None:
        torch.save(best_model_state, MODEL_FILE)
        print(f"\nBest Model A Adam checkpoint saved: {MODEL_FILE}")
        print(f"Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.4f}")

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "epoch", "train_loss", "val_loss", "train_accuracy_percent",
                "val_accuracy_percent", "epoch_time_seconds",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"Adam training history saved: {CSV_FILE}")

    return history


def generate_plot(histories):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = range(1, EPOCHS + 1)

    for name, history in histories.items():
        axes[0].plot(
            epochs, [row["train_loss"] for row in history], label=name, linewidth=2
        )
        axes[1].plot(
            epochs, [row["val_loss"] for row in history], label=name, linewidth=2
        )

    axes[0].set_title("Training Loss")
    axes[1].set_title("Validation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xticks(range(1, EPOCHS + 1, 2))

    fig.suptitle("Model A: Adam vs SGD vs SGD with Momentum", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved: {PLOT_FILE}")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs per optimizer: {EPOCHS} | Batch size: {BATCH_SIZE}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    histories = {}
    for name, settings in EXPERIMENTS.items():
        histories[name] = run_experiment(
            name, settings, device, train_loader, val_loader
        )

    generate_plot(histories)
    print(f"Best Adam model: {MODEL_FILE}")
    print(f"Adam history CSV: {CSV_FILE}")


if __name__ == "__main__":
    main()
