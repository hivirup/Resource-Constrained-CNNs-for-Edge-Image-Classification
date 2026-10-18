import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders
from modelA import ModelA_StandardCNN
from modelB import ModelB_LightweightCNN


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 17

MODELS = [
    {
        "name": "Model A",
        "class": ModelA_StandardCNN,
        "checkpoint": "optimizer_comparison_modelA_adam_best.pth",
        "confusion_matrix": "confusion_matrix_modelA.png",
    },
    {
        "name": "Model B",
        "class": ModelB_LightweightCNN,
        "checkpoint": "optimizer_comparison_modelB_adam_best.pth",
        "confusion_matrix": "confusion_matrix_modelB.png",
    },
]


def get_class_names(dataset):
    # The test split is a Subset of a ConcatDataset of ImageFolder datasets.
    original_dataset = dataset.dataset if hasattr(dataset, "dataset") else dataset

    if hasattr(original_dataset, "datasets"):
        return original_dataset.datasets[0].classes
    return original_dataset.classes


def evaluate_model(model_config, test_loader, class_names, device):
    model = model_config["class"](num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(
            model_config["checkpoint"],
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()
    print(f"\n{model_config['name']} loaded from {model_config['checkpoint']}.")

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    plt.figure(figsize=(16, 13))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {model_config['name']}", fontsize=16)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("Actual Class", fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        model_config["confusion_matrix"],
        dpi=300,
        bbox_inches="tight",
    )
    print(f"Confusion matrix saved: {model_config['confusion_matrix']}")

    print(f"\nClassification Report - {model_config['name']}\n")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    ))

    correct = sum(
        actual == predicted
        for actual, predicted in zip(all_labels, all_predictions)
    )
    accuracy = 100.0 * correct / len(all_labels)
    print(f"{model_config['name']} test accuracy: {accuracy:.2f}%")


def main():
    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = get_class_names(test_loader.dataset)
    print(f"Number of classes: {len(class_names)}")

    for model_config in MODELS:
        evaluate_model(model_config, test_loader, class_names, device)

    plt.show()


if __name__ == "__main__":
    main()
