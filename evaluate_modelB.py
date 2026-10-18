import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders
from modelB import ModelB_LightweightCNN


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 32
NUM_CLASSES = 17

MODEL_PATH = "optimizer_comparison_modelB_adam_best.pth"


# ============================================================
# Load test dataset
# ============================================================

_, _, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")


# ============================================================
# Load trained Model B
# ============================================================

model = ModelB_LightweightCNN(
    num_classes=NUM_CLASSES
).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
)

model.eval()

print("Model B loaded successfully.")


# ============================================================
# Get class names
# ============================================================

# The class names are obtained from the dataset object.
# This assumes get_dataloaders uses the same dataset structure
# described in your project.

train_dataset = test_loader.dataset

if hasattr(train_dataset, "dataset"):
    # Handles Subset objects created by random_split.
    original_dataset = train_dataset.dataset
else:
    original_dataset = train_dataset

if hasattr(original_dataset, "datasets"):
    # Handles ConcatDataset objects.
    class_names = original_dataset.datasets[0].classes
else:
    class_names = original_dataset.classes

print(f"Number of classes: {len(class_names)}")


# ============================================================
# Predict test samples
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Calculate confusion matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
)


# ============================================================
# Plot confusion matrix
# ============================================================

plt.figure(figsize=(16, 13))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.title("Confusion Matrix - Model B", fontsize=16)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)

plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_modelB.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report - Model B\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
)


# ============================================================
# Test accuracy
# ============================================================

correct = sum(
    actual == predicted
    for actual, predicted in zip(
        all_labels,
        all_predictions,
    )
)

accuracy = 100.0 * correct / len(all_labels)

print(f"\nTest Accuracy: {accuracy:.2f}%")
print("Confusion matrix saved as confusion_matrix_modelB.png")