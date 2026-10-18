"""
Q5: SOTA Fine-Tuning & Evaluation  (edited for CPU-only laptop)
Fine-tunes MobileNetV2 and EfficientNet-B0 (ImageNet pre-trained) on the Jute Pest dataset
using the SAME train/val/test splits as the custom networks.

Expected folder layout (class subfolders inside each split):
    Jute_Pest_Dataset/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg

Install:  pip install torch torchvision scikit-learn pandas matplotlib seaborn
"""
import os, time, copy
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Config ----------------
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # change to your full path if needed
IMG_SIZE = 160              # was 224 -> ~2x less compute
BATCH_SIZE = 32
EPOCHS = 5                  # was 10 (enough when backbone is frozen)
LR = 1e-3
SEED = 42
FREEZE_BACKBONE = True      # train only the new classifier head (big CPU speedup)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count())   # use all CPU cores
print(f"Device: {DEVICE} | CPU threads: {torch.get_num_threads()}")

# ---------------- Data ----------------
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),      # RandomRotation/ColorJitter removed: slow on CPU
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   eval_tf)
test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  eval_tf)
class_names = train_ds.classes
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}")
print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

# ---------------- Models ----------------
def build_model(name):
    if name == "MobileNetV2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    elif name == "EfficientNet-B0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    else:
        raise ValueError(name)
    if FREEZE_BACKBONE:
        for p in m.features.parameters():
            p.requires_grad = False
    return m.to(DEVICE)

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_size_mb(model, path="tmp_weights.pth"):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 ** 2)
    os.remove(path)
    return size

# ---------------- Train / Eval ----------------
def run_epoch(model, loader, criterion, optimizer=None, tag=""):
    train = optimizer is not None
    model.train(train)
    # keep frozen BatchNorm layers in eval mode so their statistics aren't disturbed
    if train and FREEZE_BACKBONE:
        model.features.eval()
    loss_sum, correct, n = 0.0, 0, 0
    t0 = time.time()
    with torch.set_grad_enabled(train):
        for i, (x, y) in enumerate(loader, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
            if train and i % 10 == 0:
                print(f"   {tag} batch {i}/{len(loader)} | {time.time()-t0:.0f}s", flush=True)
    return loss_sum / n, correct / n

def fine_tune(name):
    model = build_model(name)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    history = []
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, tag=f"[{name}] ep{ep}")
        vl, va = run_epoch(model, val_loader, criterion)
        scheduler.step()
        history.append((tl, ta, vl, va))
        print(f"[{name}] Epoch {ep:02d}/{EPOCHS} | train loss {tl:.4f} acc {ta:.4f} | "
              f"val loss {vl:.4f} acc {va:.4f} | {time.time()-t0:.0f}s elapsed", flush=True)
        if va > best_acc:
            best_acc, best_wts = va, copy.deepcopy(model.state_dict())
    train_time = time.time() - t0
    model.load_state_dict(best_wts)
    return model, history, train_time

@torch.no_grad()
def evaluate(model):
    model.eval()
    y_true, y_pred = [], []
    t0 = time.time()
    for x, y in test_loader:
        out = model(x.to(DEVICE))
        y_pred += out.argmax(1).cpu().tolist()
        y_true += y.tolist()
    infer_ms = (time.time() - t0) / len(test_ds) * 1000
    return y_true, y_pred, infer_ms

# ---------------- Main ----------------
results = []
for name in ["MobileNetV2", "EfficientNet-B0"]:
    print(f"\n===== {name} =====")
    model, hist, train_time = fine_tune(name)
    y_true, y_pred, infer_ms = evaluate(model)

    rep = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    total_p, train_p = count_params(model)
    size = model_size_mb(model)

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    results.append({
        "Model": name,
        "Test Accuracy": accuracy_score(y_true, y_pred),
        "Precision (macro)": rep["macro avg"]["precision"],
        "Recall (macro)": rep["macro avg"]["recall"],
        "F1 (macro)": rep["macro avg"]["f1-score"],
        "Total Params": total_p,
        "Trainable Params": train_p,
        "Model Size (MB)": round(size, 2),
        "Train Time (s)": round(train_time, 1),
        "Inference (ms/img)": round(infer_ms, 2),
    })

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} - Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(f"{name}_confusion_matrix.png", dpi=150); plt.close()

    # Training curves
    h = list(zip(*hist))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h[0], label="train"); ax[0].plot(h[2], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(h[1], label="train"); ax[1].plot(h[3], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    fig.suptitle(name); plt.tight_layout(); plt.savefig(f"{name}_curves.png", dpi=150); plt.close()

    torch.save(model.state_dict(), f"{name}_best.pth")

df = pd.DataFrame(results)
df.to_csv("sota_results.csv", index=False)
print("\n===== Summary =====")
print(df.to_string(index=False))