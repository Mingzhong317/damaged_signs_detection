import argparse, os, copy, json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# ----------------------
# Utility: EarlyStopping
# ----------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_f1):
        if self.best_score is None:
            self.best_score = val_f1
            return False
        if val_f1 < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = val_f1
            self.counter = 0
        return False

# -------------
# Data loaders
# -------------

def build_dataloaders(root: str, img_size: int = 224, batch_size: int = 32):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(Path(root) / "train", transform=train_transform)
    val_set = datasets.ImageFolder(Path(root) / "val", transform=eval_transform)
    test_set = datasets.ImageFolder(Path(root) / "test", transform=eval_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

# -----------------
# Model constructor
# -----------------

def build_model(arch: str, pretrained: bool = True):
    arch = arch.lower()
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    elif arch == "vgg16":
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model

# -----------------
# Train / Evaluate
# -----------------

def train_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device, criterion):
    model.eval()
    preds, gts, infer_times = [], [], []
    eval_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            start = time.time()
            outputs = model(inputs)
            infer_times.append(time.time() - start)
            loss = criterion(outputs, labels)
            eval_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs).cpu().numpy().ravel()
            preds.extend((probs >= 0.5).astype(int).tolist())
            gts.extend(labels.cpu().numpy().ravel().astype(int).tolist())
    avg_loss = eval_loss / len(loader.dataset)
    acc = accuracy_score(gts, preds)
    prec = precision_score(gts, preds, zero_division=0)
    rec = recall_score(gts, preds, zero_division=0)
    f1 = f1_score(gts, preds, zero_division=0)
    mean_infer = sum(infer_times) / len(infer_times)
    return avg_loss, acc, prec, rec, f1, mean_infer

# -----------------------
# YOLOv8 (classification)
# -----------------------

def train_yolov8(args, results_dict):
    from ultralytics import YOLO

    model = YOLO("yolov8s-cls.pt")
    # train(): Ultralytics recognises train/val/test folders directly for classification
    model.train(
        data=str(args.data_dir),  # dataset root containing sub‑folders
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        lr0=args.lr,
        patience=args.patience,
        optimizer="AdamW",
    )

    metrics = model.val(data=str(args.data_dir), split="test", name='val')
    cm = metrics.confusion_matrix.matrix        # (nc, nc)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    eps = 1e-16

    # ---- per-class ----
    precision_cls = tp / (tp + fp + eps)
    recall_cls    = tp / (tp + fn + eps)
    f1_cls        = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + eps)

    # ---- macro ----
    macro_p  = precision_cls.mean()
    macro_r  = recall_cls.mean()
    macro_f1 = f1_cls.mean()

    # ---- micro ----
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    micro_p  = micro_tp / (micro_tp + micro_fp + eps)
    micro_r  = micro_tp / (micro_tp + micro_fn + eps)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)

    # ---- 同 evaluate() 的欄位命名 ----
    results_dict["yolov8s-cls"] = {
        "accuracy"      : float(metrics.top1),                 # = acc
        "precision"     : float(micro_p),                      # = prec
        "recall"        : float(micro_r),                      # = rec
        "f1"            : float(micro_f1),                     # = f1
        "infer_time_ms" : float(metrics.speed["inference"]*1e3)
    }

# -------------
# Main routine
# -------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    train_loader, val_loader, test_loader = build_dataloaders(args.data_dir, args.img_size, args.batch_size)

    results = {}

    for arch in args.archs:
        arch = arch.lower()
        print(f"\n===== Training {arch} =====")

        if arch == "yolov8s-cls":
            train_yolov8(args, results)
            continue

        model = build_model(arch, pretrained=True).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, delta=args.delta)

        best_f1 = 0.0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, acc, prec, rec, f1, _ = evaluate(model, val_loader, device, criterion)
            print(
                f"Epoch {epoch:03d}: train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} "
                f"| acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}"
            )
            if f1 > best_f1:
                best_f1 = f1
                best_state = copy.deepcopy(model.state_dict())
            if early_stopping(f1):
                print("Early stopping triggered.")
                break

        # -- Test set evaluation
        model.load_state_dict(best_state)
        test_loss, acc, prec, rec, f1, infer_time = evaluate(model, test_loader, device, criterion)
        print(
            f"Test: loss={test_loss:.4f}, acc={acc:.4f}, prec={prec:.4f}, "
            f"rec={rec:.4f}, f1={f1:.4f}, infer_time={infer_time * 1000:.2f} ms"
        )

        results[arch] = {
            "test_loss": test_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "infer_time_ms": infer_time * 1000,
        }

        ckpt_path = Path(args.output_dir) / f"{arch}_best.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)

    # -------------
    # Save results
    # -------------
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    result_file = Path(args.output_dir) / f"results_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # -------------
    # Visualisation
    # -------------
    try:
        archs = list(results.keys())
        accs = [results[a]["accuracy"] for a in archs]
        f1s = [results[a]["f1"] for a in archs]

        x = range(len(archs))
        plt.figure(figsize=(10, 6))
        plt.bar([i - 0.2 for i in x], accs, width=0.4, label="Accuracy")
        plt.bar([i + 0.2 for i in x], f1s, width=0.4, label="F1‑score")
        plt.xticks(x, archs, rotation=45)
        plt.ylabel("Score")
        plt.title("Model Comparison on Damaged Sign Classification")
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / "metrics.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"[WARN] Failed to create plot: {e}")

    print(f"\nFinished. Results saved to {result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Damaged Traffic Sign Classification Training"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Dataset root with train/val/test folders"
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=[
            "resnet50",
            "densenet121",
            "efficientnet_b0",
            "mobilenet_v3_large",
            "vgg16",
            "yolov8s-cls",
        ],
        help="List of architectures to train",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")

    main(parser.parse_args())