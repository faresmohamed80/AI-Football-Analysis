"""Run this script to generate the Jupyter notebook (.ipynb) file."""
import json, os

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source": source if isinstance(source,list) else [source]})

def code(source):
    lines = source.strip().split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})

# ── Title ────────────────────────────────────────────────────────
md([
    "# ⚽ Football Action Recognition — Fine-tuning R3D-18\n",
    "**Classes:** `CROSS` | `HEADER` | `HIGH_PASS` | `PASS` | `SHOT` | `THROW_IN`  \n",
    "**Dataset:** ~458 player-cropped video clips  \n",
    "**Platform:** Google Colab (T4 GPU)  \n",
    "**Model:** R3D-18 pretrained on Kinetics-400 → fine-tuned on your data\n",
])

# ── Cell 1: Install ──────────────────────────────────────────────
md("## 1. Install & Imports")
code("""
!pip install -q torchvision torch tqdm scikit-learn matplotlib seaborn

import os, random, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models.video as video_models
import torchvision.transforms as T
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
""")

# ── Cell 2: Mount Drive ──────────────────────────────────────────
md("## 2. Mount Google Drive")
code("""
drive.mount('/content/drive')
""")

# ── Cell 3: Config ───────────────────────────────────────────────
md([
    "## 3. Configuration\n",
    "> **Important:** Upload your `manual_crops` folder to Google Drive first,  \n",
    "> then set `DATASET_PATH` to match its location.\n"
])
code("""
# ── Dataset Path ─────────────────────────────────────────────────
DATASET_PATH = '/content/drive/MyDrive/manual_crops'

# ── Classes ──────────────────────────────────────────────────────
CLASSES     = ['CROSS', 'HEADER', 'HIGH_PASS', 'PASS', 'SHOT', 'THROW_IN']
NUM_CLASSES = len(CLASSES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
print("Classes:", CLASS2IDX)

# ── Hyperparameters ───────────────────────────────────────────────
NUM_FRAMES   = 16
FRAME_SIZE   = 112
BATCH_SIZE   = 8
NUM_EPOCHS   = 40
LR           = 1e-4
LR_MIN       = 1e-6
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
VAL_SPLIT    = 0.2
SEED         = 42

# ── Output ────────────────────────────────────────────────────────
SAVE_DIR   = '/content/drive/MyDrive/action_model_output'
BEST_MODEL = os.path.join(SAVE_DIR, 'best_action_model.pt')
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
""")

# ── Cell 4: Dataset Class ────────────────────────────────────────
md("## 4. Dataset & Augmentation")
code("""
class AugmentedDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
        self.std  = torch.tensor([0.22803,  0.22145, 0.216989]).view(3,1,1,1)

        self.aug = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            T.ToTensor(),
        ]) if augment else None

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, path):
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return np.zeros((NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        frames  = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, f = cap.read()
            if ret:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE))
            else:
                f = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            frames.append(f)
        cap.release()
        return np.stack(frames)   # (T, H, W, C)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        raw = self._load_frames(path)     # (T, H, W, C) uint8
        processed = []
        for t in range(raw.shape[0]):
            frame = raw[t]
            if self.aug:
                frame = self.aug(frame)   # (C, H, W) float
            else:
                frame = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
            processed.append(frame)
        tensor = torch.stack(processed, dim=1)   # (C, T, H, W)
        tensor = (tensor - self.mean) / self.std
        return tensor, label

print("Dataset class defined.")
""")

# ── Cell 5: Build Splits ─────────────────────────────────────────
md("## 5. Load Dataset & Build Train/Val Splits")
code("""
from collections import Counter

def collect_samples(root, classes):
    samples = []
    for cls in classes:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            print(f"  WARNING: {cls} not found")
            continue
        vids = [f for f in os.listdir(d) if f.endswith('.mp4')]
        for v in vids:
            samples.append((os.path.join(d, v), CLASS2IDX[cls]))
        print(f"  {cls}: {len(vids)}")
    return samples

print("Scanning dataset...")
all_samples = collect_samples(DATASET_PATH, CLASSES)
print(f"Total: {len(all_samples)} clips")

random.shuffle(all_samples)
val_size   = int(len(all_samples) * VAL_SPLIT)
train_samp = all_samples[val_size:]
val_samp   = all_samples[:val_size]
print(f"Train: {len(train_samp)} | Val: {len(val_samp)}")

train_labels  = [s[1] for s in train_samp]
class_counts  = [train_labels.count(i) for i in range(NUM_CLASSES)]
class_weights = [1.0/c if c > 0 else 0 for c in class_counts]

print("\\nClass distribution (train):")
for i, c in enumerate(CLASSES):
    print(f"  {c}: {class_counts[i]}")

sample_weights = [class_weights[l] for _, l in train_samp]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_ds = AugmentedDataset(train_samp, augment=True)
val_ds   = AugmentedDataset(val_samp,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=2, pin_memory=True)
print(f"\\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")
""")

# ── Cell 6: Model ────────────────────────────────────────────────
md("## 6. Model Setup — Fine-tune R3D-18")
code("""
model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)

# Freeze layers 1 & 2 (low-level features), train 3 & 4 + head
for layer in [model.layer1, model.layer2]:
    for p in layer.parameters():
        p.requires_grad = False

# Replace classification head
in_feat = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(p=DROPOUT), nn.Linear(in_feat, NUM_CLASSES))

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# Weighted loss for class imbalance
cls_w_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion    = nn.CrossEntropyLoss(weight=cls_w_tensor)

# Differential LR: lower for backbone, higher for head
head_params     = list(model.fc.parameters())
backbone_params = [p for p in model.parameters() if p.requires_grad
                   and not any(p is hp for hp in head_params)]

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},
    {'params': head_params,     'lr': LR}
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN)
print("Model ready!")
""")

# ── Cell 7: Training ─────────────────────────────────────────────
md("## 7. Training")
code("""
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(loader, desc="Train", leave=False):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(clips)
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += clips.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for clips, labels in tqdm(loader, desc="Val  ", leave=False):
        clips, labels = clips.to(device), labels.to(device)
        out  = model(clips)
        loss = criterion(out, labels)
        total_loss += loss.item() * clips.size(0)
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += clips.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels

history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_val_acc = 0.0
best_weights = None

print(f"Starting training — {NUM_EPOCHS} epochs on {DEVICE}\\n{'='*55}")
for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    vl_loss, vl_acc, vp, vl = val_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()
    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, BEST_MODEL)
        star = " ★ SAVED"
    else:
        star = ""
    print(f"Ep {epoch:02d}/{NUM_EPOCHS}  tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
          f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f}  [{time.time()-t0:.0f}s]{star}")

print(f"\\nBest Val Accuracy: {best_val_acc:.4f}")
""")

# ── Cell 8: Evaluation ───────────────────────────────────────────
md("## 8. Evaluation — Curves & Confusion Matrix")
code("""
model.load_state_dict(best_weights)
_, final_acc, final_preds, final_labels = val_epoch(model, val_loader, criterion, DEVICE)

print(classification_report(final_labels, final_preds, target_names=CLASSES))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Training History  (Best Val Acc: {best_val_acc:.3f})', fontsize=13, fontweight='bold')
axes[0].plot(history['train_loss'], label='Train', color='#4FC3F7')
axes[0].plot(history['val_loss'],   label='Val',   color='#FF8A65')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history['train_acc'], label='Train', color='#4FC3F7')
axes[1].plot(history['val_acc'],   label='Val',   color='#FF8A65')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()

cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm.astype(float)/cm.sum(axis=1, keepdims=True), annot=True, fmt='.2f',
            cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, annot_kws={'size':11})
plt.title(f'Confusion Matrix', fontsize=13)
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()
print(f"Saved to: {SAVE_DIR}")
""")

# ── Cell 9: Inference ────────────────────────────────────────────
md("## 9. Quick Inference Test")
code("""
@torch.no_grad()
def predict_video(video_path, model, device):
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
    std  = torch.tensor([0.22803,  0.22145, 0.216989]).view(3,1,1,1)
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, NUM_FRAMES, dtype=int)
    frames = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ret, f = cap.read()
        if ret:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE))
        else:
            f = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        frames.append(f)
    cap.release()
    tensor = torch.from_numpy(np.stack(frames)).permute(3,0,1,2).unsqueeze(0).float() / 255.0
    # (T,H,W,C) -> (T,C,H,W) wait — stack gives (T,H,W,C) → permute(3,0,1,2) gives (C,T,H,W)?
    # np.stack(frames) -> (T,H,W,C); torch -> (T,H,W,C); permute(3,0,1,2) -> (C,T,H,W) ✓ unsqueeze(0) -> (1,C,T,H,W) ✓
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)
    model.eval()
    out   = model(tensor)
    probs = torch.softmax(out, dim=1)[0]
    pred  = probs.argmax().item()
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Prediction: {CLASSES[pred]}  ({probs[pred]*100:.1f}%)")
    print("\\nAll probabilities:")
    for cls, p in zip(CLASSES, probs.tolist()):
        bar = '█' * int(p*25)
        print(f"  {cls:<30} {p*100:5.1f}%  {bar}")
    return CLASSES[pred]

# Test on a random val clip
test_path, test_label = random.choice(val_samp)
print(f"True label: {CLASSES[test_label]}\\n")
predict_video(test_path, model, DEVICE)
""")

# ── Build notebook JSON ──────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "football_action_recognition.ipynb"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = r"d:\offside\football_action_recognition.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"Notebook saved: {out_path}")
