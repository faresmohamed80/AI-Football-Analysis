"""Run this script to generate the SlowFast Jupyter notebook (.ipynb)."""
import json, os

cells = []

def md(source):
    cells.append({"cell_type":"markdown","metadata":{},"source": source if isinstance(source,list) else [source]})

def code(source):
    lines = source.strip().split('\n')
    src = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src})

# ── Title ─────────────────────────────────────────────────────────
md([
    "# ⚽ Football Action Recognition — SlowFast R50\n",
    "**SlowFast** uses **two pathways**:\n",
    "- 🐢 **Slow path**: 8 frames → captures spatial detail (what)\n",
    "- ⚡ **Fast path**: 32 frames → captures motion (how)\n\n",
    "**Classes:** `CROSS` | `HEADER` | `HIGH_PASS` | `PASS` | `SHOT` | `THROW_IN`  \n",
    "**Dataset:** ~458 player-cropped video clips  \n",
    "**Platform:** Google Colab (T4 GPU recommended)\n",
])

# ── Cell 1: Install ───────────────────────────────────────────────
md("## 1. Install & Imports")
code("""
!pip install -q pytorchvideo
!pip install -q torch torchvision tqdm scikit-learn matplotlib seaborn

import os, random, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")

# ── Cell 2: Mount Drive ───────────────────────────────────────────
md("## 2. Mount Google Drive")
code("""
drive.mount('/content/drive')
""")

# ── Cell 3: Config ────────────────────────────────────────────────
md([
    "## 3. Configuration\n",
    "> Upload your `manual_crops` folder to Google Drive before running.\n"
])
code("""
# Dataset path in Drive
DATASET_PATH = '/content/drive/MyDrive/manual_crops'

# Classes
CLASSES     = ['CROSS', 'HEADER', 'HIGH_PASS', 'PASS', 'SHOT', 'THROW_IN']
NUM_CLASSES = len(CLASSES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASSES)}
print("Classes:", CLASS2IDX)

# SlowFast input specs
SLOW_FRAMES = 8      # slow pathway frames
FAST_FRAMES = 32     # fast pathway frames  (alpha=4 ratio)
FRAME_SIZE  = 224    # standard SlowFast input

# Training
BATCH_SIZE   = 4     # SlowFast is heavier — T4 safe at batch=4
NUM_EPOCHS   = 40
LR           = 5e-5  # low LR for pretrained fine-tuning
LR_MIN       = 1e-7
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
VAL_SPLIT    = 0.2
SEED         = 42

SAVE_DIR   = '/content/drive/MyDrive/slowfast_model_output'
BEST_MODEL = os.path.join(SAVE_DIR, 'best_slowfast_model.pt')
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
""")

# ── Cell 4: Dataset ───────────────────────────────────────────────
md([
    "## 4. SlowFast Dataset\n",
    "SlowFast needs **two tensors** per clip:\n",
    "- `slow_clip`: shape `(C, 8, H, W)` — 1 frame every 4\n",
    "- `fast_clip`: shape `(C, 32, H, W)` — all 32 frames\n"
])
code("""
class SlowFastDataset(Dataset):
    \"\"\"
    Returns (slow_clip, fast_clip, label) where:
      slow_clip: (C, SLOW_FRAMES, H, W)
      fast_clip: (C, FAST_FRAMES, H, W)
    \"\"\"
    # Kinetics normalization
    MEAN = [0.45, 0.45, 0.45]
    STD  = [0.225, 0.225, 0.225]

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

        self.spatial_aug = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(FRAME_SIZE, scale=(0.8, 1.0)),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ]) if augment else T.Compose([
            T.ToPILImage(),
            T.Resize((FRAME_SIZE, FRAME_SIZE)),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, path, n_frames):
        \"\"\"Sample n_frames evenly from the video.\"\"\"
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return [np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)] * n_frames

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
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
        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Read FAST_FRAMES — then subsample for SLOW
        fast_raw = self._read_frames(path, FAST_FRAMES)
        slow_raw = fast_raw[::4][:SLOW_FRAMES]  # every 4th frame

        def process(frames):
            tensors = []
            for f in frames:
                tensors.append(self.spatial_aug(f))  # (C, H, W)
            return torch.stack(tensors, dim=1)       # (C, T, H, W)

        slow_clip = process(slow_raw)   # (C, 8, H, W)
        fast_clip = process(fast_raw)   # (C, 32, H, W)

        return [slow_clip, fast_clip], label

print("SlowFast Dataset class defined.")
""")

# ── Cell 5: Build Splits ──────────────────────────────────────────
md("## 5. Load Dataset & Build Train/Val Splits")
code("""
from collections import Counter

def collect_samples(root, classes):
    samples = []
    for cls in classes:
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            print(f"  WARNING: {cls} not found at {d}")
            continue
        vids = [f for f in os.listdir(d) if f.endswith('.mp4')]
        for v in vids:
            samples.append((os.path.join(d, v), CLASS2IDX[cls]))
        print(f"  {cls}: {len(vids)} clips")
    return samples

print("Scanning dataset...")
all_samples = collect_samples(DATASET_PATH, CLASSES)
print(f"Total: {len(all_samples)} clips")

random.shuffle(all_samples)
val_size   = int(len(all_samples) * VAL_SPLIT)
train_samp = all_samples[val_size:]
val_samp   = all_samples[:val_size]
print(f"Train: {len(train_samp)} | Val: {len(val_samp)}")

# Class distribution + weighted sampler
train_labels  = [s[1] for s in train_samp]
class_counts  = [train_labels.count(i) for i in range(NUM_CLASSES)]
class_weights = [1.0/c if c > 0 else 0 for c in class_counts]
print("\\nClass counts (train):")
for i, c in enumerate(CLASSES):
    print(f"  {c}: {class_counts[i]}")

sample_weights = [class_weights[l] for _, l in train_samp]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

def collate_fn(batch):
    \"\"\"Custom collate to stack the [slow, fast] lists properly.\"\"\"
    slow_clips = torch.stack([b[0][0] for b in batch])
    fast_clips = torch.stack([b[0][1] for b in batch])
    labels     = torch.tensor([b[1] for b in batch])
    return [slow_clips, fast_clips], labels

train_ds = SlowFastDataset(train_samp, augment=True)
val_ds   = SlowFastDataset(val_samp,   augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
print(f"\\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")
""")

# ── Cell 6: Model ─────────────────────────────────────────────────
md([
    "## 6. Load SlowFast R50 (Pretrained on Kinetics-400)\n",
    "We freeze the early blocks and replace the final head for 6 classes.\n"
])
code("""
import torch.hub

# Load pretrained SlowFast R50 from facebookresearch/pytorchvideo
print("Loading SlowFast R50 (pretrained on Kinetics-400)...")
model = torch.hub.load(
    'facebookresearch/pytorchvideo',
    'slowfast_r50',
    pretrained=True
)

# Inspect head structure
print("\\nOriginal head:")
print(model.blocks[-1])

# Replace projection layer (400 → NUM_CLASSES)
# In pytorchvideo SlowFast, the head is in model.blocks[-1].proj
in_feat = model.blocks[-1].proj.in_features
model.blocks[-1].proj = nn.Sequential(
    nn.Dropout(p=DROPOUT),
    nn.Linear(in_feat, NUM_CLASSES)
)
print(f"\\nReplaced head: {in_feat} → {NUM_CLASSES} classes")

# Freeze early blocks (0-3), train blocks 4-5 + head
for i, block in enumerate(model.blocks):
    requires_grad = (i >= 4)
    for p in block.parameters():
        p.requires_grad = requires_grad

model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

# Loss
cls_w_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion    = nn.CrossEntropyLoss(weight=cls_w_tensor)

# Optimizer — lower LR for backbone, higher for head
head_params     = list(model.blocks[-1].parameters())
backbone_params = [p for p in model.parameters()
                   if p.requires_grad and not any(p is h for h in head_params)]

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},
    {'params': head_params,     'lr': LR}
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN)
print("\\nModel ready!")
""")

# ── Cell 7: Training ──────────────────────────────────────────────
md("## 7. Training")
code("""
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(loader, desc="Train", leave=False):
        slow = clips[0].to(device)
        fast = clips[1].to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        out  = model([slow, fast])
        loss = criterion(out, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for clips, labels in tqdm(loader, desc="Val  ", leave=False):
        slow = clips[0].to(device)
        fast = clips[1].to(device)
        labels = labels.to(device)
        out  = model([slow, fast])
        loss = criterion(out, labels)
        total_loss += loss.item() * labels.size(0)
        preds   = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels

# ─── Run Training ─────────────────────────────────────────────────
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_val_acc = 0.0
best_weights = None

print(f"Starting SlowFast training — {NUM_EPOCHS} epochs on {DEVICE}")
print("="*60)

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

    print(f"Ep {epoch:02d}/{NUM_EPOCHS}  "
          f"tr={tr_loss:.4f}/{tr_acc:.3f}  "
          f"val={vl_loss:.4f}/{vl_acc:.3f}  "
          f"[{time.time()-t0:.0f}s]{star}")

print(f"\\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {BEST_MODEL}")
""")

# ── Cell 8: Evaluation ────────────────────────────────────────────
md("## 8. Evaluation — Curves & Confusion Matrix")
code("""
model.load_state_dict(best_weights)
_, final_acc, final_preds, final_labels = val_epoch(model, val_loader, criterion, DEVICE)

print(classification_report(final_labels, final_preds, target_names=CLASSES))

# Learning curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'SlowFast Training  (Best Val Acc: {best_val_acc:.3f})', fontsize=13, fontweight='bold')
axes[0].plot(history['train_loss'], label='Train', color='#4FC3F7')
axes[0].plot(history['val_loss'],   label='Val',   color='#FF8A65')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history['train_acc'], label='Train', color='#4FC3F7')
axes[1].plot(history['val_acc'],   label='Val',   color='#FF8A65')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'slowfast_training_curves.png'), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(final_labels, final_preds)
plt.figure(figsize=(9, 7))
sns.heatmap(cm.astype(float)/cm.sum(axis=1, keepdims=True), annot=True, fmt='.2f',
            cmap='YlOrRd', xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, annot_kws={'size':11})
plt.title('Confusion Matrix — SlowFast R50', fontsize=13)
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=30, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'slowfast_confusion_matrix.png'), dpi=150)
plt.show()
print(f"Saved to: {SAVE_DIR}")
""")

# ── Cell 9: Inference ─────────────────────────────────────────────
md("## 9. Inference Test")
code("""
@torch.no_grad()
def predict_video(video_path, model, device):
    \"\"\"Predict action on a single video using SlowFast.\"\"\"
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3,1,1,1)
    std  = torch.tensor([0.225, 0.225, 0.225]).view(3,1,1,1)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fast_idx = np.linspace(0, total-1, FAST_FRAMES, dtype=int)
    slow_idx = fast_idx[::4][:SLOW_FRAMES]

    def read_at(idx_list):
        frames = []
        for fi in idx_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, f = cap.read()
            if ret:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                f = cv2.resize(f, (FRAME_SIZE, FRAME_SIZE))
                t = torch.from_numpy(f).permute(2,0,1).float() / 255.0
            else:
                t = torch.zeros(3, FRAME_SIZE, FRAME_SIZE)
            frames.append(t)
        return torch.stack(frames, dim=1)  # (C, T, H, W)

    fast_clip = read_at(fast_idx)
    slow_clip = read_at(slow_idx)
    cap.release()

    # Normalize
    fast_clip = (fast_clip - mean) / std
    slow_clip = (slow_clip - mean) / std

    slow_in = slow_clip.unsqueeze(0).to(device)  # (1, C, 8, H, W)
    fast_in = fast_clip.unsqueeze(0).to(device)  # (1, C, 32, H, W)

    model.eval()
    out   = model([slow_in, fast_in])
    probs = torch.softmax(out, dim=1)[0]
    pred  = probs.argmax().item()

    print(f"Video: {os.path.basename(video_path)}")
    print(f"Prediction: {CLASSES[pred]}  ({probs[pred]*100:.1f}%)")
    print("\\nAll probabilities:")
    for cls, p in zip(CLASSES, probs.tolist()):
        bar = '█' * int(p * 25)
        print(f"  {cls:<30} {p*100:5.1f}%  {bar}")
    return CLASSES[pred]

# Test on a random val clip
test_path, test_label = random.choice(val_samp)
print(f"True label: {CLASSES[test_label]}\\n")
predict_video(test_path, model, DEVICE)
""")

# ── Build .ipynb ──────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "football_slowfast_training.ipynb"
        },
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = r"d:\offside\football_slowfast_training.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print(f"Notebook saved: {out_path}")
