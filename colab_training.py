# ╔══════════════════════════════════════════════════════════════╗
# ║  Football Action Recognition - Fine-tuning R3D-18 on Colab  ║
# ║  Classes: CROSS, HEADER, HIGH_PASS, PASS, SHOT, THROW_IN    ║
# ║  Dataset: ~458 player-cropped video clips                    ║
# ╚══════════════════════════════════════════════════════════════╝

# ════════════════════════════════════════════════════════════════
# CELL 1 ── Install & Imports
# ════════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════════
# CELL 2 ── Mount Google Drive
# ════════════════════════════════════════════════════════════════
drive.mount('/content/drive')

# ════════════════════════════════════════════════════════════════
# CELL 3 ── Configuration
# ════════════════════════════════════════════════════════════════
# ── Dataset Path ────────────────────────────────────────────────
# Upload your 'manual_crops' folder to Google Drive, then set this:
DATASET_PATH = '/content/drive/MyDrive/manual_crops'

# ── Classes (order matters → index = class ID) ──────────────────
CLASSES = ['CROSS', 'HEADER', 'HIGH_PASS', 'PASS', 'SHOT', 'THROW_IN']
NUM_CLASSES = len(CLASSES)
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
print("Classes:", CLASS2IDX)

# ── Training Hyperparameters ─────────────────────────────────────
NUM_FRAMES   = 16      # frames sampled per clip
FRAME_SIZE   = 112     # R3D-18 input size
BATCH_SIZE   = 8       # safe for T4 GPU
NUM_EPOCHS   = 40
LR           = 1e-4    # lower LR for fine-tuning
LR_MIN       = 1e-6    # CosineAnnealing minimum
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.5
VAL_SPLIT    = 0.2     # 80/20 train/val split
SEED         = 42

# ── Save Paths ───────────────────────────────────────────────────
SAVE_DIR   = '/content/drive/MyDrive/action_model'
BEST_MODEL = os.path.join(SAVE_DIR, 'best_action_model.pt')
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ════════════════════════════════════════════════════════════════
# CELL 4 ── Dataset Class
# ════════════════════════════════════════════════════════════════
class ActionVideoDataset(Dataset):
    """
    Loads player-cropped action video clips.
    Each clip → samples NUM_FRAMES evenly → tensor (C, T, H, W)
    """

    def __init__(self, samples, transform=None):
        """
        samples: list of (video_path, label_idx) tuples
        """
        self.samples   = samples
        self.transform = transform

        # Kinetics normalization stats
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
        self.std  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3,1,1,1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_frames(path)
        if self.transform:
            frames = self.transform(frames)
        # frames: (T, H, W, C) → (T, C, H, W) → (C, T, H, W)
        tensor = frames.permute(3, 0, 1, 2).float() / 255.0  # (C,T,H,W)
        tensor = (tensor - self.mean) / self.std
        return tensor, label

    def _load_frames(self, path):
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return torch.zeros(NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3, dtype=torch.uint8)

        # Sample NUM_FRAMES evenly across the clip
        indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
        frames  = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            else:
                frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            frames.append(frame)
        cap.release()

        return torch.from_numpy(np.stack(frames))  # (T, H, W, C)


# ── Augmentation transforms ──────────────────────────────────────
def make_augment():
    """Applies augmentation to a single (H,W,C) frame."""
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ToTensor(),
    ])


class AugmentedDataset(Dataset):
    """Wraps ActionVideoDataset and applies per-frame augmentation (train only)."""

    def __init__(self, samples, augment=False):
        self.base    = ActionVideoDataset(samples)
        self.augment = make_augment() if augment else None
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
        self.std  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3,1,1,1)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        raw_frames  = self.base._load_frames(path)  # (T, H, W, C)

        frames_out = []
        for t in range(raw_frames.shape[0]):
            frame_np = raw_frames[t].numpy()  # (H, W, C)
            if self.augment:
                frame_t = self.augment(frame_np)  # (C, H, W), float
            else:
                frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
            frames_out.append(frame_t)

        # (T, C, H, W) → (C, T, H, W)
        tensor = torch.stack(frames_out, dim=1)  # (C, T, H, W)
        tensor = (tensor - self.mean) / self.std
        return tensor, label


# ════════════════════════════════════════════════════════════════
# CELL 5 ── Build Dataset Splits
# ════════════════════════════════════════════════════════════════
def collect_samples(dataset_path, classes):
    """Scan dataset folder and return list of (path, label_idx)."""
    samples = []
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        if not os.path.isdir(cls_dir):
            print(f"  WARNING: {cls} folder not found at {cls_dir}")
            continue
        vids = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
        for v in vids:
            samples.append((os.path.join(cls_dir, v), CLASS2IDX[cls]))
        print(f"  {cls}: {len(vids)} clips")
    return samples


print("Scanning dataset...")
all_samples = collect_samples(DATASET_PATH, CLASSES)
print(f"\nTotal clips: {len(all_samples)}")

# Shuffle and split
random.shuffle(all_samples)
val_size   = int(len(all_samples) * VAL_SPLIT)
train_samp = all_samples[val_size:]
val_samp   = all_samples[:val_size]
print(f"Train: {len(train_samp)}  |  Val: {len(val_samp)}")

# Class distribution
print("\nClass distribution in train set:")
from collections import Counter
train_labels = [s[1] for s in train_samp]
for idx, cls in enumerate(CLASSES):
    count = train_labels.count(idx)
    print(f"  {cls}: {count}")

# Datasets & Loaders
train_ds = AugmentedDataset(train_samp, augment=True)
val_ds   = AugmentedDataset(val_samp,   augment=False)

# Weighted sampler to handle class imbalance
class_counts  = [train_labels.count(i) for i in range(NUM_CLASSES)]
class_weights = [1.0 / c if c > 0 else 0 for c in class_counts]
sample_weights = [class_weights[label] for _, label in train_samp]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f"\nTrain batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")


# ════════════════════════════════════════════════════════════════
# CELL 6 ── Model Setup (Fine-tune R3D-18)
# ════════════════════════════════════════════════════════════════
def build_model(num_classes, dropout=0.5, freeze_layers=3):
    """
    R3D-18 pretrained on Kinetics-400.
    Freezes first `freeze_layers` ResNet blocks, trains the rest.
    """
    model = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)

    # Freeze early layers
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    for layer in layers[:freeze_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    # Replace FC head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    return model


model = build_model(NUM_CLASSES, dropout=DROPOUT, freeze_layers=2)
model = model.to(DEVICE)

# Loss with class weights for imbalance
cls_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=cls_weights_tensor)

# Optimizer: lower LR for backbone, higher for new head
head_params     = list(model.fc.parameters())
backbone_params = [p for p in model.parameters() if p.requires_grad and
                   not any(p is hp for hp in head_params)]

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LR * 0.1},
    {'params': head_params,     'lr': LR}
], weight_decay=WEIGHT_DECAY)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=LR_MIN)

print("\nModel ready!")
print(f"Optimizer: AdamW | Scheduler: CosineAnnealing | Epochs: {NUM_EPOCHS}")


# ════════════════════════════════════════════════════════════════
# CELL 7 ── Training Loop
# ════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for clips, labels in tqdm(loader, desc="Train", leave=False):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
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
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        total_loss += loss.item() * clips.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += clips.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ── Run Training ─────────────────────────────────────────────────
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
best_weights = None

print(f"\n{'='*55}")
print(f"  Starting Training — {NUM_EPOCHS} epochs on {DEVICE}")
print(f"{'='*55}\n")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    vl_loss, vl_acc, val_preds, val_labels = val_epoch(model, val_loader, criterion, DEVICE)
    scheduler.step()

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

    improved = vl_acc > best_val_acc
    if improved:
        best_val_acc = vl_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, BEST_MODEL)

    elapsed = time.time() - t0
    star = " ★" if improved else ""
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  "
          f"[{elapsed:.0f}s]{star}")


print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
print(f"Model saved to: {BEST_MODEL}")


# ════════════════════════════════════════════════════════════════
# CELL 8 ── Evaluation & Plots
# ════════════════════════════════════════════════════════════════
# Load best model for evaluation
model.load_state_dict(best_weights)
_, final_acc, final_preds, final_labels = val_epoch(model, val_loader, criterion, DEVICE)

# Classification report
print("\nClassification Report:")
print(classification_report(final_labels, final_preds, target_names=CLASSES))

# ── Learning Curves ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History', fontsize=14, fontweight='bold')

axes[0].plot(history['train_loss'], label='Train', color='#4FC3F7')
axes[0].plot(history['val_loss'],   label='Val',   color='#FF8A65')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history['train_acc'], label='Train', color='#4FC3F7')
axes[1].plot(history['val_acc'],   label='Val',   color='#FF8A65')
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].axhline(y=best_val_acc, color='green', linestyle='--', alpha=0.5,
                label=f'Best: {best_val_acc:.3f}')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
plt.show()

# ── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(final_labels, final_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, annot_kws={'size': 11})
plt.title(f'Confusion Matrix (Best Val Acc: {best_val_acc:.3f})', fontsize=13)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

print(f"\nAll outputs saved to: {SAVE_DIR}")


# ════════════════════════════════════════════════════════════════
# CELL 9 ── Quick Inference Test
# ════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_video(video_path, model, device):
    """Predict action class for a single video clip."""
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
    std  = torch.tensor([0.22803, 0.22145,  0.216989]).view(3,1,1,1)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, NUM_FRAMES, dtype=int)
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
    # (1, C, T, H, W)
    tensor = tensor.permute(0, 1, 2, 3, 4)  # already (1,C,T,H,W) — wait
    # frames: (T,H,W,C) → stack → (T,H,W,C) → permute(3,0,1,2) → (C,T,H,W)
    # unsqueeze → (1,C,T,H,W) ✓
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)

    model.eval()
    out   = model(tensor)
    probs = torch.softmax(out, dim=1)[0]
    pred  = probs.argmax().item()

    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"Prediction: {CLASSES[pred]} ({probs[pred]*100:.1f}%)")
    print("All probabilities:")
    for i, (cls, p) in enumerate(zip(CLASSES, probs.tolist())):
        bar = '█' * int(p * 25)
        print(f"  {cls:<30} {p*100:5.1f}%  {bar}")
    return CLASSES[pred], probs.tolist()


# Test on a random val sample
test_path, test_label = random.choice(val_samp)
print(f"True label: {CLASSES[test_label]}")
predict_video(test_path, model, DEVICE)
