"""
Offline Data Augmentation for Action Recognition Dataset
=========================================================
For each video, generates multiple augmented versions saved to disk.
Input:  D:/offside/data/output_data/manual_crops/<LABEL>/*.mp4
Output: D:/offside/data/output_data/augmented_dataset/<LABEL>/*.mp4

Augmentations per video (×6 new versions):
  1. _flip       → horizontal flip
  2. _speed75    → slow down (75% speed = more frames)
  3. _speed125   → speed up (125% speed = fewer frames)
  4. _bright     → increased brightness + contrast
  5. _dark       → reduced brightness
  6. _rot        → slight random rotation (-10° to +10°)
"""
import sys
import cv2
import os
import numpy as np
import random

sys.stdout.reconfigure(encoding='utf-8')

INPUT_DIR  = r"D:\offside\data\output_data\manual_crops"
OUTPUT_DIR = r"D:\offside\data\output_data\augmented_dataset"
CLASSES    = ['CROSS', 'HEADER', 'HIGH_PASS', 'PASS', 'SHOT', 'THROW_IN']

# ── Augmentation Functions ────────────────────────────────────────

def read_video(path):
    """Read all frames from a video. Returns list of BGR frames + fps."""
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def save_video(frames, path, fps, size=None):
    """Save list of frames as .mp4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    if size:
        w, h = size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        if size:
            f = cv2.resize(f, size)
        out.write(f)
    out.release()


# ── 1. Horizontal Flip ────────────────────────────────────────────
def aug_flip(frames):
    return [cv2.flip(f, 1) for f in frames]


# ── 2. Speed Change ───────────────────────────────────────────────
def aug_speed(frames, speed_factor):
    """
    speed_factor < 1 → slow down (more frames, sample with repetition)
    speed_factor > 1 → speed up (fewer frames, skip frames)
    Always returns same number of frames as input.
    """
    n = len(frames)
    # new indices to sample from original
    new_indices = np.linspace(0, n - 1, int(n / speed_factor))
    new_indices = np.clip(new_indices, 0, n - 1).astype(int)
    sampled = [frames[i] for i in new_indices]
    # now re-sample back to original length
    out_indices = np.linspace(0, len(sampled) - 1, n, dtype=int)
    return [sampled[i] for i in out_indices]


# ── 3. Brightness / Contrast ──────────────────────────────────────
def aug_brightness(frames, alpha=1.3, beta=20):
    """alpha: contrast (1.0=no change), beta: brightness."""
    result = []
    for f in frames:
        out = cv2.convertScaleAbs(f, alpha=alpha, beta=beta)
        result.append(out)
    return result


def aug_dark(frames, alpha=0.7, beta=-20):
    result = []
    for f in frames:
        out = cv2.convertScaleAbs(f, alpha=alpha, beta=beta)
        result.append(out)
    return result


# ── 4. Random Rotation ────────────────────────────────────────────
def aug_rotate(frames, angle=None):
    """Rotate all frames by a fixed random angle."""
    if angle is None:
        angle = random.uniform(-10, 10)
    h, w = frames[0].shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return [cv2.warpAffine(f, M, (w, h), borderMode=cv2.BORDER_REPLICATE) for f in frames]


# ── 5. Gaussian Noise ────────────────────────────────────────────
def aug_noise(frames, std=15):
    """Add subtle Gaussian noise — simulates compression artifacts."""
    result = []
    for f in frames:
        noise = np.random.normal(0, std, f.shape).astype(np.int16)
        out   = np.clip(f.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result.append(out)
    return result


# ── 6. Horizontal Flip + Speed ────────────────────────────────────
def aug_flip_speed(frames, speed=0.85):
    return aug_speed(aug_flip(frames), speed)


# ── Augmentation Registry ─────────────────────────────────────────
AUGMENTATIONS = {
    'flip':        lambda f: aug_flip(f),
    'speed75':     lambda f: aug_speed(f, 0.75),
    'speed125':    lambda f: aug_speed(f, 1.25),
    'bright':      lambda f: aug_brightness(f, alpha=1.3, beta=20),
    'dark':        lambda f: aug_dark(f, alpha=0.7, beta=-20),
    'rot':         lambda f: aug_rotate(f),
    'noise':       lambda f: aug_noise(f, std=12),
    'flip_speed':  lambda f: aug_flip_speed(f),
}


# ── Main Processing ───────────────────────────────────────────────
def process_class(label):
    in_dir  = os.path.join(INPUT_DIR,  label)
    out_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    videos = sorted([f for f in os.listdir(in_dir) if f.endswith('.mp4')])
    print(f"\n[{label}] {len(videos)} original videos → generating augmentations...")

    total_saved = 0

    for vid in videos:
        in_path = os.path.join(in_dir, vid)
        base    = vid.replace('.mp4', '')

        # Copy original to output dir first
        orig_out = os.path.join(out_dir, vid)
        if not os.path.exists(orig_out):
            frames, fps = read_video(in_path)
            if not frames:
                print(f"  SKIP (empty): {vid}")
                continue
            save_video(frames, orig_out, fps)
        else:
            frames, fps = read_video(in_path)
            if not frames:
                continue

        # Apply each augmentation
        for aug_name, aug_fn in AUGMENTATIONS.items():
            out_name = f"{base}_{aug_name}.mp4"
            out_path = os.path.join(out_dir, out_name)

            if os.path.exists(out_path):
                continue  # skip if already done

            try:
                aug_frames = aug_fn(frames)
                save_video(aug_frames, out_path, fps)
                total_saved += 1
            except Exception as e:
                print(f"  ERROR [{aug_name}] {vid}: {e}")

        print(f"  {vid} → +{len(AUGMENTATIONS)} versions")

    print(f"  [{label}] Total new files saved: {total_saved}")
    return len(videos), total_saved


def main():
    print("=" * 55)
    print("  Offline Data Augmentation")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Augmentations: {list(AUGMENTATIONS.keys())}")
    print("=" * 55)

    total_orig, total_new = 0, 0
    for label in CLASSES:
        orig, new = process_class(label)
        total_orig += orig
        total_new  += new

    print("\n" + "=" * 55)
    print(f"  Original videos : {total_orig}")
    print(f"  New aug videos  : {total_new}")
    print(f"  Total dataset   : {total_orig + total_new}")
    print(f"  Multiplier      : x{(total_orig + total_new) / total_orig:.1f}")
    print(f"\nDone! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
