"""
Test script for Action Recognition Model (R3D-18)
Classes: BALL_PLAYER_BLOCK, CROSS, HEADER, HIGH_PASS, PASS, PLAYER_SUCCESSFUL_TACKLE, SHOT, THROW_IN
"""
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import numpy as np
from src.config import INPUT_VIDEO_PATH
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────────────────
ACTION_WEIGHTS = r"d:\offside\weights\action.zip"
ACTION_CLASSES = [
    'BALL_PLAYER_BLOCK', 'CROSS', 'HEADER', 'HIGH_PASS',
    'PASS', 'PLAYER_SUCCESSFUL_TACKLE', 'SHOT', 'THROW_IN'
]
NUM_FRAMES = 16       # R3D-18 needs a clip of 16 frames
FRAME_SIZE = (112, 112)  # Standard R3D input size
CLIP_STRIDE = 2       # Read every 2nd frame for speed (covers 32 real frames)
CONFIDENCE_THRESHOLD = 0.3  # Minimum probability to show a prediction


def load_model():
    print("Loading Action Recognition Model (R3D-18)...")
    model = video_models.r3d_18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, len(ACTION_CLASSES))
    )
    state_dict = torch.load(ACTION_WEIGHTS, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model


def preprocess_clip(frames):
    """
    frames: list of BGR OpenCV frames
    Returns: tensor of shape (1, 3, T, H, W) normalized
    """
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

    processed = []
    for frame in frames:
        frame = cv2.resize(frame, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - mean) / std
        processed.append(frame)

    # (T, H, W, C) -> (T, C, H, W)
    clip = np.stack(processed, axis=0)                # (T, H, W, C)
    clip = np.transpose(clip, (0, 3, 1, 2))           # (T, C, H, W)
    clip = np.expand_dims(clip, axis=0)               # (1, T, C, H, W)
    clip = np.transpose(clip, (0, 2, 1, 3, 4))        # (1, C, T, H, W)
    return torch.from_numpy(clip)


def run_test():
    model = load_model()

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video: {INPUT_VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\nVideo: {total_frames} frames @ {fps:.1f} fps")
    print(f"Testing every {NUM_FRAMES * CLIP_STRIDE} frames (approx {NUM_FRAMES * CLIP_STRIDE / fps:.1f}s clips)\n")

    clip_count = 0
    frame_idx = 0

    while True:
        frames = []
        for i in range(NUM_FRAMES):
            for _ in range(CLIP_STRIDE):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
            if not ret:
                break
            frames.append(frame)

        if len(frames) < NUM_FRAMES:
            break

        clip_count += 1
        start_sec = (frame_idx - NUM_FRAMES * CLIP_STRIDE) / fps
        end_sec = frame_idx / fps

        with torch.no_grad():
            clip_tensor = preprocess_clip(frames)
            outputs = model(clip_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            top_prob, top_idx = probs.max(0)

        action = ACTION_CLASSES[top_idx.item()]
        prob = top_prob.item()

        if prob >= CONFIDENCE_THRESHOLD:
            status = "[OK]"
        else:
            status = "[?] "

        print(f"Clip {clip_count:02d} [{start_sec:.1f}s - {end_sec:.1f}s] {status} {action} ({prob*100:.1f}%)")

        # Print all class probabilities for the first 3 clips
        if clip_count <= 3:
            print("  All probabilities:")
            sorted_probs = sorted(zip(ACTION_CLASSES, probs.tolist()), key=lambda x: x[1], reverse=True)
            for cls, p in sorted_probs:
                bar = '█' * int(p * 20)
                print(f"    {cls:<30} {p*100:5.1f}%  {bar}")
            print()

    cap.release()
    print(f"\nDone! Tested {clip_count} clips.")


if __name__ == "__main__":
    run_test()
