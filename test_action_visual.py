"""
Visual test for Action Recognition - saves output video with action labels
"""
import sys
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import numpy as np
import os
sys.stdout.reconfigure(encoding='utf-8')

ACTION_WEIGHTS   = r"d:\offside\weights\action.zip"
INPUT_VIDEO      = r"D:\offside\data\input_data\new_dataset_zipped\SHOT\SHOT_074.mp4"
OUTPUT_VIDEO     = r"d:\offside\data\output_data\SHOT_074_output.mp4"
ACTION_CLASSES   = [
    'BALL_PLAYER_BLOCK', 'CROSS', 'HEADER', 'HIGH_PASS',
    'PASS', 'PLAYER_SUCCESSFUL_TACKLE', 'SHOT', 'THROW_IN'
]
NUM_FRAMES  = 16
FRAME_SIZE  = (112, 112)
CLIP_STRIDE = 2
CONF_THRESH = 0.30

# Color per action (BGR)
ACTION_COLORS = {
    'PASS':                    (0, 255, 100),
    'HIGH_PASS':               (0, 200, 255),
    'CROSS':                   (255, 200, 0),
    'SHOT':                    (0, 0, 255),
    'HEADER':                  (255, 100, 0),
    'PLAYER_SUCCESSFUL_TACKLE':(0, 255, 255),
    'BALL_PLAYER_BLOCK':       (200, 0, 255),
    'THROW_IN':                (180, 180, 0),
}


def load_model():
    model = video_models.r3d_18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, len(ACTION_CLASSES))
    )
    sd = torch.load(ACTION_WEIGHTS, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model


def preprocess_clip(frames):
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    processed = []
    for f in frames:
        f = cv2.resize(f, FRAME_SIZE)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        f = (f - mean) / std
        processed.append(f)
    clip = np.stack(processed, axis=0)
    clip = np.transpose(clip, (0, 3, 1, 2))
    clip = np.expand_dims(clip, 0)
    clip = np.transpose(clip, (0, 2, 1, 3, 4))
    return torch.from_numpy(clip)


def draw_action_overlay(frame, action, confidence, all_probs):
    h, w = frame.shape[:2]
    color = ACTION_COLORS.get(action, (255, 255, 255))

    # Main action label - large text top-center
    label = f"{action}  {confidence*100:.0f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.1
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = (w - tw) // 2
    ty = 55

    # Background pill
    cv2.rectangle(frame, (tx - 16, ty - th - 10), (tx + tw + 16, ty + 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (tx - 16, ty - th - 10), (tx + tw + 16, ty + 10), color, 2)
    cv2.putText(frame, label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    # Mini bar chart - bottom left
    bar_x, bar_y = 20, h - (len(ACTION_CLASSES) * 28) - 20
    for i, (cls, prob) in enumerate(zip(ACTION_CLASSES, all_probs)):
        bar_color = ACTION_COLORS.get(cls, (200, 200, 200))
        bar_len = int(prob * 160)
        y = bar_y + i * 28
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_len, y + 18), bar_color, -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + 160, y + 18), (80, 80, 80), 1)
        cls_short = cls.replace('_', ' ')
        cv2.putText(frame, f"{cls_short} {prob*100:.0f}%",
                    (bar_x + 165, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)

    return frame


def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded!")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing {total} frames...")

    # State
    current_action = "..."
    current_conf   = 0.0
    current_probs  = [0.0] * len(ACTION_CLASSES)
    frame_buffer   = []
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_buffer.append(frame.copy())

        # Collect every CLIP_STRIDE frames into the buffer
        # Run inference when we have NUM_FRAMES * CLIP_STRIDE raw frames
        if len(frame_buffer) == NUM_FRAMES * CLIP_STRIDE:
            # Sample every CLIP_STRIDE-th frame
            sampled = frame_buffer[::CLIP_STRIDE]
            with torch.no_grad():
                tensor = preprocess_clip(sampled)
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)[0].tolist()
                top_idx = int(np.argmax(probs))
                current_action = ACTION_CLASSES[top_idx]
                current_conf   = probs[top_idx]
                current_probs  = probs

            frame_buffer = []  # Reset buffer

        # Draw on every frame
        annotated = draw_action_overlay(frame, current_action, current_conf, current_probs)
        out.write(annotated)

        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total}  Action: {current_action} ({current_conf*100:.1f}%)")

    cap.release()
    out.release()
    print(f"\nDone! Output saved to:\n{OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
