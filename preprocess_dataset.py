"""
Pre-processing script: Ball-Possessor Player Crop for Action Recognition
- Detects all players per frame using YOLO
- Detects the ball per frame using YOLO ball detector
- Finds the player closest to the ball (the "action performer")
- Crops around that player (bbox * 1.5 padding)
- Saves a side-by-side comparison: Original | Cropped
- Processes 5 videos per label (for visual inspection)
"""
import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────
DATASET_DIR  = r"D:\offside\data\input_data\new_dataset_zipped"
OUTPUT_DIR   = r"D:\offside\data\output_data\preprocessed_preview"
PLAYER_MODEL = r"D:\offside\yolo26x.pt"
BALL_MODEL   = r"D:\offside\weights\football ball detection\weights\best.pt"

CROP_PADDING  = 1.5   # expand player bbox by this factor
VIDEOS_PER_LABEL = 5
OUTPUT_SIZE   = (224, 224)   # final crop size for the model
PREVIEW_W     = 640          # width of preview side (original)
PREVIEW_H     = 360

LABELS = [
    'BALL_PLAYER_BLOCK', 'CROSS', 'HEADER', 'HIGH_PASS',
    'PASS', 'PLAYER_SUCCESSFUL_TACKLE', 'SHOT', 'THROW_IN'
]

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
def get_ball_center(ball_model, frame):
    """Returns (cx, cy) of the ball or None."""
    results = ball_model(frame, conf=0.3, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return ((x1 + x2) // 2, (y1 + y2) // 2)
    return None


def get_player_bboxes(player_model, frame):
    """Returns list of (x1, y1, x2, y2) for all detected players."""
    results = player_model(frame, conf=0.3, verbose=False)
    bboxes = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
    return bboxes


def find_closest_player(bboxes, ball_center):
    """Returns the bbox of the player closest to the ball."""
    if not bboxes or ball_center is None:
        return None
    bx, by = ball_center
    best_bbox = None
    best_dist = float('inf')
    for (x1, y1, x2, y2) in bboxes:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = ((cx - bx)**2 + (cy - by)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_bbox = (x1, y1, x2, y2)
    return best_bbox


def pad_crop(bbox, frame_h, frame_w, padding=1.5):
    """Expands bbox by padding factor, enforces minimum size, clamped to frame bounds."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Use padding but enforce minimum 320px crop
    half_w = int((x2 - x1) * padding / 2)
    half_h = int((y2 - y1) * padding / 2)
    half = max(half_w, half_h, 160)  # minimum 320x320 crop

    nx1 = max(0, cx - half)
    ny1 = max(0, cy - half)
    nx2 = min(frame_w, cx + half)
    ny2 = min(frame_h, cy + half)
    return (nx1, ny1, nx2, ny2)


def process_video(player_model, ball_model, video_path, output_path, label):
    """Processes one video: finds action performer at clip midpoint, locks crop."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── STEP 1: Find the action performer at the midpoint ──────────────────
    mid_frame_idx = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, mid_frame = cap.read()

    anchor_bbox = None   # (x1, y1, x2, y2) of action performer

    if ret:
        ball_center = get_ball_center(ball_model, mid_frame)
        bboxes      = get_player_bboxes(player_model, mid_frame)
        player_bbox = find_closest_player(bboxes, ball_center)

        if player_bbox is not None:
            anchor_bbox = pad_crop(player_bbox, fh, fw, CROP_PADDING)
        elif bboxes:
            # Fallback: if ball not detected, use center-most player
            cx_frame, cy_frame = fw // 2, fh // 2
            bboxes_fake_ball   = bboxes
            anchor_bbox = pad_crop(find_closest_player(bboxes, (cx_frame, cy_frame)),
                                   fh, fw, CROP_PADDING)

    if anchor_bbox is None:
        # Last resort: use center square of frame
        s = min(fw, fh) // 2
        cx, cy = fw // 2, fh // 2
        anchor_bbox = (cx - s//2, cy - s//2, cx + s//2, cy + s//2)

    print(f"  Anchor bbox at frame {mid_frame_idx}: {anchor_bbox}")

    # ── STEP 2: Write all frames with the LOCKED crop ──────────────────────
    out_w = PREVIEW_W * 2
    out_h = PREVIEW_H
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind to start

    nx1, ny1, nx2, ny2 = anchor_bbox

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Fixed crop from anchor
        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (PREVIEW_W, PREVIEW_H))
        else:
            crop_resized = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

        # Draw box on original
        vis_frame = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        scale_x = PREVIEW_W / fw
        scale_y = PREVIEW_H / fh
        bx1 = int(nx1 * scale_x)
        by1 = int(ny1 * scale_y)
        bx2 = int(nx2 * scale_x)
        by2 = int(ny2 * scale_y)
        cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (0, 255, 100), 2)

        # Label overlays
        cv2.putText(vis_frame, f"ORIGINAL [{label}] f{frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(crop_resized, f"LOCKED CROP [{label}]", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 220, 255), 2)

        combined = np.hstack([vis_frame, crop_resized])
        out.write(combined)

    cap.release()
    out.release()
    print(f"  Saved: {os.path.basename(output_path)}")



# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    player_model = YOLO(PLAYER_MODEL)
    ball_model   = YOLO(BALL_MODEL)
    print("Models loaded!\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in LABELS:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            print(f"Skipping {label} - folder not found")
            continue

        videos = [f for f in os.listdir(label_dir) if f.endswith('.mp4')]
        videos = videos[:VIDEOS_PER_LABEL]

        if not videos:
            print(f"Skipping {label} - no videos found")
            continue

        out_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_label_dir, exist_ok=True)

        print(f"\n[{label}] - Processing {len(videos)} videos...")
        for vid in videos:
            in_path  = os.path.join(label_dir, vid)
            out_name = vid.replace('.mp4', '_preview.mp4')
            out_path = os.path.join(out_label_dir, out_name)
            process_video(player_model, ball_model, in_path, out_path, label)

    print(f"\nDone! All previews saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
