"""
Auto Crop Tool - No text input needed.
- Goes through each label automatically (5 videos per label)
- Shows crop window on the FIRST video of each label
- Applies the same crop to the remaining 4 videos automatically
- Keeps only the middle 50 frames (action moment)

Controls in the crop window:
  - Draw rectangle with mouse
  - Press ENTER or SPACE to confirm
  - Press C to cancel/skip this label
"""
import sys
import cv2
import os
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR     = r"D:\offside\data\input_data\new_dataset_zipped"
OUTPUT_DIR      = r"D:\offside\data\output_data\manual_crops"
VIDEOS_PER_LABEL = 75
KEEP_FRAMES     = 750   # keep only middle 50 frames (action moment)
PREVIEW_W       = 1280
PREVIEW_H       = 720

LABELS_TO_PROCESS = [
     'CROSS',
    'PASS',  'SHOT', 'THROW_IN'
]


def get_mid_frame(video_path):
    """Returns the middle frame of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def select_roi(label, video_name, frame):
    """Show crop window and return selected ROI or None."""
    fw, fh = frame.shape[1], frame.shape[0]
    scale  = min(PREVIEW_W / fw, PREVIEW_H / fh, 1.0)
    dw     = int(fw * scale)
    dh     = int(fh * scale)
    display = cv2.resize(frame, (dw, dh))

    # Add instructions overlay
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (dw, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
    cv2.putText(display, f"[{label}]  {video_name}  |  Draw crop then press ENTER  |  C = skip label",
                (10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200), 1)

    roi = cv2.selectROI("Crop Tool - Draw region then press ENTER", display,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Crop Tool - Draw region then press ENTER")

    if roi == (0, 0, 0, 0):
        return None

    # Scale back to original frame coordinates
    rx, ry, rw, rh = roi
    return (int(rx / scale), int(ry / scale), int(rw / scale), int(rh / scale))


def apply_crop(video_path, output_path, roi):
    """Crop spatially + trim temporally to middle KEEP_FRAMES."""
    x, y, w, h = roi
    cap    = cv2.VideoCapture(video_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Middle KEEP_FRAMES
    if KEEP_FRAMES and KEEP_FRAMES < total:
        start_f = (total - KEEP_FRAMES) // 2
        end_f   = start_f + KEEP_FRAMES
    else:
        start_f, end_f = 0, total

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for _ in range(end_f - start_f):
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y:y+h, x:x+w]
        if cropped.size > 0:
            out.write(cropped)

    cap.release()
    out.release()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in LABELS_TO_PROCESS:
        label_dir = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(label_dir):
            print(f"Skipping {label} - not found")
            continue

        videos = sorted([f for f in os.listdir(label_dir) if f.endswith('.mp4')])
        videos = videos[:VIDEOS_PER_LABEL]

        if not videos:
            print(f"Skipping {label} - no videos")
            continue

        out_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{label}] - {len(videos)} videos")

        for vid in videos:
            in_path  = os.path.join(label_dir, vid)
            out_path = os.path.join(out_dir, vid.replace('.mp4', '_cropped.mp4'))

            mid_frame = get_mid_frame(in_path)
            if mid_frame is None:
                print(f"  Cannot read {vid}, skipping")
                continue

            roi = select_roi(label, vid, mid_frame)
            if roi is None:
                print(f"  Skipped {vid} by user")
                continue

            print(f"  Crop locked: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            apply_crop(in_path, out_path, roi)
            print(f"    Saved: {os.path.basename(out_path)}")

    print(f"\nAll done! Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
