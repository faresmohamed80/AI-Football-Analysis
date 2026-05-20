"""
ActionRecognizer
================
Runs R3D-18 action recognition on the ball possessor only.

Usage in pipeline:
    recognizer = ActionRecognizer()
    result = recognizer.update(track_id, frame, bbox)
    # result → ('PASS', 0.91) or None
"""
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
import numpy as np
from collections import deque

ACTION_CLASSES = [
    'BALL_PLAYER_BLOCK', 'CROSS', 'HEADER', 'HIGH_PASS',
    'PASS', 'PLAYER_SUCCESSFUL_TACKLE', 'SHOT', 'THROW_IN'
]

NUM_FRAMES      = 16          # frames fed to the model
BUFFER_SIZE     = 32          # collect 32 raw frames, sample every 2nd
CROP_PADDING    = 0.4         # expand player bbox by this ratio on each side
FRAME_SIZE      = 112
CONF_THRESHOLD  = 0.35        # minimum confidence to report an action

MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD  = np.array([0.22803, 0.22145,  0.216989], dtype=np.float32)


class ActionRecognizer:
    def __init__(self, weights_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model  = self._load_model(weights_path)

        # Per-player frame buffers: track_id → deque of cropped frames
        self._buffers: dict[int, deque] = {}

        # Last prediction per player (persists until next inference)
        self._last_action: dict[int, tuple] = {}   # tid → (action, conf)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update(self, track_id: int, frame: np.ndarray, bbox: tuple):
        """
        Call every frame for the ball possessor.
        Returns (action_label, confidence) or None if buffer not ready.
        """
        crop = self._crop_player(frame, bbox)
        if crop is None:
            return self._last_action.get(track_id)

        # Init buffer for new track_id
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=BUFFER_SIZE)

        self._buffers[track_id].append(crop)

        # Run inference when buffer is full
        if len(self._buffers[track_id]) == BUFFER_SIZE:
            result = self._infer(self._buffers[track_id])
            if result[1] >= CONF_THRESHOLD:
                self._last_action[track_id] = result
                self._buffers[track_id].clear()   # reset for next clip
            else:
                result = None

            return result

        return self._last_action.get(track_id)

    def clear_player(self, track_id: int):
        """Call when a player loses possession to reset their buffer."""
        self._buffers.pop(track_id, None)

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str) -> nn.Module:
        model = video_models.r3d_18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, len(ACTION_CLASSES))
        )
        sd = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(sd)
        model.eval().to(self.device)
        print(f"[ActionRecognizer] Loaded weights from {weights_path}")
        return model

    def _crop_player(self, frame: np.ndarray, bbox: tuple):
        """Crop frame around player bbox with padding. Returns resized crop or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1

        pad_x = int(bw * CROP_PADDING)
        pad_y = int(bh * CROP_PADDING)
        nx1 = max(0, x1 - pad_x)
        ny1 = max(0, y1 - pad_y)
        nx2 = min(w, x2 + pad_x)
        ny2 = min(h, y2 + pad_y)

        crop = frame[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def _infer(self, buffer: deque) -> tuple:
        """Sample 16 frames from buffer, run model, return (action, conf)."""
        frames = list(buffer)
        # Sample evenly: every 2nd frame from 32 → 16 frames
        sampled = frames[::2][:NUM_FRAMES]
        if len(sampled) < NUM_FRAMES:
            sampled += [sampled[-1]] * (NUM_FRAMES - len(sampled))

        # Preprocess: (T, H, W, C) → (1, C, T, H, W)
        clip = np.stack(sampled, axis=0).astype(np.float32) / 255.0  # (T,H,W,C)
        clip = (clip - MEAN) / STD
        clip = np.transpose(clip, (3, 0, 1, 2))   # (C, T, H, W)
        tensor = torch.from_numpy(clip).unsqueeze(0).to(self.device)  # (1,C,T,H,W)

        with torch.no_grad():
            out   = self.model(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx   = probs.argmax().item()

        return ACTION_CLASSES[idx], float(probs[idx])
