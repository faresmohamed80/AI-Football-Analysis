import cv2
import numpy as np
from src.config import (
    TEAM_PIXEL_THRESHOLD, 
    SHIRT_CROP_HEIGHT_RATIO, 
    SHIRT_CROP_WIDTH_RATIO,
    TEAM_1_HSV,
    TEAM_2_HSV,
    REFEREE_HSV,
    TEAM_1_NAME,
    TEAM_2_NAME,
    TEAM_1_DISPLAY_COLOR,
    TEAM_2_DISPLAY_COLOR,
    REFEREE_DISPLAY_COLOR
)

class TeamClassifier:
    def __init__(self, team_1_name=TEAM_1_NAME, team_2_name=TEAM_2_NAME, 
                 team_1_hsv=TEAM_1_HSV, team_2_hsv=TEAM_2_HSV,
                 team_1_bgr=TEAM_1_DISPLAY_COLOR, team_2_bgr=TEAM_2_DISPLAY_COLOR):
        
        self.team_1_name = team_1_name
        self.team_2_name = team_2_name
        
        # Convert list of ranges from config or api to numpy arrays
        self.team_1_ranges = [
            (np.array(r["lower"]), np.array(r["upper"])) for r in team_1_hsv
        ]
        self.team_2_ranges = [
            (np.array(r["lower"]), np.array(r["upper"])) for r in team_2_hsv
        ]
        self.referee_ranges = [
            (np.array(r["lower"]), np.array(r["upper"])) for r in REFEREE_HSV
        ]

        self.box_colors = {
            self.team_1_name: team_1_bgr,
            self.team_2_name: team_2_bgr,
            "Referee": REFEREE_DISPLAY_COLOR,
            "Unknown": (128, 128, 128)
        }

    def get_player_team(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return "Unknown", self.box_colors["Unknown"]

        h, w = player_crop.shape[:2]
        aspect_ratio = h / w if w > 0 else 0

        # Adaptive crop ratios based on aspect ratio (top-down vs. vertical view)
        if aspect_ratio < 1.35:
            # Top-down perspective (birds-eye view): player looks wider and shorter.
            # Head is at the top/middle. We crop wider horizontally and lower vertically to capture shoulders/chest.
            crop_y_start = int(h * 0.25)
            crop_y_end   = int(h * 0.85)
            crop_x_start = int(w * 0.15)
            crop_x_end   = int(w * 0.85)
        else:
            # Normal perspective: player is tall and narrow.
            crop_y_start = int(h * SHIRT_CROP_HEIGHT_RATIO[0])
            crop_y_end   = int(h * SHIRT_CROP_HEIGHT_RATIO[1])
            crop_x_start = int(w * SHIRT_CROP_WIDTH_RATIO[0])
            crop_x_end   = int(w * SHIRT_CROP_WIDTH_RATIO[1])

        shirt_crop = player_crop[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        if shirt_crop.size == 0: 
            shirt_crop = player_crop

        hsv_crop = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2HSV)
        total_pixels = hsv_crop.shape[0] * hsv_crop.shape[1]
        if total_pixels == 0:
            return "Unknown", self.box_colors["Unknown"]

        # Calculate pixel RATIO for each team (normalised by crop size)
        def _pixel_ratio(ranges, target_hsv):
            count = 0
            for lower, upper in ranges:
                mask = cv2.inRange(target_hsv, lower, upper)
                count += cv2.countNonZero(mask)
            return count / (target_hsv.shape[0] * target_hsv.shape[1])  # 0.0 – 1.0

        team_1_ratio   = _pixel_ratio(self.team_1_ranges, hsv_crop)
        team_2_ratio   = _pixel_ratio(self.team_2_ranges, hsv_crop)
        referee_ratio  = _pixel_ratio(self.referee_ranges, hsv_crop)

        max_ratio = max(team_1_ratio, team_2_ratio, referee_ratio)

        # Require at least 5% matching pixels
        MIN_RATIO = 0.05

        # ── Secondary Fallback Crop if primary crop has low confidence ──
        if max_ratio < MIN_RATIO:
            # Crop the middle 60% of the player's full bbox
            fallback_crop = player_crop[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
            if fallback_crop.size > 0:
                fallback_hsv = cv2.cvtColor(fallback_crop, cv2.COLOR_BGR2HSV)
                fb_total = fallback_hsv.shape[0] * fallback_hsv.shape[1]
                if fb_total > 0:
                    fb_t1  = _pixel_ratio(self.team_1_ranges, fallback_hsv)
                    fb_t2  = _pixel_ratio(self.team_2_ranges, fallback_hsv)
                    fb_ref = _pixel_ratio(self.referee_ranges, fallback_hsv)
                    fb_max = max(fb_t1, fb_t2, fb_ref)
                    if fb_max >= MIN_RATIO:
                        max_ratio = fb_max
                        team_1_ratio = fb_t1
                        team_2_ratio = fb_t2
                        referee_ratio = fb_ref

        if max_ratio < MIN_RATIO:
            return "Unknown", self.box_colors["Unknown"]

        if max_ratio == referee_ratio:
            return "Referee", self.box_colors["Referee"]
        elif max_ratio == team_1_ratio:
            return self.team_1_name, self.box_colors[self.team_1_name]
        else:
            return self.team_2_name, self.box_colors[self.team_2_name]