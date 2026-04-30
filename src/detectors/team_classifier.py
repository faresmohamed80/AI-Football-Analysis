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
        # Use config ratios for shirt crop
        shirt_crop = player_crop[
            int(h*SHIRT_CROP_HEIGHT_RATIO[0]):int(h*SHIRT_CROP_HEIGHT_RATIO[1]), 
            int(w*SHIRT_CROP_WIDTH_RATIO[0]):int(w*SHIRT_CROP_WIDTH_RATIO[1])
        ]
        
        if shirt_crop.size == 0: shirt_crop = player_crop

        hsv_crop = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2HSV)

        # Calculate pixels for Team 1
        team_1_pixels = 0
        for lower, upper in self.team_1_ranges:
            mask = cv2.inRange(hsv_crop, lower, upper)
            team_1_pixels += cv2.countNonZero(mask)

        # Calculate pixels for Team 2
        team_2_pixels = 0
        for lower, upper in self.team_2_ranges:
            mask = cv2.inRange(hsv_crop, lower, upper)
            team_2_pixels += cv2.countNonZero(mask)
            
        # Calculate pixels for Referee
        referee_pixels = 0
        for lower, upper in self.referee_ranges:
            mask = cv2.inRange(hsv_crop, lower, upper)
            referee_pixels += cv2.countNonZero(mask)

        max_pixels = max(team_1_pixels, team_2_pixels, referee_pixels)

        if max_pixels < TEAM_PIXEL_THRESHOLD:
            return "Unknown", self.box_colors["Unknown"]

        if max_pixels == referee_pixels:
            return "Referee", self.box_colors["Referee"]
        elif max_pixels == team_1_pixels:
            return self.team_1_name, self.box_colors[self.team_1_name]
        else:
            return self.team_2_name, self.box_colors[self.team_2_name]