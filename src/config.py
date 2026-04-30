import os

# ─────────────────────────────────────────────────────────
# 1. Base Paths
# ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────
# 2. Model Weights Paths
# ─────────────────────────────────────────────────────────
PLAYER_DETECTOR_WEIGHTS   = os.path.join(BASE_DIR, "yolo26x.pt")
BALL_DETECTOR_WEIGHTS     = os.path.join(BASE_DIR, "weights", "football ball detection", "weights", "best.pt")
NUMBER_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "jersey_recognition.pt")
STADIUM_SEGMENTER_WEIGHTS = os.path.join(BASE_DIR, "weights", "Studiam_seg.pt")
FIELD_DETECTOR_WEIGHTS    = os.path.join(BASE_DIR, "weights", "football-field-detection-15", "weights", "best.pt")

# ─────────────────────────────────────────────────────────
# 3. Confidence Thresholds (Individual for each model)
# ─────────────────────────────────────────────────────────
PLAYER_CONFIDENCE = 0.3
BALL_CONFIDENCE   = 0.7
NUMBER_CONFIDENCE = 0.2
STADIUM_CONFIDENCE = 0.2
FIELD_DETECTOR_CONFIDENCE = 0.5   # Keypoint confidence for homography

# Ball class ID: 0 = custom model (ball only), 32 = COCO pre-trained model
BALL_CLASS_ID = 0

# ─────────────────────────────────────────────────────────
# 4. Video Input/Output Settings
# ─────────────────────────────────────────────────────────
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input_data", "7.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output_data", "اخر فيديو عشان زهقت 3.mp4")

# ─────────────────────────────────────────────────────────
# 5. Asset Paths (Database & Pitch Image)
# ─────────────────────────────────────────────────────────
PLAYER_DB_PATH = os.path.join(BASE_DIR, "data", "players_database.json")
PITCH_IMAGE_PATH = os.path.join(BASE_DIR, "data", "pitch_topdown.png")

# ─────────────────────────────────────────────────────────
# 6. Possession & Game Logic Settings
# ─────────────────────────────────────────────────────────
POSSESSION_CONTACT_RADIUS = 0.55  # Legacy — kept for reference only
POSSESSION_DRIBBLE_RADIUS = 1.2   # Legacy — kept for reference only
POSSESSION_RADIUS = POSSESSION_CONTACT_RADIUS  # Legacy alias
STICKY_FRAMES = 45              # Frames to hold possession after ball leaves feet zone
REQUIRED_POSSESSION_FRAMES = 3  # Frames of consecutive contact to confirm possession
MAX_BALL_SPEED_FOR_DRIBBLING = 18

# Feet zone for bbox-based possession detection
FEET_ZONE_HEIGHT_RATIO = 0.40   # Bottom 40% of player bbox = feet/legs area
FEET_ZONE_WIDTH_EXPANSION = 0.15 # Expand bbox width by 15% on each side for ball tolerance

# ─────────────────────────────────────────────────────────
# 7. Smoothing & Filtering Settings
# ─────────────────────────────────────────────────────────
RADAR_SMOOTHING = 0.15      # Homography smoothing factor
NUMBER_VOTING_FRAMES = 10   # Frames required to lock a jersey number (was 30)
BALL_INTERPOLATION_MAX = 7  # Max frames to predict ball position when hidden

# ─────────────────────────────────────────────────────────
# 8. Ball Trail Settings
# ─────────────────────────────────────────────────────────
BALL_TRAIL_ENABLED = True
BALL_TRAIL_LENGTH = 25       # Number of historical positions to draw
BALL_TRAIL_COLOR = (0, 200, 255)  # BGR color of trail (yellow-orange)
BALL_TRAIL_THICKNESS = 3     # Max thickness of trail line at newest point


# ─────────────────────────────────────────────────────────
# 7. Team Classification Settings
# ─────────────────────────────────────────────────────────
TEAM_PIXEL_THRESHOLD = 5    # Min pixels to identify a team color
SHIRT_CROP_HEIGHT_RATIO = (0.1, 0.5) # Top/Bottom ratio for shirt crop
SHIRT_CROP_WIDTH_RATIO = (0.2, 0.8)  # Left/Right ratio for shirt crop

# ─────────────────────────────────────────────────────────
# 8. Team Color Ranges (HSV)
# ─────────────────────────────────────────────────────────
# Team 1 (White + Black Goalkeeper)
TEAM_1_HSV = [
    {"lower": [0, 0, 160],    "upper": [180, 60, 255]},  # White
    {"lower": [0, 0, 0],      "upper": [180, 255, 50]}   # Black GK
]

# Team 2 (Light Green + Orange Goalkeeper)
TEAM_2_HSV = [
    {"lower": [35, 50, 50],   "upper": [85, 255, 255]},  # Light Green
    {"lower": [10, 100, 100], "upper": [25, 255, 255]}   # Orange GK
]

# Referee (Black)
REFEREE_HSV = [
    {"lower": [0, 0, 0],      "upper": [180, 255, 50]}   # Black
]

# Team Names
TEAM_1_NAME = "White Team"
TEAM_2_NAME = "Green Team"

# Team Display Colors (BGR)
TEAM_1_DISPLAY_COLOR = (255, 255, 255) # White
TEAM_2_DISPLAY_COLOR = (144, 238, 144) # Light Green
REFEREE_DISPLAY_COLOR = (0, 0, 0)     # Black

# ─────────────────────────────────────────────────────────
# 9. Visualization Settings
# ─────────────────────────────────────────────────────────
SHOW_RADAR = True
SHOW_HEATMAPS = True
MIN_FRAMES_FOR_HEATMAP = 30

# ─────────────────────────────────────────────────────────
# 10. API & Database Settings
# ─────────────────────────────────────────────────────────
API_BASE_URL = "http://98.91.219.64:8080/api/v1"
SUPABASE_URL = "https://gsvowvzdxphlguclawur.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdzdm93dnpkeHBobGd1Y2xhd3VyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI3MzU2NDgsImV4cCI6MjA4ODMxMTY0OH0.jORXG6_6LjDP07EAhJvYb9G10AKRuKaDkCjy0SfhQe8"
SUPABASE_BUCKET = "heatmaps"  # Assuming heatmaps are uploaded here
MATCH_ID = 9  # Change this to the ID of the match to analyze