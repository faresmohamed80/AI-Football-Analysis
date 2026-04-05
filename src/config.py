import os

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# مسارات النماذج (Weights)
PLAYER_DETECTOR_WEIGHTS = os.path.join(BASE_DIR, "weights", "player_detection.pt")
NUMBER_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "jersey_recognition.pt")

# مسارات الفيديوهات
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input_data", "2.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output_data", "result3.mp4")

# إعدادات النماذج
CONFIDENCE_THRESHOLD = 0.25 # نسبة الثقة لاعتماد إن فيه لاعب في المربع