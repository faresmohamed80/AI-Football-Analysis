import os

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# مسارات النماذج (Weights)
PLAYER_DETECTOR_WEIGHTS = os.path.join(BASE_DIR, "yolo26x.pt")
NUMBER_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "jersey_recognition.pt")

# مسارات الفيديوهات
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input_data", "7.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output_data", "final_result.mp4")

# إعدادات النماذج
CONFIDENCE_THRESHOLD = 0.15 # نسبة الثقة لاعتماد إن فيه لاعب في المربع