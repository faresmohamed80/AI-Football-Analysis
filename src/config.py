import os

# المسارات الأساسية
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# مسارات النماذج (Weights)
PLAYER_DETECTOR_WEIGHTS  = r"D:\offside\weights\detection Fustal\weights\best.pt"    # 250 epoch — يفرق بين لاعب وحكم
BALL_DETECTOR_WEIGHTS    = r"D:\offside\weights\detect_ball_results\weights\best.pt"
PITCH_POSE_WEIGHTS       = r"D:\offside\weights\Homography Estimation Fustal\weights\best.pt"  # 500 epoch — keypoints ملعب الصالات
NUMBER_RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "weights", "jersey_recognition.pt")

# مسارات الفيديوهات
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input_data", "5_side.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output_data", "5_side_output.mp4")

# إعدادات النماذج
CONFIDENCE_THRESHOLD = 0.25 # نسبة الثقة لاعتماد إن فيه لاعب في المربع
MIN_INTERSECT_AREA = 0.1  # 10% مساحة تقاطع اللاعب
MIN_PASS_LENGTH = 25     # طول الخط الأدنى (بكسل) لاعتباره تمريرة
LINE_COLOR = (0, 255, 0) # لون الخط أخضر
LINE_THICKNESS = 2       # سمك الخط

# إعدادات الـ Backend API
BACKEND_URL = "http://localhost:8000"  # غير الـ URL ده لو السيرفر بتاعك مختلف
MATCH_ID = 1                           # ID المباراة الحالية