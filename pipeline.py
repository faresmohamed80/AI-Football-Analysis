import cv2
import json
import os

# استدعاء إعدادات المسارات (تأكد إنها موجودة في src/config.py)
from src.config import *

# استدعاء كل الموديلات والأنظمة اللي بنيناها
from src.detectors.player_detector import PlayerDetector
from src.detectors.number_recognizer import NumberRecognizer
from src.detectors.team_classifier import TeamClassifier
from src.trackers.number_voter import NumberVotingSystem
from src.trackers.ball_tracker import BallTracker
from src.visualizer import Visualizer

# ---------------------------------------------------------
# وظيفة لتحميل قاعدة بيانات أسماء اللاعبين من ملف JSON
# ---------------------------------------------------------
def load_player_database(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"⚠️ تحذير: ملف قاعدة البيانات غير موجود في {file_path}")
    return {}

def main():
    print("⚙️ جاري تحميل النماذج (Models)... يرجى الانتظار.")
    
    # 1. تهيئة جميع النماذج والأنظمة (Initialization)
    player_detector = PlayerDetector(PLAYER_DETECTOR_WEIGHTS)
    number_recognizer = NumberRecognizer(NUMBER_RECOGNIZER_WEIGHTS)
    team_classifier = TeamClassifier()
    
    # ⚽ تهيئة متتبع الكرة باستخدام موديل yolo26 اللي طلبته 
    # (الرقم 7 هو عدد الفريمات اللي هيتوقعها لو الكورة اختفت)
    ball_tracker = BallTracker("yolo26n.pt", max_missing_frames=7)
    
    # تهيئة نظام التصويت للأرقام (هياخد قرار بعد 30 فريم)
    voter = NumberVotingSystem(required_frames=30)
    
    # تحميل قاعدة بيانات أسماء اللاعبين
    db_path = os.path.join(BASE_DIR, "data", "players_database.json")
    player_db = load_player_database(db_path)

    # 2. فتح فيديو الإدخال
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ خطأ: مش قادر أفتح الفيديو. تأكد من وجوده في: {INPUT_VIDEO_PATH}")
        return

    # إعدادات فيديو الإخراج
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print("🚀 بدأ تحليل الفيديو الشامل (AI Referee & Stats System)...")
    frame_count = 0

    # 3. الحلقة التكرارية لمعالجة الفيديو فريم بفريم
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # الفيديو انتهى

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"⏳ جاري معالجة الفريم رقم {frame_count}...")

        # ---------------------------------------------------------
        # أ. معالجة اللاعبين (الأساس)
        # ---------------------------------------------------------
        tracked_players = player_detector.detect(frame)
        players_data = []
        
        for track_id, bbox in tracked_players:
            
            # --- 1. تحديد الفريق (مع عزل النجيلة) ---
            team_name, box_color = team_classifier.get_player_team(frame, bbox)
            
            # --- 2. قراءة وتثبيت الرقم ---
            if track_id in voter.final_numbers:
                number = voter.final_numbers[track_id]
            else:
                predicted_number = number_recognizer.recognize(frame, bbox)
                number = voter.update(track_id, predicted_number)
            
            # --- 3. ربط الرقم باسم اللاعب من الداتابيز ---
            if number == "Loading...":
                player_display_name = "Identifying..."
            elif number is None or number == "":
                player_display_name = "Unknown"
            else:
                player_display_name = player_db.get(str(number), f"Player #{number}")

            # تجميع بيانات اللاعب لتمريرها للرسم
            players_data.append({
                'bbox': bbox, 
                'name': player_display_name,
                'team': team_name,
                'color': box_color
            })

        # ---------------------------------------------------------
        # ب. معالجة الكرة ⚽
        # ---------------------------------------------------------
        ball_data = ball_tracker.track(frame)

        # ---------------------------------------------------------
        # ج. الرسم على الفريم (Visualization)
        # ---------------------------------------------------------
        annotated_frame = Visualizer.draw_annotations(frame, players_data, ball_data)

        # د. حفظ الفريم في الفيديو الجديد
        out.write(annotated_frame)

    # 4. إغلاق وتحرير الملفات
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ تم الانتهاء من المعالجة بنجاح! الفيديو النهائي محفوظ في: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()