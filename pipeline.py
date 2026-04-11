import cv2
import json
import os

# استدعاء إعدادات المسارات
from src.config import *
import numpy as np
from ultralytics import YOLO

# استدعاء كل الموديلات والأنظمة
from src.detectors.player_detector import PlayerDetector
from src.detectors.number_recognizer import NumberRecognizer
from src.detectors.team_classifier import TeamClassifier
from src.trackers.number_voter import NumberVotingSystem
from src.trackers.ball_tracker import BallTracker
from src.trackers.stat_tracker import MatchStats
from src.visualizer import Visualizer
from src.trackers.radar_tracker import PitchRadar
# ---------------------------------------------------------
# وظيفة لتحميل قاعدة بيانات أسماء اللاعبين
# ---------------------------------------------------------
from src.trackers.semantic_mapper import SemanticPitchMapper

def load_player_database(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"⚠️ تحذير: ملف قاعدة البيانات غير موجود في {file_path}")
    return {}



def main():
    print("⚙️ جاري تحميل النماذج والأنظمة (Models)... يرجى الانتظار.")
    
    # 1. تهيئة النماذج الأساسية
    player_detector = PlayerDetector(PLAYER_DETECTOR_WEIGHTS)
    number_recognizer = NumberRecognizer(NUMBER_RECOGNIZER_WEIGHTS)
    team_classifier = TeamClassifier()
    
    # 2. تهيئة أنظمة التتبع والإحصائيات
    ball_tracker = BallTracker("yolo26m.pt", max_missing_frames=7)
    voter = NumberVotingSystem(required_frames=30)
    stats_tracker = MatchStats() # نظام الاستحواذ
    
    # 🔴 تحميل موديل السجمنتيشن للملعب للرادار
    pitch_segmenter = YOLO(os.path.join(BASE_DIR, "weights", "Studiam_seg.pt"))
    
    # 3. تحميل قاعدة بيانات اللاعبين
    db_path = os.path.join(BASE_DIR, "data", "players_database.json")
    player_db = load_player_database(db_path)

    # 4. فتح فيديو الإدخال
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
    frame_count = 0
     # هنحتاج نقرأ أول فريم بس عشان نعرف أبعاد الفيديو
    ret, first_frame = cap.read()
    if not ret: return
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # نرجع الفيديو للأول تاني

    # تهيئة كلاس الرادار الجديد
    radar = PitchRadar(frame_w=w, frame_h=h)
    
    # 🔴 تهيئة الـ Semantic Mapper الجديد
    semantic_mapper = SemanticPitchMapper(radar_w=radar.radar_w, radar_h=radar.radar_h)

    print("🚀 بدأ تحليل الفيديو الشامل (AI Referee & Stats System)...")

    # 5. الحلقة التكرارية لمعالجة الفيديو فريم بفريم
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # الفيديو انتهى

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"⏳ جاري معالجة الفريم رقم {frame_count}...")

        # ---------------------------------------------------------
        # تحديث إزاحة الكاميرا (Pan) عبر المعالم السيمانتيكية
        # ---------------------------------------------------------
        try:
            seg_results = pitch_segmenter(frame, verbose=False)
            dx, dy = semantic_mapper.get_camera_offset(seg_results, w, h, radar.matrix)
            radar.update_matrix(dx, dy)
        except Exception as e:
            print(f"⚠️ خطأ في حساب إزاحة الكاميرا: {e}")

        # ---------------------------------------------------------
        # أ. معالجة اللاعبين (الأساس)
        # ---------------------------------------------------------
        tracked_players = player_detector.detect(frame)
        players_data = []
        
        for track_id, bbox in tracked_players:
            # تحديد الفريق (مع عزل النجيلة)
            team_name, box_color = team_classifier.get_player_team(frame, bbox)
            
            # قراءة وتثبيت الرقم
            if track_id in voter.final_numbers:
                number = voter.final_numbers[track_id]
            else:
                predicted_number = number_recognizer.recognize(frame, bbox)
                number = voter.update(track_id, predicted_number)
            
            # ربط الرقم باسم اللاعب من الداتابيز
            if number == "Loading...":
                player_display_name = "Identifying..."
            elif number is None or number == "":
                player_display_name = "Unknown"
            else:
                player_display_name = player_db.get(str(number), f"Player #{number}")

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
        # ج. حساب الإحصائيات (الاستحواذ) 📊
        # ---------------------------------------------------------
        stats_tracker.update(players_data, ball_data)

        # ---------------------------------------------------------
        # د. الرسم على الفريم (Visualization) 🎨
        # ---------------------------------------------------------
        # 1. رسم المربعات حول اللاعبين والكرة
        annotated_frame = Visualizer.draw_annotations(frame, players_data, ball_data)
        
        # 2. رسم لوحة الإحصائيات الشفافة
        annotated_frame = stats_tracker.draw_stats(annotated_frame)

        # 3. رسم الرادار المصغر في زاوية الشاشة
        annotated_frame = radar.draw_radar(annotated_frame, players_data, ball_data)

        # 4. عرض الفريم (بعد ما جمعنا عليه كل حاجة)

        # ---------------------------------------------------------
        # هـ. حفظ الفريم
        # ---------------------------------------------------------
        out.write(annotated_frame)

    # 6. إغلاق وتحرير الملفات
    cap.release()
    out.release()
    
    # طباعة النتيجة النهائية للاستحواذ في الكونسول
    final_stats = stats_tracker.get_possession_stats()
    print(f"\n✅ تم الانتهاء! النتيجة النهائية للاستحواذ:")
    print(f"🔵 الفريق الأزرق: {final_stats['Blue Team']}%")
    print(f"⚪ الفريق الأبيض: {final_stats['White Team']}%")
    print(f"الفيديو النهائي محفوظ في: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()