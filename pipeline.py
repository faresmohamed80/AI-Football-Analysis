import cv2
import json
import os
import requests

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
from src.trackers.distance_speed import SpeedDistanceTracker
from src.trackers.heatmap_tracker import HeatmapTracker
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



def send_results_to_backend(match_id: int, backend_url: str, stats_tracker, speed_tracker, player_names_map: dict):
    """Sends the final match analysis results to the backend API."""
    final_stats = stats_tracker.get_possession_stats()
    event_stats = stats_tracker.get_event_stats()

    # Collect per-player speed and distance
    player_stats = []
    for track_id, distance in speed_tracker.total_distance.items():
        player_stats.append({
            "track_id": track_id,
            "player_name": player_names_map.get(track_id, f"Player #{track_id}"),
            "total_distance": round(float(distance), 2),
            "top_speed_kmh": round(float(speed_tracker.max_speeds.get(track_id, 0)) * 3.6, 2)
        })

    payload = {
        "match_id": match_id,
        "team_stats": {
            "possession": final_stats,
            "passes_red": event_stats["passes_red"],
            "passes_green": event_stats["passes_green"],
            "interceptions_red": event_stats["interceptions_red"],
            "interceptions_green": event_stats["interceptions_green"]
        },
        "player_stats": player_stats
    }

    try:
        response = requests.post(
            f"{backend_url}/api/v1/ai/analyze-match/{match_id}",
            json=payload,
            timeout=10
        )
        print(f"✅ Results sent to backend: {response.status_code}")
        print(response.json())
    except requests.exceptions.ConnectionError:
        print(f"⚠️ Could not connect to backend at {backend_url}. Results were NOT sent.")
    except Exception as e:
        print(f"⚠️ Failed to send results: {e}")


def main():
    print("⚙️ Loading models and systems... Please wait.")
    
    # 1. Initialize core detectors
    player_detector = PlayerDetector(PLAYER_DETECTOR_WEIGHTS)
    number_recognizer = NumberRecognizer(NUMBER_RECOGNIZER_WEIGHTS)
    team_classifier = TeamClassifier()
    
    # 2. Initialize tracking and stats systems
    ball_tracker = BallTracker(BALL_DETECTOR_WEIGHTS, max_missing_frames=7)
    voter = NumberVotingSystem(required_frames=30)
    stats_tracker = MatchStats() # Possession tracking system
    
    # 🔴 Loading stadium segmentation model for radar
    pitch_segmenter = YOLO(os.path.join(BASE_DIR, "weights", "Studiam_seg.pt"))
    
    # 3. Load player database
    db_path = os.path.join(BASE_DIR, "data", "players_database.json")
    player_db = load_player_database(db_path)

    # 4. Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video. Ensure it exists at: {INPUT_VIDEO_PATH}")
        return

    # إعدادات فيديو الإخراج
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    # تهيئة تتبع السرعة والمسافة والـ Heatmap (بعد قراءة الـ fps)
    speed_tracker = SpeedDistanceTracker(fps=fps)
    heatmap_tracker = HeatmapTracker(
        pitch_image_path=os.path.join(BASE_DIR, "data", "pitch_topdown.png"),
        output_dir=os.path.join(BASE_DIR, "data", "output_data", "heatmaps")
    )
    frame_count = 0
    player_names_map = {}  # {track_id: player_name} — updated each frame for the backend report
     # هنحتاج نقرأ أول فريم بس عشان نعرف أبعاد الفيديو
    ret, first_frame = cap.read()
    if not ret: return
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # نرجع الفيديو للأول تاني

    # تهيئة كلاس الرادار الجديد
    radar = PitchRadar(frame_w=w, frame_h=h)
    
    # 🔴 تهيئة الـ Semantic Mapper الجديد
    semantic_mapper = SemanticPitchMapper(radar_w=radar.radar_w, radar_h=radar.radar_h)

    print("🚀 Starting comprehensive video analysis (AI Referee & Stats System)...")

    # 🔴 Select clip to analyze (from the 1st minute for 30 seconds)
    start_time_sec = 220
    duration_sec = 20
    start_frame = int(start_time_sec * fps)
    total_frames_to_process = int(duration_sec * fps)
    end_frame = start_frame + total_frames_to_process
    
    success = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not success:
        print(f"⚠️ Warning: cap.set failed. Manually skipping to frame {start_frame}...")
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret: break

    print(f"🎞️ Analysis will process {total_frames_to_process} frames (approx {duration_sec}s).")

    # 5. Main loop for processing frame by frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
            
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Stop after processing the requested number of frames
        if frame_count >= total_frames_to_process:
            print(f"⏹️ Processed {total_frames_to_process} frames. Analysis complete.")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"⏳ Processing frame {current_frame} ({frame_count}/{total_frames_to_process})...")

        # ---------------------------------------------------------
        # تحديث إزاحة الكاميرا (Pan) عبر المعالم السيمانتيكية
        # ---------------------------------------------------------
        try:
            seg_results = pitch_segmenter(frame, verbose=False)
            dx, dy = semantic_mapper.get_camera_offset(seg_results, w, h, radar.matrix)
            radar.update_matrix(dx, dy)
        except Exception as e:
            print(f"⚠️ Camera offset calculation error: {e}")

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
                'track_id': track_id,
                'name': player_display_name,
                'team': team_name,
                'color': box_color
            })
            # Update the running name lookup for the backend report
            if player_display_name not in ("Identifying...", "Unknown"):
                player_names_map[track_id] = player_display_name

        # ---------------------------------------------------------
        # ب. معالجة الكرة ⚽
        # ---------------------------------------------------------
        ball_data = ball_tracker.track(frame)

        # ---------------------------------------------------------
        # ج. حساب الإحصائيات (الاستحواذ + السرعة + الهيت ماب) 📊
        # ---------------------------------------------------------
        stats_tracker.update(players_data, ball_data)

        # تحديث تتبع السرعة والمسافة
        tracks_for_speed = {}
        heatmap_positions = {}
        for p in players_data:
            tid = p.get('track_id')
            if tid is None: continue
            bx1, by1, bx2, by2 = p['bbox']
            feet_x = (bx1 + bx2) / 2
            feet_y = by2
            tracks_for_speed[tid] = (feet_x, feet_y)
            # إحداثيات مُطبَّعة (0-1) للـ heatmap
            label = p.get('name') or f"#{tid}"
            heatmap_positions[label] = (feet_x / width, feet_y / height)

        total_dist, speeds = speed_tracker.update(tracks_for_speed)
        heatmap_tracker.update(heatmap_positions)

        # ---------------------------------------------------------
        # د. الرسم على الفريم (Visualization) 🎨
        # ---------------------------------------------------------
        # 1. رسم المؤثرات حول اللاعبين والكرة
        annotated_frame = Visualizer.draw_annotations(frame, players_data, ball_data)

        # 2. رسم السرعة / المسافة فوق كل لاعب
        annotated_frame = Visualizer.draw_speed_distance(annotated_frame, players_data, speeds, total_dist)
        
        # 3. رسم لوحة الإحصائيات الشفافة
        annotated_frame = stats_tracker.draw_stats(annotated_frame)

        # 4. رسم الرادار المصغر في زاوية الشاشة
        annotated_frame = radar.draw_radar(annotated_frame, players_data, ball_data)

        # 4. عرض الفريم (بعد ما جمعنا عليه كل حاجة)

        # ---------------------------------------------------------
        # هـ. حفظ الفريم
        # ---------------------------------------------------------
        out.write(annotated_frame)

    # 6. إغلاق وتحرير الملفات
    cap.release()
    out.release()

    # 7. توليد صور الـ Heatmap لكل لاعب
    heatmap_tracker.generate_heatmaps(min_frames=30)
    
    final_stats = stats_tracker.get_possession_stats()
    event_stats = stats_tracker.get_event_stats()
    
    print(f"\n✅ Finished! Final Statistics:")
    print(f"🔴 Red Team Possession: {final_stats['Red Team']}%")
    print(f"🟢 Green Team Possession: {final_stats['Green Team']}%")
    
    print(f"\n🔄 Events (Passes):")
    print(f"🔴 Red Team Passes: {event_stats['passes_red']}")
    print(f"🟢 Green Team Passes: {event_stats['passes_green']}")
    
    print(f"\n⚔️ Events (Interceptions/Tackles):")
    print(f"🔴 Red Team Interceptions: {event_stats['interceptions_red']}")
    print(f"🟢 Green Team Interceptions: {event_stats['interceptions_green']}")
    print(f"\nVideo saved to: {OUTPUT_VIDEO_PATH}")

    # Send results to backend
    print("\n📡 Sending results to backend...")
    send_results_to_backend(MATCH_ID, BACKEND_URL, stats_tracker, speed_tracker, player_names_map)

if __name__ == "__main__":
    main()