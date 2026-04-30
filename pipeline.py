import cv2
import json
import os
from collections import deque

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

from src.api_client import APIClient, hex_to_hsv, hex_to_bgr

# ---------------------------------------------------------
# (API will handle player data now)
# ---------------------------------------------------------

def main():
    print("⚙️ Loading models and systems... Please wait.")
    
    # --- API Integration: Fetch Match Data ---
    api = APIClient()
    match_data = api.fetch_match_data(MATCH_ID)
    
    home_team = match_data["home_team"]
    away_team = match_data["away_team"]
    player_db = match_data["players_db"]

    team_1_name = home_team.get("team_name") or TEAM_1_NAME
    team_2_name = away_team.get("team_name") or TEAM_2_NAME
    
    # Fetch ALL color ranges from DB (primary + secondary + GK)
    home_hsv_list = [
        hex_to_hsv(home_team.get("primary_tshirt_colors")),
        hex_to_hsv(home_team.get("secondary_tshirt_colors")),
        hex_to_hsv(home_team.get("goalkeeper_tshirt_colors")),
    ]
    away_hsv_list = [
        hex_to_hsv(away_team.get("primary_tshirt_colors")),
        hex_to_hsv(away_team.get("secondary_tshirt_colors")),
        hex_to_hsv(away_team.get("goalkeeper_tshirt_colors")),
    ]
    
    # Filter out None values (NULL entries from DB)
    team_1_hsv = [r for r in home_hsv_list if r is not None] or TEAM_1_HSV
    team_2_hsv = [r for r in away_hsv_list if r is not None] or TEAM_2_HSV
    
    print(f"Team 1 ({team_1_name}) HSV Ranges: {len(team_1_hsv)} colors loaded from DB")
    print(f"Team 2 ({team_2_name}) HSV Ranges: {len(team_2_hsv)} colors loaded from DB")
    
    team_1_bgr = hex_to_bgr(home_team.get("primary_tshirt_colors")) if home_team.get("primary_tshirt_colors") else TEAM_1_DISPLAY_COLOR
    team_2_bgr = hex_to_bgr(away_team.get("primary_tshirt_colors")) if away_team.get("primary_tshirt_colors") else TEAM_2_DISPLAY_COLOR
    # -----------------------------------------

    # 1. Initialize core detectors
    player_detector = PlayerDetector(PLAYER_DETECTOR_WEIGHTS)
    number_recognizer = NumberRecognizer(NUMBER_RECOGNIZER_WEIGHTS)
    team_classifier = TeamClassifier(
        team_1_name=team_1_name, team_2_name=team_2_name,
        team_1_hsv=team_1_hsv, team_2_hsv=team_2_hsv,
        team_1_bgr=team_1_bgr, team_2_bgr=team_2_bgr
    )
    
    # 2. Initialize tracking and stats systems
    ball_tracker = BallTracker(BALL_DETECTOR_WEIGHTS, max_missing_frames=BALL_INTERPOLATION_MAX)
    voter = NumberVotingSystem(required_frames=NUMBER_VOTING_FRAMES)
    stats_tracker = MatchStats(
        team_1_name=team_1_name, team_2_name=team_2_name,
        team_1_color=team_1_bgr, team_2_color=team_2_bgr
    )
    
    # 🔴 Loading stadium segmentation model for radar
    pitch_segmenter = YOLO(STADIUM_SEGMENTER_WEIGHTS)

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
        pitch_image_path=PITCH_IMAGE_PATH,
        output_dir=os.path.join(BASE_DIR, "data", "output_data", "heatmaps")
    )
    frame_count = 0
    ball_trail = deque(maxlen=BALL_TRAIL_LENGTH)  # Ball position history for trail
     # هنحتاج نقرأ أول فريم بس عشان نعرف أبعاد الفيديو
    ret, first_frame = cap.read()
    if not ret: return
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # نرجع الفيديو للأول تاني

    # تهيئة رادار الملعب
    radar_seg = PitchRadar(frame_w=w, frame_h=h, radar_w=400, radar_h=240)
    
    # 🔴 Semantic Mapper
    semantic_mapper = SemanticPitchMapper(
        radar_w=radar_seg.radar_w, 
        radar_h=radar_seg.radar_h, 
        smoothing=RADAR_SMOOTHING
    )

    # 🔴 Select clip to analyze
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_video_frames / fps
    
    start_time_sec = 350 # Requested start
    duration_sec = 30
    
    if start_time_sec >= video_duration_sec:
        print(f"⚠️ Warning: Start time ({start_time_sec}s) is beyond video duration ({video_duration_sec:.1f}s). Resetting to 0.")
        start_time_sec = 0
        
    start_frame = int(start_time_sec * fps)
    total_frames_to_process = int(duration_sec * fps)
    end_frame = start_frame + total_frames_to_process
    
    print(f"🎞️ Video duration: {video_duration_sec:.1f}s. Processing {total_frames_to_process} frames starting from {start_time_sec}s.")
    
    success = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not success:
        print(f"⚠️ Warning: cap.set failed. Manually skipping to frame {start_frame}...")
        for _ in range(start_frame):
            ret, _ = cap.read()
            if not ret: break

    print(f"🎞️ Analysis will process {total_frames_to_process} frames (approx {duration_sec}s).")

    track_id_to_name = {}

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
        # تحديث إزاحة الكاميرا (Pan) عبر المعالم السيمانتيكية (Segmentation)
        # ---------------------------------------------------------
        try:
            seg_results = pitch_segmenter(frame, conf=STADIUM_CONFIDENCE, verbose=False)
            dx, dy = semantic_mapper.get_camera_offset(seg_results, w, h, radar_seg.matrix)
            radar_seg.update_matrix(dx, dy)
        except Exception as e:
            print(f"⚠️ Semantic offset calculation error: {e}")

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
                if player_display_name and player_display_name not in ["Identifying...", "Unknown"]:
                    track_id_to_name[track_id] = player_display_name

            players_data.append({
                'bbox': bbox,
                'track_id': track_id,
                'name': player_display_name,
                'team': team_name,
                'color': box_color
            })

        # ---------------------------------------------------------
        # ب. معالجة الكرة ⚽
        # ---------------------------------------------------------
        ball_data = ball_tracker.track(frame)

        # Update ball trail with current real position (skip interpolated)
        if ball_data is not None:
            bbox, is_interpolated = ball_data
            if bbox is not None and not is_interpolated:
                bx1, by1, bx2, by2 = bbox
                ball_cx = (bx1 + bx2) // 2
                ball_cy = (by1 + by2) // 2
                ball_trail.append((ball_cx, ball_cy))

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
        # 1. Draw player/ball annotations (supervision)
        annotated_frame = Visualizer.draw_annotations(
            frame, players_data, ball_data,
            possessor_name=stats_tracker.current_possessor,
            ball_trail=ball_trail
        )

        # 2. رسم السرعة / المسافة فوق كل لاعب
        annotated_frame = Visualizer.draw_speed_distance(annotated_frame, players_data, speeds, total_dist)
        
        # 3. رسم لوحة الإحصائيات الشفافة
        annotated_frame = stats_tracker.draw_stats(annotated_frame)

        # 4. رسم الرادار
        annotated_frame = radar_seg.draw_radar(
            annotated_frame, players_data, ball_data, 
            position="bottom-left", title="Pitch Radar"
        )

        # 4. عرض الفريم (بعد ما جمعنا عليه كل حاجة)

        # ---------------------------------------------------------
        # هـ. حفظ الفريم
        # ---------------------------------------------------------
        out.write(annotated_frame)

    # 6. إغلاق وتحرير الملفات
    cap.release()
    out.release()

    # 7. Generate Heatmap images for each player
    heatmap_paths = heatmap_tracker.generate_heatmaps(min_frames=MIN_FRAMES_FOR_HEATMAP)
    
    final_stats = stats_tracker.get_possession_stats()
    event_stats = stats_tracker.get_event_stats()
    
    print(f"\n✅ Finished! Final Statistics:")
    print(f"🔹 {team_1_name} Possession: {final_stats.get(team_1_name, 0)}%")
    print(f"🔸 {team_2_name} Possession: {final_stats.get(team_2_name, 0)}%")
    
    print(f"\n🔄 Events (Passes):")
    print(f"🔹 {team_1_name} Passes: {event_stats['passes_t1']}")
    print(f"🔸 {team_2_name} Passes: {event_stats['passes_t2']}")
    
    print(f"\n⚔️ Events (Interceptions/Tackles):")
    print(f"🔹 {team_1_name} Interceptions: {event_stats['inter_t1']}")
    print(f"🔸 {team_2_name} Interceptions: {event_stats['inter_t2']}")
    
    print(f"\nVideo saved to: {OUTPUT_VIDEO_PATH}")

    # --- API Integration: Upload Heatmaps & Submit Results ---
    heatmap_urls = {}
    if heatmap_paths:
        for player_name, h_path in heatmap_paths.items():
            url = api.upload_heatmap(player_name, h_path)
            if url:
                heatmap_urls[player_name] = url
                
    # Prepare player stats
    player_stats_payload = []
    
    for tid, t_dist in speed_tracker.total_distance.items():
        p_name = track_id_to_name.get(tid, f"Player #{tid}")
        t_speed = speed_tracker.top_speeds.get(tid, 0)
        
        player_stats_payload.append({
            "track_id": int(tid),
            "player_name": p_name,
            "total_distance": float(t_dist),
            "top_speed": float(t_speed)
        })

    api.submit_ai_results(
        match_id=MATCH_ID,
        final_stats=final_stats,
        event_stats=event_stats,
        player_stats=player_stats_payload,
        heatmap_urls=heatmap_urls
    )

if __name__ == "__main__":
    main()