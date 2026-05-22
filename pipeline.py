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
    # ─────────────────────────────────────────────────────────
    # 🔧 Mode Selection: Local or Backend
    # ─────────────────────────────────────────────────────────
    print("\n" + "═"*50)
    print("  AI Football Analysis - Mode Selection")
    print("═"*50)
    print("  1. Local Mode   → Output saved on this machine")
    print("  2. Backend Mode → Upload results to API server")
    print("═"*50)
    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ⚠️  Please enter 1 or 2")
    run_local = (choice == "1")
    mode_label = "🖥️  LOCAL" if run_local else "☁️  BACKEND"
    print(f"\n  ✅ Running in {mode_label} mode\n")

    print("⚙️ Loading models and systems... Please wait.")
    
    # --- API Integration: Fetch Match Data ---
    api = APIClient()
    try:
        match_data = api.fetch_match_data(MATCH_ID)
        home_team  = match_data["home_team"]
        away_team  = match_data["away_team"]
        player_db  = match_data["players_db"]
    except Exception as e:
        print(f"⚠️  API unavailable ({e.__class__.__name__}). Running with fallback defaults.")
        home_team = {"team_name": TEAM_1_NAME,
                     "primary_tshirt_colors": None,
                     "secondary_tshirt_colors": None,
                     "goalkeeper_tshirt_colors": None}
        away_team = {"team_name": TEAM_2_NAME,
                     "primary_tshirt_colors": None,
                     "secondary_tshirt_colors": None,
                     "goalkeeper_tshirt_colors": None}
        player_db = {}

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

    # 3. Action Recognition configuration
    # Note: Deep learning Action Recognition (R3D-18) has been replaced by physical rule-based logic in MatchStats.
    
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
    track_id_to_team = {}   # track_id → team name (for aggregating team stats)

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
            if team_name not in ('Referee', 'Unknown'):
                track_id_to_team[track_id] = team_name


        # ---------------------------------------------------------
        # ب. معالجة الكرة ⚽
        # ---------------------------------------------------------
        ball_data = ball_tracker.track(frame, players_data)

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
        stats_tracker.update(players_data, ball_data, radar_seg.matrix, radar_seg.dx, radar_seg.dy)

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
            # Store by track_id (tid) to allow proper merging at the end of the video
            heatmap_positions[tid] = (feet_x / width, feet_y / height)

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
            position="bottom-left", title="Pitch Radar",
            team_1_color=team_1_bgr, team_2_color=team_2_bgr,
            team_1_name=team_1_name, team_2_name=team_2_name
        )

        # 5. حفظ الفريم
        out.write(annotated_frame)

    # 6. إغلاق وتحرير الملفات
    cap.release()
    out.release()

    # Merge heatmap positions by player name (only for identified players)
    from collections import defaultdict
    merged_heatmap_positions = defaultdict(list)
    for tid, positions in heatmap_tracker.player_positions.items():
        p_name = track_id_to_name.get(tid)
        # Skip players whose jersey number was never identified
        if not p_name or p_name in ["Identifying...", "Unknown"]:
            continue  # ← Do NOT generate heatmap for unidentified players
        merged_heatmap_positions[p_name].extend(positions)

    heatmap_tracker.player_positions = merged_heatmap_positions
    print(f"\n🗺️  Generating heatmaps for {len(merged_heatmap_positions)} identified players...")

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
    
    print(f"Video saved to: {OUTPUT_VIDEO_PATH}")

    # --- Local Stats Output (always printed) ---
    print("\n" + "="*58)
    print("  FINAL MATCH STATISTICS")
    print("="*58)
    for tname, stats in team_stats_payload.items():
        print(f"\n  {tname}:")
        for k, v in stats.items():
            print(f"    {k:<28} {v}")

    print("\n  Player Actions (top 10):")
    player_action_stats = sorted(
        player_stats_payload,
        key=lambda x: sum(x["actions"].values()),
        reverse=True
    )
    for p in player_action_stats[:10]:
        print(f"    {p['player_name']:<25} ({p['team']}) {p['actions']}")

    # Save stats to JSON file locally
    import json as _json
    local_stats_path = os.path.join(BASE_DIR, "data", "output_data", "match_stats.json")
    local_output = {
        "team_stats":   team_stats_payload,
        "player_stats": player_stats_payload,
        "heatmap_paths": {k: str(v) for k, v in (heatmap_paths or {}).items()},
    }
    with open(local_stats_path, "w", encoding="utf-8") as f:
        _json.dump(local_output, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Stats saved locally → {local_stats_path}")

    # --- API Integration: Upload Heatmaps & Submit Results ---
    heatmap_urls = {}
    if not run_local:
        if heatmap_paths:
            for player_name, h_path in heatmap_paths.items():
                url = api.upload_heatmap(player_name, h_path)
                if url:
                    heatmap_urls[player_name] = url
                
    # Prepare player stats (speed + distance + actions)
    # Group and merge statistics by player name to prevent duplicate entries
    from collections import Counter
    merged_player_stats = {}
    
    # 1. First, map track_ids to their final names
    final_id_to_name = {}
    for tid in speed_tracker.total_distance.keys():
        p_name = track_id_to_name.get(tid, f"Player #{tid}")
        final_id_to_name[tid] = p_name
        
    for tid in stats_tracker.player_actions.keys():
        if tid not in final_id_to_name:
            p_name = track_id_to_name.get(tid, f"Player #{tid}")
            final_id_to_name[tid] = p_name

    # 2. Iterate and merge
    for tid, t_name in final_id_to_name.items():
        t_team = track_id_to_team.get(tid, "Unknown")
        t_dist = speed_tracker.total_distance.get(tid, 0.0)
        t_speed = speed_tracker.top_speeds.get(tid, 0.0)
        tid_actions = stats_tracker.player_actions.get(tid, Counter())
        
        if t_name in ["Identifying...", "Unknown"]:
            t_name = f"Player #{tid}"
            
        if t_name not in merged_player_stats:
            merged_player_stats[t_name] = {
                "track_id": int(tid),
                "player_name": t_name,
                "team": t_team,
                "total_distance": float(t_dist),
                "top_speed": float(t_speed),
                "actions": Counter(tid_actions)
            }
        else:
            entry = merged_player_stats[t_name]
            if entry["team"] == "Unknown" and t_team != "Unknown":
                entry["team"] = t_team
            entry["total_distance"] += float(t_dist)
            entry["top_speed"] = max(entry["top_speed"], float(t_speed))
            entry["actions"].update(tid_actions)

    player_stats_payload = []
    for p_name, data in merged_player_stats.items():
        data["actions"] = dict(data["actions"])
        player_stats_payload.append(data)

    # ── Comprehensive Team Stats ───────────────────────────────────────
    possession_stats  = stats_tracker.get_possession_stats()
    team_action_stats = stats_tracker.get_team_action_stats()

    team_stats_payload = {}
    for tname in [team_1_name, team_2_name]:
        is_t1 = (tname == team_1_name)

        # Get merged players belonging to this team
        team_players = [p for p in player_stats_payload if p["team"] == tname]
        team_dists   = [p["total_distance"] for p in team_players]
        team_speeds  = [p["top_speed"] for p in team_players]

        team_stats_payload[tname] = {
            # Possession
            "possession_pct":    possession_stats.get(tname, 0),
            # Distance
            "total_distance_km": round(sum(team_dists) / 1000, 2),
            "avg_distance_km":   round((sum(team_dists) / len(team_dists) / 1000) if team_dists else 0, 2),
            # Speed
            "top_speed_kmh":     round(max(team_speeds) if team_speeds else 0, 1),
            "avg_top_speed_kmh": round((sum(team_speeds) / len(team_speeds)) if team_speeds else 0, 1),
            # Traditional events
            "passes":            event_stats["passes_t1" if is_t1 else "passes_t2"],
            "interceptions":     event_stats["inter_t1"  if is_t1 else "inter_t2"],
            # Action model events (SHOT, HEADER, CROSS, etc.)
            **{f"action_{k.lower()}": v
               for k, v in team_action_stats.get(tname, {}).items()},
        }

    # ── Print full summary ─────────────────────────────────────────────
    print("\n" + "="*58)
    print("  FINAL MATCH STATISTICS")
    print("="*58)
    for tname, stats in team_stats_payload.items():
        print(f"\n  {tname}:")
        for k, v in stats.items():
            print(f"    {k:<28} {v}")

    print("\n  Player Actions (top 10):")
    player_action_stats = sorted(
        player_stats_payload, 
        key=lambda x: sum(x["actions"].values()), 
        reverse=True
    )
    for p in player_action_stats[:10]:
        print(f"    {p['player_name']:<25} ({p['team']}) {p['actions']}")

    if not run_local:
        api.submit_ai_results(
            match_id=MATCH_ID,
            final_stats=final_stats,
            event_stats=event_stats,
            player_stats=player_stats_payload,
            heatmap_urls=heatmap_urls,
            team_stats=team_stats_payload
        )
        print("\n☁️  Results submitted to backend API successfully!")
    else:
        print("\n🖥️  Local mode: Skipped API upload. All results saved on machine.")

if __name__ == "__main__":
    main()
