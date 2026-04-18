import cv2
import numpy as np

class MatchStats:
    def __init__(self, dynamic_ratio=0.35, required_frames=4, max_speed=20, sticky_frames=45):
        self.dynamic_ratio = dynamic_ratio
        self.required_frames = required_frames
        self.max_speed = max_speed

        # --- Sticky Possession ---
        # لما فريق يمسك الكرة، يفضل في الاستحواذ لحد ما صنفنا يدكت
        # إن فريق تاني مسك الكرة فعلاً (مش مجرد سرعة الكرة عالية)
        self.sticky_frames = sticky_frames   # عدد فريمات هدوء قبل Free Ball
        self.last_possessor_team = None
        self.last_possessor_name = "Free Ball"
        self.frames_without_contact = 0      # عداد لما متلاقيش لاعب قريب

        self.team_possession_frames = {
            "Blue Team": 0,
            "White Team": 0
        }
        self.total_possession_frames = 0
        self.current_possessor = "Free Ball"

        self.candidate_team = None
        self.candidate_player = None
        self.candidate_count = 0
        self.last_ball_center = None

    def update(self, players_data, ball_data):
        # لو مفيش كرة محددة، نفضل على آخر استحواذ
        if not ball_data or ball_data[0] is None or ball_data[1] == True:
            self._carry_possession()
            return

        ball_bbox = ball_data[0]
        ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_y = (ball_bbox[1] + ball_bbox[3]) / 2
        current_ball_center = (ball_x, ball_y)

        # حساب سرعة الكرة
        ball_speed = 0
        if self.last_ball_center is not None:
            ball_speed = np.sqrt(
                (ball_x - self.last_ball_center[0])**2 +
                (ball_y - self.last_ball_center[1])**2
            )
        self.last_ball_center = current_ball_center

        # إيجاد أقرب لاعب للكرة
        min_norm_dist = float('inf')
        closest_team = None
        closest_player = "Unknown"

        for player in players_data:
            p_bbox = player['bbox']
            team = player['team']

            if team in ("Referee", "Unknown"):
                continue

            player_height = max(p_bbox[3] - p_bbox[1], 1)
            feet_x = (p_bbox[0] + p_bbox[2]) / 2
            feet_y = p_bbox[3]

            raw_dist = np.sqrt((ball_x - feet_x)**2 + (ball_y - feet_y)**2)
            norm_dist = raw_dist / player_height

            if norm_dist < min_norm_dist:
                min_norm_dist = norm_dist
                closest_team = team
                closest_player = player['name']

        # ---- منطق الاستحواذ اللاصق (Sticky Possession) ----
        contact_detected = (min_norm_dist <= self.dynamic_ratio and closest_team is not None)

        if contact_detected:
            # لاعب قريب من الكرة
            if self.candidate_team == closest_team and self.candidate_player == closest_player:
                self.candidate_count += 1
            else:
                self.candidate_team = closest_team
                self.candidate_player = closest_player
                self.candidate_count = 1

            if self.candidate_count >= self.required_frames:
                # ✅ استحواذ مؤكد
                self.last_possessor_team = closest_team
                self.last_possessor_name = f"{closest_player} ({closest_team})"
                self.frames_without_contact = 0

                self.team_possession_frames[closest_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.last_possessor_name

        elif ball_speed > self.max_speed:
            # الكرة بتتحرك بسرعة عالية (ضربة أو باسة)
            # لو فيه فريق ماسك قبل كده، نفضل ناسب الوقت له لأنه ممكن يكون سبرنت
            self._carry_possession()

        else:
            # الكرة هادية ومفيش لاعب قريب
            self.candidate_count = 0
            self._carry_possession()

    def _carry_possession(self):
        """يفضل ناسب الاستحواذ للفريق الأخير لحد ما يعدي sticky_frames بدون تواصل"""
        if self.last_possessor_team is not None:
            self.frames_without_contact += 1

            if self.frames_without_contact <= self.sticky_frames:
                # لسه في فترة السماح، نحسب الوقت للفريق الأخير
                self.team_possession_frames[self.last_possessor_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.last_possessor_name
            else:
                # انتهت فترة السماح
                self.current_possessor = "Free Ball"
        else:
            self.current_possessor = "Free Ball"

    def get_possession_stats(self):
        if self.total_possession_frames == 0:
            return {"Blue Team": 50, "White Team": 50} 
        
        green_pct = (self.team_possession_frames["Blue Team"] / self.total_possession_frames) * 100
        white_pct = (self.team_possession_frames["White Team"] / self.total_possession_frames) * 100
        
        return {"Blue Team": int(green_pct), "White Team": int(white_pct)}

    def draw_stats(self, frame):
        stats = self.get_possession_stats()
        blue_pct = stats["Blue Team"]
        white_pct = stats["White Team"]

        # إعدادات اللوحة (Panel) في الجانب الأيسر العلوي بطريقة عصرية
        x, y = 20, 20
        w, h = 330, 110 
        
        # رسم خلفية داكنة بشفافية خفيفة
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # شريط جانبي ملون للزينة (لمسة Broadcast)
        cv2.rectangle(frame, (x, y), (x + 6, y + h), (0, 215, 255), -1)

        # النصوص
        cv2.putText(frame, "L I V E  M A T C H  S T A T S", (x + 20, y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Possession: Blue {blue_pct}% | White {white_pct}%", (x + 20, y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        # شريط الاستحواذ الديناميكي (أرفع وأكثر شياكة)
        bar_y = y + 68
        cv2.rectangle(frame, (x + 20, bar_y), (x + w - 20, bar_y + 6), (220, 220, 220), -1) 
        
        blue_width = int((w - 40) * (blue_pct / 100))
        if blue_width > 0:
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + blue_width, bar_y + 6), (255, 50, 50), -1) 
            
        cv2.putText(frame, f"Ball: {self.current_possessor}", (x + 20, y + 95), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        return frame