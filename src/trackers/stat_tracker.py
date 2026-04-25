import cv2
import numpy as np

class MatchStats:
    def __init__(self, dynamic_ratio=0.35, required_frames=4, max_speed=20, sticky_frames=90):
        self.dynamic_ratio = dynamic_ratio
        self.required_frames = required_frames
        self.max_speed = max_speed

        # --- Sticky Possession ---
        self.sticky_frames = sticky_frames   
        self.last_possessor_team = None
        self.last_possessor_name = "Free Ball"
        self.frames_without_contact = 0      

        self.team_possession_frames = {
            "Red Team": 0,
            "Green Team": 0
        }
        self.total_possession_frames = 0
        self.current_possessor = "Free Ball"

        self.candidate_team = None
        self.candidate_player = None
        self.candidate_count = 0
        self.last_ball_center = None

        # --- Event Detection (Heuristics) ---
        self.event_counts = {
            "passes_red": 0,
            "passes_green": 0,
            "interceptions_red": 0,
            "interceptions_green": 0
        }
        
        # --- UI Alerts ---
        self.current_alert = None
        self.alert_frames = 0
        self.alert_color = (255, 255, 255)

    def update(self, players_data, ball_data):
        # تقليل عداد إطارات التنبيه لكي يختفي تدريجياً
        if self.alert_frames > 0:
            self.alert_frames -= 1
        else:
            self.current_alert = None

        if not ball_data or ball_data[0] is None or ball_data[1] == True:
            self._carry_possession()
            return

        ball_bbox = ball_data[0]
        ball_x = (ball_bbox[0] + ball_bbox[2]) / 2
        ball_y = (ball_bbox[1] + ball_bbox[3]) / 2
        current_ball_center = (ball_x, ball_y)

        ball_speed = 0
        if self.last_ball_center is not None:
            ball_speed = np.sqrt(
                (ball_x - self.last_ball_center[0])**2 +
                (ball_y - self.last_ball_center[1])**2
            )
        self.last_ball_center = current_ball_center

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

        # 🔴 تعديل: الاستحواذ يذهب للاعب الأقرب حتى لو الكورة بعدت شوية أثناء الجري (Dribbling)
        # 1.5 يعني الكورة في محيط متر ونص لـ 2 متر من اللاعب
        contact_detected = (min_norm_dist <= 1.5 and closest_team is not None)

        if contact_detected:
            if self.candidate_team == closest_team and self.candidate_player == closest_player:
                self.candidate_count += 1
            else:
                self.candidate_team = closest_team
                self.candidate_player = closest_player
                self.candidate_count = 1

            if self.candidate_count >= self.required_frames:
                # تحقيق استحواذ جديد مؤكد
                new_possessor_name = f"{closest_player} ({closest_team})"
                
                # --- Event Detection Logic ---
                # لو الاستحواذ القديم كان مع لاعب مختلف، حدث تغير!
                if self.last_possessor_name != "Free Ball" and self.last_possessor_name != new_possessor_name:
                    
                    if self.last_possessor_team == closest_team:
                        # نفس الفريق، لاعب مختلف -> تمريرة (Pass)
                        if closest_team == "Red Team":
                            self.event_counts["passes_red"] += 1
                        else:
                            self.event_counts["passes_green"] += 1
                            
                        self.current_alert = "NICE PASS!"
                        self.alert_color = (0, 255, 0) # أخضر للتمريرات
                        self.alert_frames = 30 # يظهر لثانية تقريباً
                        
                    elif self.last_possessor_team != closest_team:
                        # فريق مختلف -> اعتراض/قطع (Interception/Tackle)
                        # 🔥 تعديل احترافي: لا نحسبها اعتراض إلا لو كان الفريق التاني فاقد الكورة من وقت قليل جداً (أقل من ثانية)
                        # لو الكورة كانت تائهة (Free Ball) لفترة طويلة، ده بيبقى مجرد "استلام كورة تائهة" مش اعتراض
                        if self.frames_without_contact < 20: 
                            if closest_team == "Red Team":
                                self.event_counts["interceptions_red"] += 1
                                self.alert_color = (0, 0, 255) # أحمر للاعتراض
                            else:
                                self.event_counts["interceptions_green"] += 1
                                self.alert_color = (50, 255, 50) # أخضر للاعتراض
                                
                            self.current_alert = "INTERCEPTION!"
                            self.alert_frames = 35
                        else:
                            # لو الكورة بقالها كتير حرة، بنحدث الاستحواذ بس بدون احتساب "اعتراض"
                            pass
                # ------------------------------

                self.last_possessor_team = closest_team
                self.last_possessor_name = new_possessor_name
                self.frames_without_contact = 0

                self.team_possession_frames[closest_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.last_possessor_name

        elif ball_speed > self.max_speed:
            # لو الكورة سرعتها عالية جداً (باصة أو شوتة)، نفقد الاستحواذ مؤقتاً
            self._carry_possession()

        else:
            # لو الكورة سرعتها هادية وبعدت عن اللاعب، نفضل سايبينها معاه لفترة أطول (دريبل)
            self.candidate_count = 0
            self._carry_possession()

    def _carry_possession(self):
        if self.last_possessor_team is not None:
            self.frames_without_contact += 1

            if self.frames_without_contact <= self.sticky_frames:
                self.team_possession_frames[self.last_possessor_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.last_possessor_name
            else:
                self.current_possessor = "Free Ball"
        else:
            self.current_possessor = "Free Ball"

    def get_possession_stats(self):
        if self.total_possession_frames == 0:
            return {"Red Team": 50, "Green Team": 50} 
        
        red_pct = (self.team_possession_frames["Red Team"] / self.total_possession_frames) * 100
        green_pct = (self.team_possession_frames["Green Team"] / self.total_possession_frames) * 100
        
        return {"Red Team": int(red_pct), "Green Team": int(green_pct)}
        
    def get_event_stats(self):
        return self.event_counts

    def draw_stats(self, frame):
        stats = self.get_possession_stats()
        red_pct = stats["Red Team"]
        green_pct = stats["Green Team"]

        x, y = 20, 20
        w, h = 330, 110 
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (x, y), (x + 6, y + h), (0, 215, 255), -1)

        cv2.putText(frame, "L I V E  M A T C H  S T A T S", (x + 20, y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Possession: Red {red_pct}% | Green {green_pct}%", (x + 20, y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        
        bar_y = y + 68
        cv2.rectangle(frame, (x + 20, bar_y), (x + w - 20, bar_y + 6), (220, 220, 220), -1) 
        
        red_width = int((w - 40) * (red_pct / 100))
        if red_width > 0:
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + red_width, bar_y + 6), (0, 0, 255), -1) 
            
        cv2.putText(frame, f"Ball: {self.current_possessor}", (x + 20, y + 95), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        
        # رسم البوب أب (Pop-up Alert) في منتصف أسفل الشاشة إذا كان هناك تنبيه نشط
        if self.alert_frames > 0 and self.current_alert:
            # تطبيق تأثير الشفافية (Fade-out)
            alpha = min(1.0, self.alert_frames / 15.0) 
            
            alert_overlay = frame.copy()
            # خلفية داكنة خفيفة وراء الكلمة
            fh, fw = frame.shape[:2]
            text_size = cv2.getTextSize(self.current_alert, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
            box_x1 = int((fw - text_size[0]) / 2) - 20
            box_y1 = fh - 100 - text_size[1] - 10
            box_x2 = box_x1 + text_size[0] + 40
            box_y2 = fh - 100 + 10
            
            cv2.rectangle(alert_overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.addWeighted(alert_overlay, alpha * 0.6, frame, 1 - (alpha * 0.6), 0, frame)
            
            # رسم الكلمة بلون مميز
            cv2.putText(frame, self.current_alert, (box_x1 + 20, fh - 100), cv2.FONT_HERSHEY_DUPLEX, 1.2, self.alert_color, 2, cv2.LINE_AA)

        return frame