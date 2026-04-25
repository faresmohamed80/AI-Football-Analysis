import cv2
import numpy as np

class Visualizer:
    @staticmethod
    def draw_player_base(frame, x1, y1, x2, y2, color):
        """رسم دائرة تحت أقدام اللاعب (أسلوب Supervision)"""
        center_x = int((x1 + x2) / 2)
        y_bottom = int(y2)
        
        # أبعاد البيضاوي (Ellipse)
        axes_x = int((x2 - x1) * 0.5) 
        axes_y = int(axes_x * 0.3)  
        
        # إنشاء طبقة شفافة للبيضاوي (للشياكة)
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, y_bottom), (axes_x, axes_y), 0, 0, 360, color, -1, cv2.LINE_AA)
        
        # دمج الطبقة الشفافة
        alpha = 0.3 # شفافية الحشو
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # رسم إطار خارجي حاد
        cv2.ellipse(frame, (center_x, y_bottom), (axes_x, axes_y), 0, 0, 360, color, 2, cv2.LINE_AA)
        cv2.ellipse(frame, (center_x, y_bottom), (axes_x, axes_y), 0, 0, 360, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def draw_player_label(frame, x1, y1, x2, y2, label, color):
        """رسم بطاقة تعريف اللاعب بشكل عصري (Minimalist)"""
        if not label: return
        
        # تنظيف النص لجعله أنيقاً
        display_text = label.split(' (')[0] # إزالة اسم الفريق الطويل
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.45
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
        
        center_x = int((x1 + x2) / 2)
        bg_x1 = center_x - (text_w // 2) - 8
        bg_y1 = int(y1) - text_h - 18
        bg_x2 = center_x + (text_w // 2) + 8
        bg_y2 = int(y1) - 6
        
        # رسم خلفية داكنة شفافة أو باهتة
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (25, 25, 25), -1)
        # إطار ملون سفلي فقط للمسة عصرية
        cv2.line(frame, (bg_x1, bg_y2), (bg_x2, bg_y2), color, 2)
        
        # النص
        cv2.putText(frame, display_text, (bg_x1 + 8, bg_y2 - 6), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)

    @staticmethod
    def draw_speed_distance(frame, players_data, speeds, total_dist):
        """رسم السرعة اللحظية والمسافة الكلية فوق كل لاعب بشكل عصري"""
        for p in players_data:
            tid = p.get('track_id')
            if tid is None: continue

            x1, y1, x2, y2 = p['bbox']
            color = p.get('color', (200, 200, 200))

            speed_ms = speeds.get(tid, 0.0)
            speed_kmh = speed_ms * 3.6
            dist_m = total_dist.get(tid, 0.0)

            line1 = f"{speed_kmh:.1f} km/h"
            line2 = f"{dist_m:.0f} m"

            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.4
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)

            box_w = max(w1, w2) + 14
            box_h = h1 + h2 + 18
            center_x = int((x1 + x2) / 2)
            bx1 = center_x - box_w // 2
            bx2 = center_x + box_w // 2
            by1 = int(y2) + 6
            by2 = by1 + box_h

            # خلفية شفافة
            sub = frame[by1:by2, bx1:bx2]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                black = np.full_like(sub, (20, 20, 20))
                cv2.addWeighted(black, 0.75, sub, 0.25, 0, sub)
                frame[by1:by2, bx1:bx2] = sub

            # شريط علوي بلون الفريق
            cv2.line(frame, (bx1, by1), (bx2, by1), color, 2)

            # النصوص
            cv2.putText(frame, line1, (bx1 + 7, by1 + h1 + 5),
                        font, scale, (0, 230, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, line2, (bx1 + 7, by1 + h1 + h2 + 12),
                        font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
        return frame

    @staticmethod
    def draw_annotations(frame, players_data, ball_data=None):
        for data in players_data:
            x1, y1, x2, y2 = data['bbox']
            name = data['name']
            team = data.get('team', 'Unknown')
            color = data.get('color', (0, 255, 0)) 
            
            # رسم المؤثرات العصرية بدل المستطيل
            Visualizer.draw_player_base(frame, x1, y1, x2, y2, color)
            Visualizer.draw_player_label(frame, x1, y1, x2, y2, name, color)

        # 2. رسم الكرة بطريقة عصرية ⚽
        if ball_data is not None:
            bbox, is_interpolated = ball_data
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2) + 2 

                color = (0, 0, 255) if is_interpolated else (0, 215, 255) # برتقالي متوهج للكرة الحقيقية
                
                # رسم تأثير توهج الكرة
                cv2.circle(frame, (center_x, center_y), radius+2, (0,0,0), 2, cv2.LINE_AA)
                cv2.circle(frame, (center_x, center_y), radius, color, -1 if is_interpolated else 2, cv2.LINE_AA)
                
                # مؤشر مثلث يطفو فوق الكرة
                pt1 = (center_x, center_y - radius - 8)
                pt2 = (center_x - 6, center_y - radius - 18)
                pt3 = (center_x + 6, center_y - radius - 18)
                triangle_cnt = np.array([pt1, pt2, pt3])
                cv2.drawContours(frame, [triangle_cnt], 0, color, -1, cv2.LINE_AA)
                
        return frame