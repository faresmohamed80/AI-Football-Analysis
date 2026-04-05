import cv2

class Visualizer:
    @staticmethod
    def draw_annotations(frame, players_data, ball_data=None):
        # 1. رسم اللاعبين (نفس الكود القديم)
        for data in players_data:
            x1, y1, x2, y2 = data['bbox']
            name = data['name']
            team = data.get('team', 'Unknown')
            color = data.get('color', (0, 255, 0)) 
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({team})"
            
            if name:
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_width, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 2. رسم الكرة ⚽
        if ball_data is not None:
            bbox, is_interpolated = ball_data
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                # حساب مركز الكرة ونصف قطرها
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2) + 2 

                # لو الكرة حقيقية (برتقالي)، لو متوقعة (أحمر)
                color = (0, 0, 255) if is_interpolated else (0, 165, 255)
                
                # رسم دائرة حول الكرة
                cv2.circle(frame, (center_x, center_y), radius, color, -1 if is_interpolated else 2)
                
                # كتابة "Ball" فوقها
                cv2.putText(frame, "Ball", (center_x - 15, center_y - radius - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame