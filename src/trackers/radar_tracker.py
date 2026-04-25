import cv2
import numpy as np

class PitchRadar:
    # 🔴 التعديل هنا: قمنا بتغيير الـ Ratio الافتراضي ليكون أفقيًا (Radar_w > Radar_h)
    def __init__(self, frame_w, frame_h, radar_w=500, radar_h=300):
        self.radar_w = radar_w
        self.radar_h = radar_h

        # نقاط افتراضية مبدئية لكي يعمل الرادار إذا فشل الموديل في أول فريم
        self.src_pts = np.float32([
            [415, 320],   
            [1505, 320],  
            [1880, 950],  
            [40, 950]     
        ])

        # 2. نقاط الرادار (المستطيل المثالي)
        # تم تعديلها لتربط الشاشة بـ 55% من مساحة الرادار لتعكس البث التلفزيوني بدقة (تجنب الشد الأفقية)
        pitch_view_width = radar_w * 0.55
        offset = (radar_w - pitch_view_width) / 2.0
        
        self.dst_pts = np.float32([
            [offset, 0],
            [offset + pitch_view_width, 0],
            [offset + pitch_view_width, radar_h],
            [offset, radar_h]
        ])

        # حساب المصفوفة المبدئية للرادار العادي لل fallback
        self.matrix = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.dx = 0
        self.dy = 0

    def set_matrix(self, matrix):
        """تحديث المصفوفة بالكامل (مثل هوموجرافي المفاصل)"""
        if matrix is not None:
            self.matrix = matrix
            self.dx = 0 # تصفير الإزاحة لأن المصفوفة الجديدة تشمل كل شيء
            self.dy = 0

    def update_matrix(self, dx, dy):
        """تحديث زاوية الرادار ديناميكياً باستخدام إزاحة الكاميرا (Panning)"""
        if dx is not None and dy is not None:
            self.dx = dx
            self.dy = dy

    def draw_radar(self, frame, players_data, ball_data):
        # إنشاء خلفية خضراء للرادار
        radar_img = np.zeros((self.radar_h, self.radar_w, 3), dtype=np.uint8)
        radar_img[:] = (34, 139, 34) # Forest Green
        
        # رسم خطوط الملعب البيضاء (للشياكة) - تم تعديلها لتكون أفقية
        cv2.rectangle(radar_img, (0, 0), (self.radar_w, self.radar_h), (255, 255, 255), 2) # الإطار
        cv2.line(radar_img, (self.radar_w // 2, 0), (self.radar_w // 2, self.radar_h), (255, 255, 255), 2) # خط النص
        cv2.circle(radar_img, (self.radar_w // 2, self.radar_h // 2), 40, (255, 255, 255), 2) # دايرة السنتر

        # رسم اللاعبين
        for player in players_data:
            p_bbox = player['bbox']
            team = player['team']
            if team == "Referee" or team == "Unknown": continue
            
            # بناخد نقطة المنتصف السفلية (بين قدمي اللاعب)
            feet_x = (p_bbox[0] + p_bbox[2]) / 2
            feet_y = p_bbox[3]
            
            # تحويل النقطة من الكاميرا للرادار
            pt = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(pt, self.matrix)
            rx, ry = int(transformed_pt[0][0][0]), int(transformed_pt[0][0][1])

            # تطبيق الإزاحة الناتجة عن دوران الكاميرا
            rx += self.dx
            ry += self.dy

            # تحديد اللون (أزرق وأبيض وأسود)
            color = (255, 0, 0) if team == "Blue Team" else (255, 255, 255)
            
            # رسم الدائرة لو النقطة جوه الرادار
            if 0 <= rx <= self.radar_w and 0 <= ry <= self.radar_h:
                cv2.circle(radar_img, (rx, ry), 5, color, -1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 7, (220, 220, 220), 1, cv2.LINE_AA) 
                cv2.circle(radar_img, (rx, ry), 9, (0, 0, 0), 1, cv2.LINE_AA) 

        # دمج الرادار في الركن السفلي الأيمن من الفيديو الأصلي بطريقة عصرية (شفافية)
        h, w = frame.shape[:2]
        
        radar_x = w - self.radar_w - 20
        radar_y = h - self.radar_h - 20
        
        # إطار أبيض أنيق للرادار
        cv2.rectangle(radar_img, (0,0), (self.radar_w-1, self.radar_h-1), (220, 220, 220), 2, cv2.LINE_AA)
        
        # وضع الرادار بشفافية
        roi = frame[radar_y:radar_y+self.radar_h, radar_x:radar_x+self.radar_w]
        cv2.addWeighted(radar_img, 0.85, roi, 0.15, 0, roi)
        
        return frame