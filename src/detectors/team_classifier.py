import cv2
import numpy as np

class TeamClassifier:
    def __init__(self):
        # إعدادات ألوان الـ HSV (مضبوطة على إضاءة الملاعب)
        # 🔵 الفريق الأزرق (تشيلسي)
        self.lower_blue = np.array([90, 50, 50])
        self.upper_blue = np.array([130, 255, 255])
        
        # ⚪ الفريق الأبيض (سوانزي) - تشبع لوني قليل وإضاءة عالية
        self.lower_white = np.array([0, 0, 180])
        self.upper_white = np.array([180, 50, 255])
        
        # ⚫ الحكم (أسود) - إضاءة قليلة جداً
        self.lower_black = np.array([0, 0, 0])
        self.upper_black = np.array([180, 255, 50])

        self.box_colors = {
            "Blue Team": (255, 0, 0),     # مربع أزرق (OpenCV BGR)
            "White Team": (255, 255, 255),# مربع أبيض
            "Referee": (0, 0, 0),         # مربع أسود
            "Unknown": (0, 255, 0)        # مربع أخضر
        }

    def get_player_team(self, frame, bbox):
        # التأكد من الإحداثيات إنها أرقام صحيحة (integers)
        x1, y1, x2, y2 = map(int, bbox)

        # حماية من خروج المربع بره حدود الشاشة
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_frame, x2), min(h_frame, y2)

        # 1. قص صورة اللاعب
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return "Unknown", self.box_colors["Unknown"]

        # 2. قص ذكي لمنطقة الصدر فقط (بناءً على فكرتك الممتازة)
        h, w = player_crop.shape[:2]
        shirt_crop = player_crop[int(h*0.1):int(h*0.5), int(w*0.2):int(w*0.8)]
        
        if shirt_crop.size == 0:
            shirt_crop = player_crop # في حالة المربعات الصغيرة جداً

        # 3. تحويل منطقة الصدر لنظام HSV (عشان نتجنب مشاكل الضل والنور)
        hsv_crop = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2HSV)

        # 4. عمل ماسكات (Masks) لكل لون
        mask_blue = cv2.inRange(hsv_crop, self.lower_blue, self.upper_blue)
        mask_white = cv2.inRange(hsv_crop, self.lower_white, self.upper_white)
        mask_black = cv2.inRange(hsv_crop, self.lower_black, self.upper_black)

        # 5. عد البيكسلات المطابقة لكل لون في منطقة الصدر
        blue_pixels = cv2.countNonZero(mask_blue)
        white_pixels = cv2.countNonZero(mask_white)
        black_pixels = cv2.countNonZero(mask_black)

        # 6. تحديد الفريق صاحب أعلى عدد بيكسلات
        max_pixels = max(blue_pixels, white_pixels, black_pixels)

        # لو مفيش أي لون واضح (تجاهل)
        if max_pixels < 5:
            return "Unknown", self.box_colors["Unknown"]

        # إرجاع اسم الفريق ولون المربع
        if max_pixels == blue_pixels:
            return "Blue Team", self.box_colors["Blue Team"]
        elif max_pixels == white_pixels:
            return "White Team", self.box_colors["White Team"]
        else:
            return "Referee", self.box_colors["Referee"]