import cv2
import numpy as np

class TeamClassifier:
    def __init__(self):
        # إعدادات ألوان الـ HSV (مضبوطة لمباراة الخماسي)
        # 🔴 الفريق الأحمر
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 🟠 حارس الفريق الأحمر (برتقالي)
        self.lower_orange = np.array([10, 100, 100])
        self.upper_orange = np.array([22, 255, 255])
        
        # 🟢 الفريق الأخضر الفاتح
        self.lower_green = np.array([35, 50, 50])
        self.upper_green = np.array([85, 255, 255])
        
        # 🟡 حارس الفريق الأخضر (أصفر) - تم دمجه هنا
        self.lower_yellow = np.array([20, 100, 100])
        self.upper_yellow = np.array([35, 255, 255])

        self.box_colors = {
            "Red Team": (0, 0, 255),      # BGR الأحمر
            "Green Team": (0, 255, 0),    # BGR الأخضر
            "Referee": (0, 255, 255),     # BGR الأصفر
            "Unknown": (128, 128, 128)
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
        mask_red1 = cv2.inRange(hsv_crop, self.lower_red1, self.upper_red1)
        mask_red2 = cv2.inRange(hsv_crop, self.lower_red2, self.upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_orange = cv2.inRange(hsv_crop, self.lower_orange, self.upper_orange)
        mask_green = cv2.inRange(hsv_crop, self.lower_green, self.upper_green)
        mask_yellow = cv2.inRange(hsv_crop, self.lower_yellow, self.upper_yellow)

        # 5. عد البيكسلات المطابقة لكل لون في منطقة الصدر
        # 🔴 دمج الأحمر مع البرتقالي (الحارس)
        red_pixels = cv2.countNonZero(mask_red) + cv2.countNonZero(mask_orange)
        # 🟢 دمج الأخضر مع الأصفر (الحارس)
        green_pixels = cv2.countNonZero(mask_green) + cv2.countNonZero(mask_yellow)

        # 6. تحديد الفريق صاحب أعلى عدد بيكسلات
        max_pixels = max(red_pixels, green_pixels)

        # لو مفيش أي لون واضح (تجاهل)
        if max_pixels < 5:
            return "Unknown", self.box_colors["Unknown"]

        # إرجاع اسم الفريق ولون المربع
        if max_pixels == red_pixels:
            return "Red Team", self.box_colors["Red Team"]
        else:
            return "Green Team", self.box_colors["Green Team"]