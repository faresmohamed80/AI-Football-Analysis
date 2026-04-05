import cv2
import numpy as np

class TeamClassifier:
    def __init__(self):
        self.team_colors = {
            "Blue Team": np.array([0, 0, 255]),      # أزرق
            "White Team": np.array([255, 255, 255]), # أبيض
            "Referee": np.array([255, 255, 0])       # أصفر
        }
        
        self.box_colors = {
            "Blue Team": (255, 0, 0),     # مربع أزرق (OpenCV BGR)
            "White Team": (255, 255, 255),# مربع أبيض
            "Referee": (0, 255, 255),     # مربع أصفر
            "Unknown": (0, 255, 0)        # مربع أخضر
        }

    def get_player_team(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        
        # 1. قص صورة اللاعب
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return "Unknown", self.box_colors["Unknown"]

        # 2. قص ذكي لمنطقة الصدر فقط (أخذ من 10% لـ 50% من الطول، ومن 20% لـ 80% من العرض)
        h, w = player_crop.shape[:2]
        shirt_crop = player_crop[int(h*0.1):int(h*0.5), int(w*0.2):int(w*0.8)]
        
        if shirt_crop.size == 0:
            shirt_crop = player_crop # في حالة المربعات الصغيرة جداً

        # 3. عزل لون النجيلة (الخلفية الخضراء)
        hsv_crop = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2HSV)
        
        # درجات اللون الأخضر في نظام HSV
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # عمل ماسك للنجيلة وعكسه (عشان ناخد اللي مش أخضر بس)
        green_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)

        # 4. تحويل الصورة لـ RGB واستخراج بيكسلات التيشيرت فقط
        shirt_rgb = cv2.cvtColor(shirt_crop, cv2.COLOR_BGR2RGB)
        pixels = shirt_rgb[non_green_mask == 255] # البيكسلات غير الخضراء
        
        # لو الصورة كلها طلعت خضراء (خطأ في القص)، نستخدم كل الصورة كبديل
        if len(pixels) == 0:
            pixels = shirt_rgb.reshape((-1, 3))
            
        pixels = np.float32(pixels)

        # 5. استخدام K-Means (محتاجين لون واحد بس لأننا شيلنا الخلفية خلاص)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 1 
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        dominant_color = centers[0]

        # 6. مقارنة اللون المتبقي بألوان الفرق
        min_dist = float('inf')
        assigned_team = "Unknown"
        
        for team_name, team_color in self.team_colors.items():
            dist = np.linalg.norm(dominant_color - team_color)
            if dist < min_dist:
                min_dist = dist
                assigned_team = team_name

        return assigned_team, self.box_colors.get(assigned_team, self.box_colors["Unknown"])