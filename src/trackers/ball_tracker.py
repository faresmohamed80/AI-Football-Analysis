from ultralytics import YOLO
import numpy as np

class BallTracker:
    def __init__(self, weights_path, max_missing_frames=5):
        print("⚽ جاري تحميل موديل الكرة...")
        self.model = YOLO(weights_path)
        
        # متغيرات التوقع (Interpolation)
        self.max_missing_frames = max_missing_frames # أقصى عدد فريمات نتوقعها قبل ما نستسلم
        self.missing_count = 0
        self.last_bbox = None
        self.velocity = (0, 0) # السرعة على محور X و Y

    def track(self, frame):
        """
        بيحاول يكتشف الكرة، لو ملقاهاش بيتوقع مكانها.
        بيرجع: (إحداثيات المربع, هل هذا المربع متوقع أم حقيقي؟)
        """
        # 1. تشغيل الموديل للبحث عن الكرة
        # افترضنا أنك تستخدم موديل YOLO المتدرب على الكرات. 
        # لو تستخدم YOLO العادي، الـ class الخاص بالكرة هو 32.
        results = self.model(frame, conf=0.2, verbose=False) # الثقة قليلة عشان نلقط الكرة وهي مموهة
        
        best_box = None
        max_conf = 0
        
        # البحث عن المربع صاحب أعلى نسبة ثقة (لأن الكرة واحدة في الملعب)
        for result in results:
            for box in result.boxes:
                # إذا كنت تستخدم موديل YOLO العادي، قم بإزالة علامة # من السطر التالي:
                # if int(box.cls[0]) != 32: continue 
                
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        # 2. في حالة العثور على الكرة بنجاح
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            # حساب السرعة لو كان عندنا مكان سابق للكرة
            if self.last_bbox is not None and self.missing_count == 0:
                lx1, ly1, lx2, ly2 = self.last_bbox
                last_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
                
                # السرعة = المكان الحالي - المكان السابق
                self.velocity = (current_center[0] - last_center[0], current_center[1] - last_center[1])
            
            # تحديث المتغيرات
            self.last_bbox = best_box
            self.missing_count = 0
            return [int(v) for v in best_box], False # False = الكرة حقيقية مش متوقعة

        # 3. في حالة اختفاء الكرة (لم يكتشفها الموديل)
        else:
            if self.last_bbox is not None and self.missing_count < self.max_missing_frames:
                # توقع المكان الجديد بناءً على السرعة السابقة
                self.missing_count += 1
                x1, y1, x2, y2 = self.last_bbox
                vx, vy = self.velocity
                
                new_x1, new_y1 = x1 + vx, y1 + vy
                new_x2, new_y2 = x2 + vx, y2 + vy
                
                self.last_bbox = [new_x1, new_y1, new_x2, new_y2]
                return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)], True # True = الكرة متوقعة
            
            else:
                # الكرة اختفت لفترة طويلة جداً (طلعت بره الملعب مثلاً)
                self.last_bbox = None
                self.velocity = (0, 0)
                return None, False