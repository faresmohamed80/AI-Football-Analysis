from ultralytics import YOLO
import numpy as np

class BallTracker:
    def __init__(self, weights_path, max_missing_frames=5):
        print("⚽ Loading ball detection model...")
        self.model = YOLO(weights_path)
        
        # متغيرات التوقع (Interpolation)
        self.max_missing_frames = max_missing_frames # أقصى عدد فريمات نتوقعها قبل ما نستسلم
        self.missing_count = 0
        self.last_bbox = None
        self.velocity = (0, 0) # السرعة على محور X و Y

    def track(self, frame):
        # تشغيل الموديل للبحث عن الكرة
        results = self.model(frame, conf=0.2, verbose=False) 
        
        best_box = None
        max_conf = 0
        
        for result in results:
            for box in result.boxes:
                # 🔴 التريكة هنا: فلترة المخرجات عشان ناخد الكورة بس
                
                class_id = int(box.cls[0])
                
                # لو الموديل بتاعك متدرب على الكورة بس (Custom)، شيل الهاشتاج من السطر اللي تحت:
                # if class_id != 0: continue 
                
                # لو الموديل بتاعك عام (Pre-trained COCO)، شيل الهاشتاج من السطر اللي تحت:
                if class_id != 32: continue 
                
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        # 2. في حالة العثور على الكرة بنجاح
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if self.last_bbox is not None and self.missing_count == 0:
                lx1, ly1, lx2, ly2 = self.last_bbox
                last_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
                self.velocity = (current_center[0] - last_center[0], current_center[1] - last_center[1])
            
            self.last_bbox = best_box
            self.missing_count = 0
            return [int(v) for v in best_box], False # الكرة حقيقية

        # 3. في حالة اختفاء الكرة (التوقع Interpolation)
        else:
            if self.last_bbox is not None and self.missing_count < self.max_missing_frames:
                self.missing_count += 1
                x1, y1, x2, y2 = self.last_bbox
                vx, vy = self.velocity
                
                new_x1, new_y1 = x1 + vx, y1 + vy
                new_x2, new_y2 = x2 + vx, y2 + vy
                
                self.last_bbox = [new_x1, new_y1, new_x2, new_y2]
                return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)], True # الكرة متوقعة
            
            else:
                self.last_bbox = None
                self.velocity = (0, 0)
                return None, False