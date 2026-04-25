from ultralytics import YOLO
import cv2

class NumberRecognizer:
    def __init__(self, weights_path):
        """Load the jersey number recognition model"""
        print("Loading jersey number recognition model...")
        # Load your custom weights (trained on 0-9)
        self.model = YOLO(weights_path)

    def recognize(self, frame, bbox):
        """
        يستقبل الفريم وإحداثيات اللاعب، ويقص الصورة لمعرفة الرقم الحقيقي
        """
        x1, y1, x2, y2 = bbox
        
        # قص صورة اللاعب من الفريم الكامل
        player_crop = frame[y1:y2, x1:x2]
        
        # التأكد من أن القص تم بنجاح وأن الصورة ليست فارغة
        if player_crop.size == 0:
            return None
            
        # 1. تمرير صورة اللاعب (المقصوصة) لموديل الأرقام
        # يمكنك تقليل الـ conf إذا كان الموديل لا يكتشف الأرقام بسهولة
        results = self.model(player_crop, conf=0.4, verbose=False)
        
        digits = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # أ. جلب إحداثي X للرقم (عشان نعرف مكانه فين على التيشيرت)
                digit_x1 = int(box.xyxy[0][0])
                
                # ب. جلب اسم الكلاس (الرقم نفسه من 0 لـ 9)
                digit_class = int(box.cls[0]) 
                
                # حفظ مكان الرقم وقيمته
                digits.append((digit_x1, str(digit_class)))
                
        # لو ملقاش أي أرقام على التيشيرت في الفريم ده
        if len(digits) == 0:
            return None
            
        # 2. ترتيب الأرقام من اليسار لليمين بناءً على إحداثي X
        # الخطوة دي مهمة جداً عشان لو الرقم 25، مايتقريش 52
        digits.sort(key=lambda d: d[0])
        
        # 3. دمج الأرقام مع بعض في نص واحد (String)
        # مثلاً: ['1', '0'] هتتحول لـ "10"
        final_number = "".join([d[1] for d in digits])
        
        return final_number