from ultralytics import YOLO
from src.config import CONFIDENCE_THRESHOLD

class PlayerDetector:
    def __init__(self, weights_path):
        print("جاري تحميل موديل اللاعبين (مع نظام التتبع)...")
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """
        يستقبل فريم ويرجع قائمة تحتوي على (track_id, bbox) لكل لاعب
        """
        # 🔴 التعديل هنا: ضفنا classes=[0] عشان نفلتر الأشخاص (Person) فقط
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            conf=CONFIDENCE_THRESHOLD, 
            classes=[0], # 0 هو الـ ID الخاص بالأشخاص في موديلات YOLO
            verbose=False
        )
        
        tracked_players = []
        for result in results:
            boxes = result.boxes
            # التأكد من وجود أرقام تتبع (IDs)
            if boxes.id is not None: 
                track_ids = boxes.id.int().cpu().tolist()
                bboxes = boxes.xyxy.int().cpu().tolist()
                
                for track_id, bbox in zip(track_ids, bboxes):
                    tracked_players.append((track_id, bbox))
                    
        return tracked_players