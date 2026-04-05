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
        # استخدمنا track بدل predict (أو الاستدعاء المباشر) مع persist=True للحفاظ على الـ IDs
        results = self.model.track(frame, persist=True, tracker="botsort.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)
        
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