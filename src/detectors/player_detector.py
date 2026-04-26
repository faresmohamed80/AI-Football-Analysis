from ultralytics import YOLO
from src.config import CONFIDENCE_THRESHOLD

# Futsal detection model class mapping:
# 0 = player, 1 = referee (model distinguishes between them)
PLAYER_CLASS_ID  = 0
REFEREE_CLASS_ID = 1

class PlayerDetector:
    def __init__(self, weights_path):
        print("Loading futsal player detection model (ByteTrack)...")
        self.model = YOLO(weights_path)

    def detect(self, frame):
        """
        Returns a list of (track_id, bbox, is_referee) for every detected person.
        Uses ByteTrack for stable IDs across frames.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            classes=[PLAYER_CLASS_ID, REFEREE_CLASS_ID],   # both players & referees
            verbose=False
        )

        tracked_persons = []
        for result in results:
            boxes = result.boxes
            if boxes.id is None:
                continue
            track_ids = boxes.id.int().cpu().tolist()
            bboxes    = boxes.xyxy.int().cpu().tolist()
            classes   = boxes.cls.int().cpu().tolist()

            for track_id, bbox, cls_id in zip(track_ids, bboxes, classes):
                is_referee = (cls_id == REFEREE_CLASS_ID)
                tracked_persons.append((track_id, bbox, is_referee))

        return tracked_persons