from ultralytics import YOLO
import numpy as np
from src.config import BALL_CONFIDENCE, BALL_CLASS_ID

class BallTracker:
    def __init__(self, weights_path, max_missing_frames=5):
        print("⚽ Loading ball detection model...")
        self.model = YOLO(weights_path)
        
        # Interpolation variables
        self.max_missing_frames = max_missing_frames
        self.missing_count = 0
        self.last_bbox = None
        self.velocity = (0, 0)

    def track(self, frame):
        results = self.model(frame, conf=BALL_CONFIDENCE, verbose=False)
        
        best_box = None
        max_conf  = 0
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                
                # Filter to only the expected ball class (set BALL_CLASS_ID in config.py)
                if class_id != BALL_CLASS_ID:
                    continue
                
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        # Ball detected successfully
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            if self.last_bbox is not None and self.missing_count == 0:
                lx1, ly1, lx2, ly2 = self.last_bbox
                last_center = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)
                self.velocity = (
                    current_center[0] - last_center[0],
                    current_center[1] - last_center[1]
                )
            
            self.last_bbox = best_box
            self.missing_count = 0
            return [int(v) for v in best_box], False  # Real detection

        # Ball not found — interpolate position
        else:
            if self.last_bbox is not None and self.missing_count < self.max_missing_frames:
                self.missing_count += 1
                x1, y1, x2, y2 = self.last_bbox
                vx, vy = self.velocity
                
                new_x1, new_y1 = x1 + vx, y1 + vy
                new_x2, new_y2 = x2 + vx, y2 + vy
                
                self.last_bbox = [new_x1, new_y1, new_x2, new_y2]
                return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)], True  # Interpolated
            
            else:
                self.last_bbox = None
                self.velocity = (0, 0)
                return None, False