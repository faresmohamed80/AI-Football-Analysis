import numpy as np
import cv2
from collections import defaultdict

class SpeedDistanceTracker:
    def __init__(self, fps, homography_matrix=None, pixel_to_meter=0.02):
        self.fps = fps
        self.homography_matrix = homography_matrix
        self.pixel_to_meter_ratio = pixel_to_meter

        self.prev_positions = {}
        self.total_distance = defaultdict(float)
        self.speeds = {}
        self.max_speeds = defaultdict(float)  # تتبع أقصى سرعة لكل لاعب

        # 🔥 history للـ smoothing
        self.speed_history = defaultdict(list)

    def convert_position(self, point):
        if self.homography_matrix is not None:
            px = np.array([[point]], dtype='float32')
            transformed = cv2.perspectiveTransform(px, self.homography_matrix)
            return transformed[0][0]

        x, y = point
        return (x * self.pixel_to_meter_ratio,
                y * self.pixel_to_meter_ratio)

    def update(self, tracks):
        """
        tracks: dict -> {track_id: (x, y)}
        """

        for track_id, current_pos in tracks.items():

            current_pos = self.convert_position(current_pos)

            if track_id in self.prev_positions:
                prev_pos = self.prev_positions[track_id]

                # 🔥 المسافة
                distance = np.linalg.norm(
                    np.array(current_pos) - np.array(prev_pos)
                )

                # ❌ ignore noise صغير جدًا
                if distance < 0.01:
                    distance = 0

                # إجمالي المسافة
                self.total_distance[track_id] += distance

                # 🔥 السرعة اللحظية
                speed = distance * self.fps  # m/s

                # ❌ limit غير منطقي
                if speed > 12:   # 12 m/s ≈ 43 km/h (max sprint)
                    speed = 12

                # 🔥 smoothing باستخدام history
                self.speed_history[track_id].append(speed)

                if len(self.speed_history[track_id]) > 5:
                    self.speed_history[track_id].pop(0)

                smooth_speed = np.mean(self.speed_history[track_id])

                self.speeds[track_id] = smooth_speed
                
                # تحديث أقصى سرعة مسجلة لهذا اللاعب
                if smooth_speed > self.max_speeds[track_id]:
                    self.max_speeds[track_id] = smooth_speed

            self.prev_positions[track_id] = current_pos

        return self.total_distance, self.speeds