import cv2
import numpy as np
import supervision as sv
from collections import deque

from src.config import (
    BALL_TRAIL_ENABLED,
    BALL_TRAIL_LENGTH,
    BALL_TRAIL_COLOR,
    BALL_TRAIL_THICKNESS,
)


class Visualizer:

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _players_to_detections(players_data):
        """Convert our player dicts into a sv.Detections object."""
        if not players_data:
            return sv.Detections.empty()

        bboxes, labels, colors = [], [], []
        for p in players_data:
            x1, y1, x2, y2 = map(int, p['bbox'])
            bboxes.append([x1, y1, x2, y2])
            labels.append(p.get('name') or "")
            colors.append(p.get('color', (128, 128, 128)))

        xyxy = np.array(bboxes, dtype=np.float32)
        det  = sv.Detections(xyxy=xyxy)
        det.data['labels'] = labels
        det.data['colors'] = colors
        return det

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def draw_annotations(frame, players_data, ball_data=None,
                         possessor_name: str | None = None,
                         ball_trail: deque | None = None):
        """
        Draw:
        - supervision EllipseAnnotator for every detected player
        - dark-background label above each player (always readable)
        - gold triangle above the ball possessor
        - ball trail (fading colored line)
        - ball marker
        """

        # ── 1. Build sv.Detections ─────────────────────────────────────
        det = Visualizer._players_to_detections(players_data)

        if len(det) > 0:
            labels  = det.data['labels']
            raw_bgr = det.data['colors']

            # Ellipse: per-player team colour (RGB for supervision)
            sv_colors = [
                sv.Color(r=int(c[2]), g=int(c[1]), b=int(c[0]))
                for c in raw_bgr
            ]
            palette = sv.ColorPalette(colors=sv_colors)

            ellipse_ann = sv.EllipseAnnotator(
                color=palette,
                thickness=2,
                start_angle=-45,
                end_angle=235,
                color_lookup=sv.ColorLookup.INDEX,
            )
            frame = ellipse_ann.annotate(scene=frame, detections=det)

            # ── 2. Labels: always dark background + white text ──────────
            # Build a dark-grey palette so labels are always readable,
            # regardless of team colour (fixes white-on-white issue).
            dark_palette = sv.ColorPalette(
                colors=[sv.Color(r=30, g=30, b=30)] * len(det)
            )
            label_ann = sv.LabelAnnotator(
                color=dark_palette,
                text_scale=0.4,
                text_thickness=1,
                text_padding=5,
                border_radius=4,
                text_position=sv.Position.TOP_CENTER,
                text_color=sv.Color.WHITE,
                color_lookup=sv.ColorLookup.INDEX,
            )
            frame = label_ann.annotate(
                scene=frame,
                detections=det,
                labels=list(labels),
            )

            # ── 3. Triangle above ball possessor ───────────────────────
            if possessor_name:
                poss_indices = [
                    i for i, lbl in enumerate(labels)
                    if lbl and lbl in possessor_name
                ]
                if poss_indices:
                    poss_det = det[poss_indices]
                    tri_ann = sv.TriangleAnnotator(
                        color=sv.Color(r=255, g=215, b=0),  # Gold
                        base=18,
                        height=16,
                        position=sv.Position.TOP_CENTER,
                        outline_thickness=1,
                        outline_color=sv.Color.BLACK,
                        color_lookup=sv.ColorLookup.INDEX,
                    )
                    frame = tri_ann.annotate(scene=frame, detections=poss_det)

        # ── 4. Ball trail ───────────────────────────────────────────────
        if BALL_TRAIL_ENABLED and ball_trail and len(ball_trail) > 1:
            pts = list(ball_trail)
            n   = len(pts)
            for i in range(1, n):
                # Fade: older points are thinner and more transparent
                alpha     = i / n                          # 0..1, newest = 1
                thickness = max(1, int(BALL_TRAIL_THICKNESS * alpha))
                # Blend trail colour toward dark with age
                c = tuple(int(v * alpha) for v in BALL_TRAIL_COLOR)
                cv2.line(frame, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)

        # ── 5. Ball marker ──────────────────────────────────────────────
        if ball_data is not None:
            bbox, is_interpolated = ball_data
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = max((x2 - x1), (y2 - y1)) // 2 + 3

                color = (0, 0, 200) if is_interpolated else (0, 215, 255)
                cv2.circle(frame, (cx, cy), radius + 2, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), radius, color,
                           -1 if is_interpolated else 2, cv2.LINE_AA)

                # Small indicator arrow above ball
                pt1 = (cx,     cy - radius - 6)
                pt2 = (cx - 5, cy - radius - 14)
                pt3 = (cx + 5, cy - radius - 14)
                tri = np.array([pt1, pt2, pt3])
                cv2.drawContours(frame, [tri], 0, color, -1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Speed / distance overlay
    # ------------------------------------------------------------------

    @staticmethod
    def draw_speed_distance(frame, players_data, speeds, total_dist):
        """Draw speed and total-distance badge below each player."""
        for p in players_data:
            tid = p.get('track_id')
            if tid is None:
                continue

            x1, y1, x2, y2 = map(int, p['bbox'])
            color = p.get('color', (200, 200, 200))

            speed_ms  = speeds.get(tid, 0.0)
            speed_kmh = speed_ms * 3.6
            dist_m    = total_dist.get(tid, 0.0)

            line1 = f"{speed_kmh:.1f} km/h"
            line2 = f"{dist_m:.0f} m"

            font      = cv2.FONT_HERSHEY_DUPLEX
            scale     = 0.4
            thickness = 1

            (w1, h1), _ = cv2.getTextSize(line1, font, scale, thickness)
            (w2, h2), _ = cv2.getTextSize(line2, font, scale, thickness)

            box_w    = max(w1, w2) + 14
            box_h    = h1 + h2 + 18
            center_x = (x1 + x2) // 2
            bx1 = center_x - box_w // 2
            bx2 = center_x + box_w // 2
            by1 = y2 + 6
            by2 = by1 + box_h

            sub = frame[by1:by2, bx1:bx2]
            if sub.shape[0] > 0 and sub.shape[1] > 0:
                black = np.full_like(sub, (20, 20, 20))
                cv2.addWeighted(black, 0.75, sub, 0.25, 0, sub)
                frame[by1:by2, bx1:bx2] = sub

            cv2.line(frame, (bx1, by1), (bx2, by1), color, 2)
            cv2.putText(frame, line1, (bx1 + 7, by1 + h1 + 5),
                        font, scale, (0, 230, 255), thickness, cv2.LINE_AA)
            cv2.putText(frame, line2, (bx1 + 7, by1 + h1 + h2 + 12),
                        font, scale, (200, 200, 200), thickness, cv2.LINE_AA)

        return frame