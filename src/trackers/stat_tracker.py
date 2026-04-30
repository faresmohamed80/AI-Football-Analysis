import cv2
import numpy as np
from src.config import (
    STICKY_FRAMES,
    REQUIRED_POSSESSION_FRAMES,
    FEET_ZONE_HEIGHT_RATIO,
    FEET_ZONE_WIDTH_EXPANSION,
    TEAM_1_NAME,
    TEAM_2_NAME,
    TEAM_1_DISPLAY_COLOR,
    TEAM_2_DISPLAY_COLOR,
)


class MatchStats:
    """
    Possession & event tracking based on bbox feet-zone intersection.

    Logic:
    ─────
    • Ball is "at" a player if its centre falls inside the player's FEET ZONE
      (bottom FEET_ZONE_HEIGHT_RATIO of their bounding box, slightly widened).
    • A PASS is recorded when the ball moves from player A's feet zone →
      player B's feet zone and both are on the SAME team.
    • An INTERCEPTION is recorded when the ball moves from A → B and they are
      on DIFFERENT teams.
    • If the ball leaves A's zone and returns to A's zone (dribble), no event.
    • Possession is "sticky": the last possessor keeps credit for STICKY_FRAMES
      frames after the ball leaves their feet.
    """

    def __init__(self, team_1_name=TEAM_1_NAME, team_2_name=TEAM_2_NAME,
                 team_1_color=TEAM_1_DISPLAY_COLOR, team_2_color=TEAM_2_DISPLAY_COLOR):
        self.required_frames = REQUIRED_POSSESSION_FRAMES
        self.sticky_frames   = STICKY_FRAMES

        # ── Dynamic team names ─────────────────────────────────────────
        self.team_1_name  = team_1_name
        self.team_2_name  = team_2_name
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

        # ── Current confirmed possessor ────────────────────────────────
        self.possessor_tid   = None   # track_id
        self.possessor_team  = None
        self.possessor_label = "Free Ball"   # display string
        self.frames_without_contact = 0

        # ── Candidate (building up consecutive frames) ─────────────────
        self.candidate_tid   = None
        self.candidate_team  = None
        self.candidate_name  = None
        self.candidate_count = 0

        # ── Statistics ─────────────────────────────────────────────────
        self.team_possession_frames = {team_1_name: 0, team_2_name: 0}
        self.total_possession_frames = 0
        self.current_possessor = "Free Ball"   # shown in HUD

        self.event_counts = {
            "passes_team1":        0,
            "passes_team2":        0,
            "interceptions_team1": 0,
            "interceptions_team2": 0,
        }

        # ── UI alerts ──────────────────────────────────────────────────
        self.current_alert = None
        self.alert_frames  = 0
        self.alert_color   = (255, 255, 255)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _feet_zone(bbox):
        """Return (x1, fy1, x2, y2) — the feet area of a player bbox."""
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        fy1 = y2 - int(h * FEET_ZONE_HEIGHT_RATIO)
        # Expand width slightly so ball touching the side is also captured
        w_margin = int((x2 - x1) * FEET_ZONE_WIDTH_EXPANSION)
        return (x1 - w_margin, fy1, x2 + w_margin, y2)

    @staticmethod
    def _ball_in_feet_zone(ball_cx, ball_cy, bbox):
        """True if the ball centre falls inside the player's feet zone."""
        fx1, fy1, fx2, fy2 = MatchStats._feet_zone(bbox)
        return fx1 <= ball_cx <= fx2 and fy1 <= ball_cy <= fy2

    def _fire_event(self, from_team, to_team, to_team_name):
        """Record pass or interception and set HUD alert."""
        if from_team == to_team:
            # Same team → PASS
            key = "passes_team1" if to_team == self.team_1_name else "passes_team2"
            self.event_counts[key] += 1
            self.current_alert = "NICE PASS!"
            self.alert_color   = (0, 220, 0)
            self.alert_frames  = 45
        else:
            # Different team → INTERCEPTION
            key = "interceptions_team1" if to_team == self.team_1_name else "interceptions_team2"
            self.event_counts[key] += 1
            self.current_alert = "INTERCEPTION!"
            self.alert_color   = (0, 80, 255)
            self.alert_frames  = 50

    def _carry_possession(self):
        """Continue crediting the last possessor for sticky_frames."""
        if self.possessor_team is not None:
            self.frames_without_contact += 1
            if self.frames_without_contact <= self.sticky_frames:
                self.team_possession_frames[self.possessor_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.possessor_label
            else:
                self.current_possessor = "Free Ball"
        else:
            self.current_possessor = "Free Ball"

    # ──────────────────────────────────────────────────────────────────
    # Main update — called every frame
    # ──────────────────────────────────────────────────────────────────

    def update(self, players_data, ball_data):
        # Countdown alert display
        if self.alert_frames > 0:
            self.alert_frames -= 1
        else:
            self.current_alert = None

        # ── Ball must be a real detection (not interpolated) ───────────
        if not ball_data or ball_data[0] is None or ball_data[1] is True:
            self._carry_possession()
            return

        bbox_ball = ball_data[0]
        ball_cx = (bbox_ball[0] + bbox_ball[2]) // 2
        ball_cy = (bbox_ball[1] + bbox_ball[3]) // 2

        # ── Find which player's feet zone contains the ball ───────────
        contact_tid   = None
        contact_team  = None
        contact_name  = None

        for p in players_data:
            if p['team'] in ('Referee', 'Unknown'):
                continue
            if self._ball_in_feet_zone(ball_cx, ball_cy, p['bbox']):
                contact_tid  = p['track_id']
                contact_team = p['team']
                contact_name = p.get('name') or f"Player #{contact_tid}"
                break  # Take the first match (closest bboxes rarely overlap)

        # ── Process contact ────────────────────────────────────────────
        if contact_tid is not None:
            # Accumulate candidate frames for this track_id
            if contact_tid == self.candidate_tid:
                self.candidate_count += 1
            else:
                # New candidate — reset counter
                self.candidate_tid   = contact_tid
                self.candidate_team  = contact_team
                self.candidate_name  = contact_name
                self.candidate_count = 1

            # Enough frames to confirm possession?
            if self.candidate_count >= self.required_frames:
                prev_tid  = self.possessor_tid
                prev_team = self.possessor_team

                # ── Detect pass / interception ────────────────────────
                if prev_tid is not None and prev_tid != contact_tid:
                    # Ball physically moved from one player to another
                    self._fire_event(prev_team, contact_team, contact_name)

                # Confirm new possessor
                self.possessor_tid   = contact_tid
                self.possessor_team  = contact_team
                self.possessor_label = f"{contact_name} ({contact_team})"
                self.frames_without_contact = 0

                # Make sure team key exists (handles Unknown/Referee edge cases)
                if contact_team in self.team_possession_frames:
                    self.team_possession_frames[contact_team] += 1
                self.total_possession_frames += 1
                self.current_possessor = self.possessor_label

        else:
            # No player's feet zone contains the ball
            self.candidate_count = 0   # Reset candidate streak
            self._carry_possession()

    # ──────────────────────────────────────────────────────────────────
    # Stats accessors
    # ──────────────────────────────────────────────────────────────────

    def get_possession_stats(self):
        if self.total_possession_frames == 0:
            return {self.team_1_name: 50, self.team_2_name: 50}
        t1 = self.team_possession_frames.get(self.team_1_name, 0)
        t2 = self.team_possession_frames.get(self.team_2_name, 0)
        total = self.total_possession_frames
        return {self.team_1_name: int(t1 / total * 100),
                self.team_2_name: int(t2 / total * 100)}

    def get_event_stats(self):
        return {
            "passes_t1": self.event_counts["passes_team1"],
            "passes_t2": self.event_counts["passes_team2"],
            "inter_t1":  self.event_counts["interceptions_team1"],
            "inter_t2":  self.event_counts["interceptions_team2"],
        }

    # ──────────────────────────────────────────────────────────────────
    # HUD drawing
    # ──────────────────────────────────────────────────────────────────

    def draw_stats(self, frame):
        stats  = self.get_possession_stats()
        t1_pct = stats[self.team_1_name]
        t2_pct = stats[self.team_2_name]

        x, y, w, h = 20, 20, 340, 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x, y), (x + 6, y + h), (0, 215, 255), -1)

        cv2.putText(frame, "L I V E  M A T C H  S T A T S",
                    (x + 20, y + 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Possession: {self.team_1_name} {t1_pct}% | {self.team_2_name} {t2_pct}%",
                    (x + 20, y + 55), cv2.FONT_HERSHEY_DUPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        bar_y = y + 68
        cv2.rectangle(frame, (x + 20, bar_y), (x + w - 20, bar_y + 6), (220, 220, 220), -1)
        t1_w = int((w - 40) * t1_pct / 100)
        if t1_w > 0:
            cv2.rectangle(frame, (x + 20, bar_y), (x + 20 + t1_w, bar_y + 6), self.team_1_color, -1)

        cv2.putText(frame, f"Ball: {self.current_possessor}",
                    (x + 20, y + 95), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)

        # Pop-up alert
        if self.alert_frames > 0 and self.current_alert:
            alpha = min(1.0, self.alert_frames / 15.0)
            alert_overlay = frame.copy()
            fh, fw = frame.shape[:2]
            text_size = cv2.getTextSize(self.current_alert, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
            bx1 = int((fw - text_size[0]) / 2) - 20
            by1 = fh - 110 - text_size[1] - 10
            bx2 = bx1 + text_size[0] + 40
            by2 = fh - 110 + 10
            cv2.rectangle(alert_overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
            cv2.addWeighted(alert_overlay, alpha * 0.65, frame, 1 - alpha * 0.65, 0, frame)
            cv2.putText(frame, self.current_alert,
                        (bx1 + 20, fh - 110), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                        self.alert_color, 2, cv2.LINE_AA)

        return frame