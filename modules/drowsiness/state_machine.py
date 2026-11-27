"""
Temporal State Machine and Event Management for Task 3 Drowsiness Detection.
Differentiates between:
1. Normal short blinks (1 to 6 consecutive frames).
2. Prolonged eye-closure / micro-sleep (>= 15 consecutive frames).
3. Yawning / fatigue warnings (>= 12 consecutive frames with open mouth).
4. Alert states: ALERT (Green), WARNING (Amber), CRITICAL (Red).
Ensures that a single photograph never produces a claim of prolonged drowsiness.
"""

import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any


class DrowsinessState(str, Enum):
    ALERT = "ALERT"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"



class DrowsinessStateMachine:
    """
    Stateful temporal manager for video/camera streams.
    Tracks frame-by-frame transitions and event histories.
    """

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.55,
        blink_min_frames: int = 1,
        blink_max_frames: int = 7,
        drowsy_min_frames: int = 15,
        yawn_min_frames: int = 12,
        alarm_cooldown_frames: int = 25
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.drowsy_min_frames = drowsy_min_frames
        self.yawn_min_frames = yawn_min_frames
        self.alarm_cooldown_frames = alarm_cooldown_frames

        self.reset()

    def reset(self):
        """Reset state counters and history."""
        self.consecutive_closed_frames: int = 0
        self.consecutive_yawn_frames: int = 0
        self.frames_since_last_alarm: int = 999
        self.total_processed_frames: int = 0

        # Statistics
        self.total_blinks: int = 0
        self.drowsy_events_count: int = 0
        self.yawn_events_count: int = 0

        # Current status
        self.current_state: str = "ALERT"    # "ALERT" | "WARNING" | "CRITICAL"
        self.alarm_active: bool = False
        self.status_message: str = "Driver is Alert"

        # Event History Log
        self.event_log: List[Dict[str, Any]] = []

    def update(
        self,
        ear: float,
        mar: float,
        face_detected: bool = True,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a single video/camera frame and update the temporal state.
        
        Returns:
            Dict containing telemetry metrics, state, alarm trigger, and counters.
        """
        self.total_processed_frames += 1
        self.frames_since_last_alarm += 1
        now_ts = timestamp if timestamp is not None else time.time()
        now_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]

        if not face_detected:
            # Face lost or occluded
            self.consecutive_closed_frames = 0
            self.consecutive_yawn_frames = 0
            self.alarm_active = False
            self.current_state = "ALERT"
            self.status_message = "No Face in View"
            return self._build_telemetry(ear, mar, blink_event=False, drowsy_event=False, yawn_event=False)

        blink_event = False
        drowsy_event = False
        yawn_event = False

        # --- 1. EYE CLOSURE ANALYSIS ---
        if ear < self.ear_threshold:
            self.consecutive_closed_frames += 1

            if self.consecutive_closed_frames >= self.drowsy_min_frames:
                # Prolonged eye closure -> CRITICAL DROWSINESS
                self.current_state = "CRITICAL"
                self.status_message = f"CRITICAL: Drowsiness Detected! ({self.consecutive_closed_frames} frames)"

                # Only trigger new alarm episode if cooldown elapsed or entering critical
                if not self.alarm_active and self.frames_since_last_alarm >= self.alarm_cooldown_frames:
                    self.alarm_active = True
                    self.drowsy_events_count += 1
                    drowsy_event = True
                    self.frames_since_last_alarm = 0
                    self.event_log.append({
                        "timestamp": now_str,
                        "event_type": "DROWSINESS_ALERT",
                        "severity": "CRITICAL",
                        "frames_duration": self.consecutive_closed_frames,
                        "ear": round(ear, 3),
                        "mar": round(mar, 3),
                        "details": f"Eyes closed continuously for {self.consecutive_closed_frames} frames"
                    })
            elif self.consecutive_closed_frames > self.blink_max_frames:
                # Intermediate sluggish eyes
                self.current_state = "WARNING"
                self.status_message = f"Warning: Eyes closing ({self.consecutive_closed_frames} frames)"
                self.alarm_active = False
            else:
                # Potential blink in progress
                self.current_state = "ALERT"
                self.status_message = "Blink in progress"
                self.alarm_active = False
        else:
            # Eyes are currently OPEN
            if self.blink_min_frames <= self.consecutive_closed_frames <= self.blink_max_frames:
                # Normal completed blink
                self.total_blinks += 1
                blink_event = True

            # If recovering from a drowsy event
            if self.alarm_active:
                self.alarm_active = False
                self.event_log.append({
                    "timestamp": now_str,
                    "event_type": "RECOVERY",
                    "severity": "INFO",
                    "frames_duration": self.consecutive_closed_frames,
                    "ear": round(ear, 3),
                    "mar": round(mar, 3),
                    "details": f"Driver woke up / reopened eyes after {self.consecutive_closed_frames} frames"
                })

            self.consecutive_closed_frames = 0

        # --- 2. YAWN / MOUTH OPENING ANALYSIS ---
        if mar > self.mar_threshold:
            self.consecutive_yawn_frames += 1
            if self.consecutive_yawn_frames >= self.yawn_min_frames:
                if self.current_state != "CRITICAL":
                    self.current_state = "WARNING"
                    self.status_message = f"Warning: Yawning Detected! ({self.consecutive_yawn_frames} frames)"

                # Register yawn episode once per yawn
                if self.consecutive_yawn_frames == self.yawn_min_frames:
                    self.yawn_events_count += 1
                    yawn_event = True
                    self.event_log.append({
                        "timestamp": now_str,
                        "event_type": "YAWN_WARNING",
                        "severity": "WARNING",
                        "frames_duration": self.consecutive_yawn_frames,
                        "ear": round(ear, 3),
                        "mar": round(mar, 3),
                        "details": f"Yawning detected with MAR {mar:.2f}"
                    })
        else:
            self.consecutive_yawn_frames = 0

        # Final state check if eyes open and not yawning
        if self.consecutive_closed_frames == 0 and self.consecutive_yawn_frames == 0:
            self.current_state = "ALERT"
            self.status_message = "Driver is Alert"
            self.alarm_active = False

        return self._build_telemetry(ear, mar, blink_event, drowsy_event, yawn_event)

    def _build_telemetry(
        self,
        ear: float,
        mar: float,
        blink_event: bool,
        drowsy_event: bool,
        yawn_event: bool
    ) -> Dict[str, Any]:
        """Compile comprehensive telemetry dictionary."""
        return {
            "current_state": self.current_state,
            "status_message": self.status_message,
            "alarm_active": self.alarm_active,
            "consecutive_closed_frames": self.consecutive_closed_frames,
            "consecutive_yawn_frames": self.consecutive_yawn_frames,
            "total_blinks": self.total_blinks,
            "drowsy_events_count": self.drowsy_events_count,
            "yawn_events_count": self.yawn_events_count,
            "ear": ear,
            "mar": mar,
            "blink_detected": blink_event,
            "drowsiness_detected": drowsy_event,
            "yawn_detected": yawn_event,
            "total_frames": self.total_processed_frames
        }

    def evaluate_static_image(self, ear: float, mar: float, face_detected: bool) -> Dict[str, Any]:
        """
        Evaluate a single static photograph without claiming temporal drowsiness.
        A single photo can confirm instantaneous eye state (Open/Closed) and mouth state (Normal/Yawn),
        but explicitly DOES NOT claim prolonged drowsiness.
        """
        if not face_detected:
            return {
                "instantaneous_state": "NO_FACE",
                "eye_state": "NO_FACE",
                "mouth_state": "NO_FACE",
                "ear": ear,
                "mar": mar,
                "badge_class": "badge-neutral",
                "status_text": "No Face Detected",
                "description": "No human face was localized in the provided image."
            }

        eye_closed = ear < self.ear_threshold
        mouth_yawn = mar > self.mar_threshold

        if eye_closed and mouth_yawn:
            state = "WARNING"
            status_text = "Eyes Closed & Mouth Open"
            desc = "Instantaneous frame shows closed eyes and wide mouth. (Note: Video stream required to verify if prolonged)."
            badge_class = "badge-carnivore"
        elif eye_closed:
            state = "WARNING"
            status_text = "Eyes Closed (Photo Snapshot)"
            desc = "Eyes are closed in this frame. Could be a momentary natural blink (100ms) or micro-sleep. Video stream required to determine duration."
            badge_class = "badge-omnivore"
        elif mouth_yawn:
            state = "WARNING"
            status_text = "Yawning / Wide Mouth"
            desc = "Mouth Aspect Ratio exceeds yawning threshold in this frame."
            badge_class = "badge-omnivore"
        else:
            state = "ALERT"
            status_text = "Eyes Open / Alert"
            desc = "Driver's eyes are fully open and mouth is at normal resting position."
            badge_class = "badge-herbivore"

        return {
            "instantaneous_state": state,
            "eye_state": "CLOSED" if eye_closed else "OPEN",
            "mouth_state": "YAWNING" if mouth_yawn else "NORMAL",
            "ear": ear,
            "mar": mar,
            "badge_class": badge_class,
            "status_text": status_text,
            "description": desc
        }


class OccupantTrack:
    """
    Stateful tracker for an individual vehicle occupant across video frames.
    Maintains independent biometric state machine, blink counts, and age estimate.
    """

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        age_info: Optional[Dict[str, Any]] = None,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.55,
        drowsy_min_frames: int = 15
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.age_info = age_info or {"estimated_age": 25, "label": "~25 ± 5 yrs"}
        self.state_machine = DrowsinessStateMachine(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            drowsy_min_frames=drowsy_min_frames
        )
        self.frames_unseen = 0
        self.last_ear = 0.35
        self.last_mar = 0.10
        self.last_status = "ALERT"

    @property
    def centroid(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    def update_measurement(
        self,
        bbox: Tuple[int, int, int, int],
        ear: float,
        mar: float,
        age_info: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update track with new frame measurements."""
        self.bbox = bbox
        self.frames_unseen = 0
        self.last_ear = ear
        self.last_mar = mar
        if age_info is not None:
            self.age_info = age_info

        tel = self.state_machine.update(ear=ear, mar=mar, face_detected=True, timestamp=timestamp)
        self.last_status = tel["current_state"]
        return tel


class MultiPersonDrowsinessTracker:
    """
    Multi-occupant temporal video tracker for vehicle environments.
    Tracks multiple passengers simultaneously and manages independent state machines.
    Fulfills:
    - Requirement 2: Detect multiple people in one image or video.
    - Requirement 3: Count total number of people detected.
    - Requirement 4: Count number of sleeping people.
    - Requirement 5: Highlight sleeping people with RED bounding boxes.
    - Requirement 6: Predict the age of each detected person.
    - Requirement 7: Display pop-up showing sleeping people and their age.
    """

    def __init__(
        self,
        ear_threshold: float = 0.22,
        mar_threshold: float = 0.55,
        drowsy_min_frames: int = 15,
        max_missed_frames: int = 15
    ):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.drowsy_min_frames = drowsy_min_frames
        self.max_missed_frames = max_missed_frames

        self.tracks: Dict[int, OccupantTrack] = {}
        self.next_track_id: int = 1
        self.unified_event_log: List[Dict[str, Any]] = []
        self.total_processed_frames: int = 0

    def reset(self):
        """Reset all multi-occupant tracks and history."""
        self.tracks.clear()
        self.next_track_id = 1
        self.unified_event_log.clear()
        self.total_processed_frames = 0

    def update(
        self,
        detected_people: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process a list of detected people in the current video frame.
        Associates detections with persistent occupant tracks via spatial proximity.
        """
        self.total_processed_frames += 1
        now_ts = timestamp if timestamp is not None else time.time()
        now_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]

        unmatched_detections = list(range(len(detected_people)))
        active_track_ids = list(self.tracks.keys())
        matched_tracks = set()

        # Proximity matching between detections and existing tracks
        for d_idx in list(unmatched_detections):
            det = detected_people[d_idx]
            d_bbox = det.get("bbox", (0, 0, 0, 0))
            d_cx = d_bbox[0] + d_bbox[2] / 2.0
            d_cy = d_bbox[1] + d_bbox[3] / 2.0

            best_tid = None
            best_dist = 999999.0

            for tid in active_track_ids:
                if tid in matched_tracks:
                    continue
                track = self.tracks[tid]
                t_cx, t_cy = track.centroid
                dist = ((d_cx - t_cx) ** 2 + (d_cy - t_cy) ** 2) ** 0.5
                max_dim = max(d_bbox[2], d_bbox[3], 50)
                if dist < max_dim * 1.5 and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                matched_tracks.add(best_tid)
                unmatched_detections.remove(d_idx)
                track = self.tracks[best_tid]
                ear = det.get("ear", 0.35)
                mar = det.get("mar", 0.10)
                age_info = det.get("age_info")
                tel = track.update_measurement(d_bbox, ear, mar, age_info, timestamp=now_ts)

                # Check if new alert was fired by this track
                if tel.get("drowsiness_detected", False):
                    self.unified_event_log.append({
                        "timestamp": now_str,
                        "occupant_id": track.track_id,
                        "age": track.age_info.get("estimated_age", 25),
                        "age_label": track.age_info.get("label", "~25 ± 5 yrs"),
                        "event_type": "DROWSINESS_ALERT",
                        "severity": "CRITICAL",
                        "frames_duration": track.state_machine.consecutive_closed_frames,
                        "ear": round(ear, 3),
                        "mar": round(mar, 3),
                        "details": f"Occupant #{track.track_id} ({track.age_info.get('label')}) closed eyes for {track.state_machine.consecutive_closed_frames} frames"
                    })

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            det = detected_people[d_idx]
            new_tid = self.next_track_id
            self.next_track_id += 1
            d_bbox = det.get("bbox", (0, 0, 0, 0))
            new_track = OccupantTrack(
                track_id=new_tid,
                bbox=d_bbox,
                age_info=det.get("age_info"),
                ear_threshold=self.ear_threshold,
                mar_threshold=self.mar_threshold,
                drowsy_min_frames=self.drowsy_min_frames
            )
            ear = det.get("ear", 0.35)
            mar = det.get("mar", 0.10)
            tel = new_track.update_measurement(d_bbox, ear, mar, det.get("age_info"), timestamp=now_ts)
            self.tracks[new_tid] = new_track
            matched_tracks.add(new_tid)

        # Increment unseen count for unmatched tracks and prune stale tracks
        stale_tids = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.frames_unseen += 1
                if track.frames_unseen > self.max_missed_frames:
                    stale_tids.append(tid)

        for tid in stale_tids:
            del self.tracks[tid]

        # Aggregate frame summary
        total_occupants = len(matched_tracks)
        sleeping_tracks = [
            t for t in self.tracks.values()
            if t.track_id in matched_tracks and t.state_machine.current_state == "CRITICAL"
        ]
        sleeping_count = len(sleeping_tracks)
        awake_count = total_occupants - sleeping_count
        alarm_active = any(t.state_machine.alarm_active for t in self.tracks.values())

        if sleeping_count > 0:
            ages_str = ", ".join([f"Occupant #{t.track_id} ({t.age_info['label']})" for t in sleeping_tracks])
            popup_summary = f"Alert: {sleeping_count} occupant{'s' if sleeping_count > 1 else ''} sleeping! {ages_str}"
        else:
            popup_summary = f"All {total_occupants} occupant{'s are' if total_occupants > 1 else ' is'} awake."

        return {
            "total_people": total_occupants,
            "sleeping_count": sleeping_count,
            "awake_count": awake_count,
            "sleeping_occupants": [
                {
                    "track_id": t.track_id,
                    "age": t.age_info.get("estimated_age", 25),
                    "age_label": t.age_info.get("label", "~25 ± 5 yrs"),
                    "ear": t.last_ear,
                    "mar": t.last_mar,
                    "closed_frames": t.state_machine.consecutive_closed_frames
                }
                for t in sleeping_tracks
            ],
            "alarm_active": alarm_active,
            "popup_summary": popup_summary,
            "active_tracks": list(self.tracks.values()),
            "total_frames": self.total_processed_frames
        }
