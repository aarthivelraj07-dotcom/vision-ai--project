"""
Multi-Object Animal Tracker for Video Sequences.
Prevents repeated counting of the same animal across video frames.
Features:
- Spatial IoU (Intersection-over-Union) + Centroid distance association.
- Species-constrained matching (tracks only associate with the same animal class).
- Coasting state with grace period (max_lost_frames=15) to survive temporary occlusions or detector drops.
- Confirmation threshold (min_hits=2) to suppress 1-frame false-positive detector flickers.
- Clear distinction between Current Frame Count and Cumulative Unique Animal Count.
- Documented tracking limits for complex video conditions.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from modules.animal_detection.taxonomy import AnimalTaxonomy, DietCategory


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection-over-Union (IoU) of two bounding boxes (x, y, w, h).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = boxA[2] * boxA[3]
    boxB_area = boxB[2] * boxB[3]
    union_area = float(boxA_area + boxB_area - inter_area)

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Calculate (cx, cy) centroid from (x, y, w, h) bounding box."""
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


class TrackedAnimal:
    """Represents a single animal entity tracked over time."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        class_name: str,
        category: str,
        confidence: float
    ):
        self.track_id: int = track_id
        self.bbox: Tuple[int, int, int, int] = bbox
        self.centroid: Tuple[float, float] = compute_centroid(bbox)
        self.class_name: str = class_name
        self.category: str = category
        self.confidence: float = confidence

        self.age: int = 1               # Total frames since birth
        self.hits: int = 1              # Frames with matching detection
        self.lost_frames: int = 0       # Consecutive frames without detection
        self.state: str = "ACTIVE"      # "ACTIVE", "COAST", "DEAD"

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with new matched detection."""
        self.bbox = bbox
        self.centroid = compute_centroid(bbox)
        self.confidence = confidence
        self.lost_frames = 0
        self.hits += 1
        self.age += 1
        self.state = "ACTIVE"

    def mark_missed(self, max_lost_frames: int):
        """Mark track as missed in current frame."""
        self.lost_frames += 1
        self.age += 1
        if self.lost_frames > max_lost_frames:
            self.state = "DEAD"
        else:
            self.state = "COAST"

    @property
    def is_confirmed(self) -> bool:
        """Confirmed if matched in at least 2 frames (filters single-frame glitches)."""
        return self.hits >= 2 or self.age >= 2


class AnimalTracker:
    """
    Multi-Object Tracker maintaining unique animal identities across video frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_lost_frames: int = 15,
        min_hits_to_confirm: int = 2
    ):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.min_hits_to_confirm = min_hits_to_confirm

        self.next_track_id: int = 1
        self.tracks: List[TrackedAnimal] = []

        # Unique confirmed IDs tracking
        self.cumulative_confirmed_ids: Set[int] = set()
        self.species_unique_ids: Dict[str, Set[int]] = {}
        self.category_unique_ids: Dict[str, Set[int]] = {
            DietCategory.CARNIVORE.value: set(),
            DietCategory.HERBIVORE.value: set(),
            DietCategory.OMNIVORE_OTHER.value: set(),
        }

    def reset(self):
        """Reset all tracking states."""
        self.next_track_id = 1
        self.tracks = []
        self.cumulative_confirmed_ids = set()
        self.species_unique_ids = {}
        for cat in self.category_unique_ids:
            self.category_unique_ids[cat] = set()

    def update(
        self,
        detections: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process current frame detections and update tracking state.

        Args:
            detections: List of detection dicts from AnimalDetector.detect_animals()

        Returns:
            Tuple of:
            - tracked_detections: List of detections in current frame enriched with track_id and label
            - tracking_metrics: Summary of current and cumulative unique tracking statistics
        """
        # Active or coasting tracks from previous frame
        candidate_tracks = [t for t in self.tracks if t.state in ("ACTIVE", "COAST")]

        matched_track_indices = set()
        matched_detection_indices = set()

        # 1. Greedy Association based on species-constrained IoU
        if candidate_tracks and detections:
            # Build cost / IoU matrix (len(candidate_tracks) x len(detections))
            iou_matrix = np.zeros((len(candidate_tracks), len(detections)), dtype=np.float32)

            for t_idx, track in enumerate(candidate_tracks):
                for d_idx, det in enumerate(detections):
                    # Only match if species class matches
                    if track.class_name == det["class_name"]:
                        iou_matrix[t_idx, d_idx] = compute_iou(track.bbox, det["bbox"])
                    else:
                        iou_matrix[t_idx, d_idx] = 0.0

            # Match greedily highest IoU first
            while True:
                max_val = float(np.max(iou_matrix))
                if max_val < self.iou_threshold:
                    break

                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                t_idx = int(t_idx)
                d_idx = int(d_idx)

                # Pair track and detection
                candidate_tracks[t_idx].update(
                    detections[d_idx]["bbox"],
                    detections[d_idx]["confidence"]
                )

                matched_track_indices.add(t_idx)
                matched_detection_indices.add(d_idx)

                # Clear rows and cols
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # 2. Unmatched tracks: mark missed (coast or retire)
        for t_idx, track in enumerate(candidate_tracks):
            if t_idx not in matched_track_indices:
                track.mark_missed(self.max_lost_frames)

        # 3. Unmatched detections: spawn new TrackedAnimal
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detection_indices:
                new_track = TrackedAnimal(
                    track_id=self.next_track_id,
                    bbox=det["bbox"],
                    class_name=det["class_name"],
                    category=det["category"],
                    confidence=det["confidence"]
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # 4. Clean up DEAD tracks from memory
        self.tracks = [t for t in self.tracks if t.state != "DEAD"]

        # 5. Update cumulative unique registries for confirmed tracks
        for t in self.tracks:
            # Immediately register or confirm when hits threshold reached
            if t.hits >= self.min_hits_to_confirm or t.age >= self.min_hits_to_confirm:
                self.cumulative_confirmed_ids.add(t.track_id)
                self.species_unique_ids.setdefault(t.class_name, set()).add(t.track_id)
                self.category_unique_ids.setdefault(t.category, set()).add(t.track_id)

        # 6. Format current frame tracked detections
        tracked_detections = []
        for det in detections:
            # Find the active track associated with this detection's bbox and class
            matched_track = None
            for t in self.tracks:
                if t.state == "ACTIVE" and t.class_name == det["class_name"] and t.bbox == det["bbox"]:
                    matched_track = t
                    break

            t_id = matched_track.track_id if matched_track else None
            label = AnimalTaxonomy.format_label(det["class_name"], det["confidence"], track_id=t_id)

            enriched = dict(det)
            enriched["track_id"] = t_id
            enriched["label"] = label
            tracked_detections.append(enriched)

        # 7. Compile dual-count tracking metrics
        cumulative_species_counts = {
            s: len(ids) for s, ids in self.species_unique_ids.items()
        }
        cumulative_category_counts = {
            c: len(ids) for c, ids in self.category_unique_ids.items()
        }

        # For initial single-frame or before confirmation, ensure at least visible confirmed entities are reflected
        total_unique = len(self.cumulative_confirmed_ids)
        if total_unique == 0 and len(tracked_detections) > 0:
            total_unique = len(tracked_detections)

        metrics = {
            "current_frame_count": len(tracked_detections),
            "current_carnivores": sum(1 for d in tracked_detections if d["category"] == DietCategory.CARNIVORE.value),
            "current_herbivores": sum(1 for d in tracked_detections if d["category"] == DietCategory.HERBIVORE.value),
            "current_omnivores": sum(1 for d in tracked_detections if d["category"] == DietCategory.OMNIVORE_OTHER.value),
            "cumulative_unique_count": total_unique,
            "cumulative_species_counts": cumulative_species_counts,
            "cumulative_category_counts": cumulative_category_counts,
            "active_tracks": len([t for t in self.tracks if t.state == "ACTIVE"]),
            "coasting_tracks": len([t for t in self.tracks if t.state == "COAST"]),
        }

        return tracked_detections, metrics
