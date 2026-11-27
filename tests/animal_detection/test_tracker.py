"""
Unit Tests for AnimalTracker Multi-Object Tracking Engine.
Validates Phase 2 requirements:
1. Persistent track ID across consecutive video frames (no duplicate counting).
2. Temporary missed detections handling (coasting state up to 15 frames).
3. Re-association after occlusion without spawning new track ID.
4. Track retirement after exceeding max_lost_frames (>15).
5. New entity spawning with distinct track ID.
6. Multi-species tracking independence.
7. Current Frame Count vs Cumulative Unique Count disambiguation.
8. Reset lifecycle.
"""

import unittest
from modules.animal_detection.tracker import AnimalTracker, compute_iou, compute_centroid
from modules.animal_detection.taxonomy import DietCategory


class TestAnimalTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = AnimalTracker(iou_threshold=0.30, max_lost_frames=15, min_hits_to_confirm=2)

    def test_01_iou_and_centroid_calculations(self):
        """Verify IoU overlap and centroid formulas."""
        boxA = (100, 100, 50, 50)
        boxB = (100, 100, 50, 50)
        # Identical boxes -> IoU = 1.0
        self.assertAlmostEqual(compute_iou(boxA, boxB), 1.0)

        # Disjoint boxes -> IoU = 0.0
        boxC = (300, 300, 50, 50)
        self.assertEqual(compute_iou(boxA, boxC), 0.0)

        # Half overlap
        boxD = (125, 100, 50, 50)
        iou = compute_iou(boxA, boxD)
        self.assertGreater(iou, 0.3)
        self.assertLess(iou, 0.4)

        # Centroid
        cx, cy = compute_centroid(boxA)
        self.assertEqual(cx, 125.0)
        self.assertEqual(cy, 125.0)

    def test_02_repeated_detections_maintain_single_track_id(self):
        """Verify animal appearing in 10 consecutive frames is counted ONCE, not 10 times."""
        mock_zebra = {
            "bbox": (100, 100, 80, 80),
            "class_name": "zebra",
            "category": DietCategory.HERBIVORE.value,
            "confidence": 0.88,
        }

        # Frame 1 to 10
        for frame_idx in range(1, 11):
            # Slightly moving animal (drift by 1 pixel per frame)
            moving_bbox = (100 + frame_idx, 100, 80, 80)
            det = dict(mock_zebra)
            det["bbox"] = moving_bbox

            tracked_dets, metrics = self.tracker.update([det])

            self.assertEqual(len(tracked_dets), 1)
            # Track ID must remain #1 throughout all 10 frames
            self.assertEqual(tracked_dets[0]["track_id"], 1)

            # Current frame count is 1
            self.assertEqual(metrics["current_frame_count"], 1)

        # Crucial check: Cumulative unique count must be 1, NOT 10!
        self.assertEqual(metrics["cumulative_unique_count"], 1)
        self.assertEqual(metrics["cumulative_species_counts"].get("zebra"), 1)

    def test_03_temporary_missed_frames_reassociation(self):
        """Verify an animal temporarily missed for 3 frames reassociates with the SAME track ID."""
        det_cat = {
            "bbox": (200, 200, 60, 60),
            "class_name": "cat",
            "category": DietCategory.CARNIVORE.value,
            "confidence": 0.90,
        }

        # Frame 1: Initial detection -> Track #1
        td, m1 = self.tracker.update([det_cat])
        self.assertEqual(td[0]["track_id"], 1)

        # Frame 2: Confirmed in second frame
        td, m2 = self.tracker.update([det_cat])
        self.assertEqual(td[0]["track_id"], 1)

        # Frames 3, 4, 5: Animal temporarily occluded (0 detections)
        for _ in range(3):
            td, m_empty = self.tracker.update([])
            self.assertEqual(m_empty["current_frame_count"], 0)
            self.assertEqual(m_empty["coasting_tracks"], 1)

        # Frame 6: Animal re-emerges near previous location
        re_emerge_cat = dict(det_cat)
        re_emerge_cat["bbox"] = (205, 202, 60, 60)  # slight movement
        td, m6 = self.tracker.update([re_emerge_cat])

        # Must reassociate with existing Track #1, NOT create Track #2!
        self.assertEqual(td[0]["track_id"], 1)
        self.assertEqual(m6["cumulative_unique_count"], 1)
        self.assertEqual(m6["active_tracks"], 1)
        self.assertEqual(m6["coasting_tracks"], 0)

    def test_04_track_retirement_after_exceeding_max_lost(self):
        """Verify a track is marked DEAD and retired after 16 consecutive missed frames."""
        det_dog = {
            "bbox": (150, 150, 70, 70),
            "class_name": "dog",
            "category": DietCategory.OMNIVORE_OTHER.value,
            "confidence": 0.85,
        }

        # Establish confirmed track in frames 1 and 2
        self.tracker.update([det_dog])
        self.tracker.update([det_dog])
        self.assertEqual(len(self.tracker.tracks), 1)

        # Feed 15 empty frames -> should stay in COAST state
        for _ in range(15):
            _, m = self.tracker.update([])
            self.assertEqual(m["coasting_tracks"], 1)

        # Feed 16th empty frame -> exceeds max_lost_frames=15, retired
        _, m16 = self.tracker.update([])
        self.assertEqual(m16["coasting_tracks"], 0)
        self.assertEqual(m16["active_tracks"], 0)
        self.assertEqual(len(self.tracker.tracks), 0)

        # But cumulative unique count still remembers the animal that was seen!
        self.assertEqual(m16["cumulative_unique_count"], 1)

    def test_05_multiple_different_species_tracking(self):
        """Verify multiple different animal species are tracked with distinct IDs and species counts."""
        frame_dets = [
            {"bbox": (50, 50, 60, 60), "class_name": "cat", "category": DietCategory.CARNIVORE.value, "confidence": 0.91},
            {"bbox": (250, 250, 100, 100), "class_name": "elephant", "category": DietCategory.HERBIVORE.value, "confidence": 0.95},
        ]

        # Frame 1
        td1, m1 = self.tracker.update(frame_dets)
        self.assertEqual(m1["current_frame_count"], 2)

        # Frame 2
        td2, m2 = self.tracker.update(frame_dets)
        self.assertEqual(m2["current_frame_count"], 2)
        self.assertEqual(m2["cumulative_unique_count"], 2)

        # Species counts check
        self.assertEqual(m2["cumulative_species_counts"]["cat"], 1)
        self.assertEqual(m2["cumulative_species_counts"]["elephant"], 1)

        # Category counts check
        self.assertEqual(m2["cumulative_category_counts"][DietCategory.CARNIVORE.value], 1)
        self.assertEqual(m2["cumulative_category_counts"][DietCategory.HERBIVORE.value], 1)

    def test_06_reset_clears_state(self):
        """Verify tracker reset restores initial clean state."""
        mock_cow = {"bbox": (50, 50, 80, 80), "class_name": "cow", "category": DietCategory.HERBIVORE.value, "confidence": 0.88}
        self.tracker.update([mock_cow])
        self.tracker.update([mock_cow])
        self.assertEqual(self.tracker.cumulative_confirmed_ids, {1})

        self.tracker.reset()
        self.assertEqual(self.tracker.next_track_id, 1)
        self.assertEqual(len(self.tracker.tracks), 0)
        self.assertEqual(len(self.tracker.cumulative_confirmed_ids), 0)


if __name__ == "__main__":
    unittest.main()
