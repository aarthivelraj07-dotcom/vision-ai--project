"""
Animal Detection and Multi-Object Tracking Module.
Task 2 of the AI/ML Internship Multi-Task Application.
Features:
- Multi-animal detection in single images and video frames.
- Pretrained YOLOv8 / YOLOv11 ONNX inference using OpenCV DNN on CPU.
- Tri-category dietary taxonomy: Carnivore (Red), Herbivore (Green), Omnivore/Other (Amber).
- Real-time video tracking (spatial IoU + centroid association) to eliminate duplicate animal counts.
- Independent Streamlit user interface with popups, notifications, and species metrics.
"""

from modules.animal_detection.taxonomy import (
    AnimalTaxonomy,
    DietCategory,
    COCO_ANIMAL_CLASSES,
    COCO_ANIMAL_INDICES,
    CATEGORY_COLORS_BGR,
    CATEGORY_COLORS_HEX,
)

__all__ = [
    "AnimalTaxonomy",
    "DietCategory",
    "COCO_ANIMAL_CLASSES",
    "COCO_ANIMAL_INDICES",
    "CATEGORY_COLORS_BGR",
    "CATEGORY_COLORS_HEX",
]
