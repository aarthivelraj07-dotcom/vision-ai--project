"""
Animal Detection Engine using Pretrained YOLOv8 ONNX with OpenCV DNN.
Configured strictly for CPU inference (zero GPU / CUDA dependencies).
Features:
- Aspect-ratio preserving letterbox preprocessing to (640, 640).
- Forward-pass inference via cv2.dnn.
- Slicing and filtering for the 10 MS COCO animal species.
- Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes.
- Three-category dietary assignment via AnimalTaxonomy.
- Clean bounding box and label rendering.
"""

import os
import threading
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

from modules.animal_detection.taxonomy import (
    AnimalTaxonomy,
    DietCategory,
    COCO_ANIMAL_CLASSES,
    COCO_ANIMAL_INDICES,
)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "animal_detection",
    "yolov8n.onnx"
)

DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_NMS_THRESHOLD = 0.45


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target_size with aspect-ratio preservation and symmetric padding.
    Returns:
        (letterboxed_image, scale_factor, (pad_x, pad_y))
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )

    return padded, scale, (pad_left, pad_top)


class AnimalDetector:
    """
    OpenCV DNN Animal Detector executing YOLOv8 ONNX model on CPU.
    Thread-safe singleton pattern ensures model is loaded only once.
    """
    _instance: Optional["AnimalDetector"] = None
    _lock = threading.RLock()

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_path = self._resolve_model_path(model_path)
        self.net = None
        self.is_loaded = False

        self._load_network()

    @classmethod
    def get_instance(
        cls,
        model_path: Optional[str] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD
    ) -> "AnimalDetector":
        """Thread-safe singleton accessor."""
        with cls._lock:
            if cls._instance is None or not cls._instance.is_loaded:
                cls._instance = cls(
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (useful for isolated tests)."""
        with cls._lock:
            cls._instance = None

    def _resolve_model_path(self, path: Optional[str]) -> str:
        """Resolve path to yolov8n.onnx across workspace locations."""
        if path and os.path.exists(path):
            return os.path.abspath(path)

        if os.path.exists(DEFAULT_MODEL_PATH):
            return DEFAULT_MODEL_PATH

        # Fallback to relative models/animal_detection/yolov8n.onnx
        rel_path = os.path.join("models", "animal_detection", "yolov8n.onnx")
        if os.path.exists(rel_path):
            return os.path.abspath(rel_path)

        raise FileNotFoundError(
            f"Animal detection ONNX model not found at {DEFAULT_MODEL_PATH}. "
            "Ensure yolov8n.onnx is staged in models/animal_detection/."
        )

    def _load_network(self):
        """Initialize OpenCV DNN Net with CPU backend."""
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Strictly configure CPU execution
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.is_loaded = True

    def detect_animals(
        self,
        image: np.ndarray,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect animals in an image (BGR format).
        Returns list of detection dictionaries:
        [
            {
                'bbox': (x, y, w, h),
                'class_name': str,
                'category': str ('Carnivore' | 'Herbivore' | 'Omnivore / Other'),
                'confidence': float,
                'color_bgr': (B, G, R),
                'color_hex': str,
                'label': str
            }, ...
        ]
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        if not self.is_loaded or self.net is None:
            return []

        # Ensure image has valid dimensions and convert to 3-channel BGR
        try:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] == 1:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] != 3:
                    return []
            else:
                return []
        except Exception:
            return []

        c_thresh = self.conf_threshold if conf_threshold is None else conf_threshold
        n_thresh = self.nms_threshold if nms_threshold is None else nms_threshold

        orig_h, orig_w = image.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return []

        try:
            # 1. Letterbox Preprocessing to (640, 640)
            padded, scale, (pad_x, pad_y) = letterbox(image, target_size=(640, 640))

            # 2. Blob generation: BGR to RGB, scale to [0, 1]
            blob = cv2.dnn.blobFromImage(
                padded,
                scalefactor=1.0 / 255.0,
                size=(640, 640),
                swapRB=True,
                crop=False
            )

            # 3. Model Forward Pass
            self.net.setInput(blob)
            raw_output = self.net.forward()
        except Exception:
            return []

        # Output shape parsing
        # YOLOv8 format: (1, 84, 8400) -> transpose to (8400, 84)
        if len(raw_output.shape) == 3:
            if raw_output.shape[1] < raw_output.shape[2]:
                predictions = np.transpose(raw_output[0], (1, 0))  # (8400, 84)
            else:
                predictions = raw_output[0]                        # (8400, 84)
        else:
            predictions = raw_output

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        # 4. Parse Detections and filter for the 10 COCO animal classes
        # Columns 0..3: [xc, yc, w, h], Columns 4..83: 80 COCO class probabilities
        class_scores = predictions[:, 4:]  # (8400, 80)
        animal_scores = class_scores[:, COCO_ANIMAL_INDICES]  # (8400, 10)

        best_animal_idx = np.argmax(animal_scores, axis=1)
        best_scores = np.max(animal_scores, axis=1)

        valid_mask = best_scores >= c_thresh
        valid_indices = np.where(valid_mask)[0]

        for idx in valid_indices:
            score = float(best_scores[idx])
            coco_class_id = COCO_ANIMAL_INDICES[best_animal_idx[idx]]

            row = predictions[idx]
            xc = float(row[0])
            yc = float(row[1])
            w = float(row[2])
            h = float(row[3])

            x1 = int(round((xc - w / 2.0 - pad_x) / scale))
            y1 = int(round((yc - h / 2.0 - pad_y) / scale))
            w_orig = int(round(w / scale))
            h_orig = int(round(h / scale))

            # Clamp to image boundaries
            x1 = max(0, min(orig_w - 1, x1))
            y1 = max(0, min(orig_h - 1, y1))
            w_orig = max(1, min(orig_w - x1, w_orig))
            h_orig = max(1, min(orig_h - y1, h_orig))

            boxes.append([x1, y1, w_orig, h_orig])
            confidences.append(score)
            class_ids.append(coco_class_id)

        if not boxes:
            return []

        # 5. Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            score_threshold=c_thresh,
            nms_threshold=n_thresh
        )

        detections = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bbox = tuple(boxes[idx])
                conf = confidences[idx]
                cid = class_ids[idx]
                class_name = COCO_ANIMAL_CLASSES[cid]
                category = AnimalTaxonomy.get_category(class_name)
                color_bgr = AnimalTaxonomy.get_color_bgr(category)
                color_hex = AnimalTaxonomy.get_color_hex(category)
                label = AnimalTaxonomy.format_label(class_name, conf)

                detections.append({
                    "bbox": bbox,
                    "class_name": class_name,
                    "category": category.value,
                    "confidence": conf,
                    "color_bgr": color_bgr,
                    "color_hex": color_hex,
                    "label": label,
                })

        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Render styled bounding boxes and category-coded labels onto image:
        - Red for Carnivore
        - Green for Herbivore
        - Amber for Omnivore / Other
        """
        annotated = image.copy()
        if not detections:
            return annotated

        for det in detections:
            x, y, w, h = det["bbox"]
            color = det.get("color_bgr", (0, 180, 255))
            label = det.get("label", det.get("class_name", "Animal"))

            # Draw bounding box (thickness 3 for crisp visibility)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

            if show_labels:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Header background position
                header_y1 = max(0, y - th - 10)
                header_y2 = y
                cv2.rectangle(annotated, (x, header_y1), (x + tw + 10, header_y2), color, -1)

                # Label text (white on category-color banner)
                cv2.putText(
                    annotated,
                    label,
                    (x + 5, header_y2 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        return annotated

    def get_summary_statistics(
        self,
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate species-wise and category-wise statistics from detections list.
        """
        total = len(detections)
        carnivores = sum(1 for d in detections if d["category"] == DietCategory.CARNIVORE.value)
        herbivores = sum(1 for d in detections if d["category"] == DietCategory.HERBIVORE.value)
        omnivores = sum(1 for d in detections if d["category"] == DietCategory.OMNIVORE_OTHER.value)

        species_counts: Dict[str, int] = {}
        species_confidences: Dict[str, List[float]] = {}
        species_categories: Dict[str, str] = {}

        for d in detections:
            s_name = d["class_name"]
            species_counts[s_name] = species_counts.get(s_name, 0) + 1
            species_confidences.setdefault(s_name, []).append(d["confidence"])
            species_categories[s_name] = d["category"]

        species_breakdown = []
        for s_name, count in sorted(species_counts.items(), key=lambda x: -x[1]):
            avg_conf = float(np.mean(species_confidences[s_name]))
            species_breakdown.append({
                "species": s_name.capitalize(),
                "category": species_categories[s_name],
                "count": count,
                "avg_confidence": avg_conf,
                "color_hex": AnimalTaxonomy.get_color_hex(species_categories[s_name]),
            })

        return {
            "total_count": total,
            "carnivore_count": carnivores,
            "herbivore_count": herbivores,
            "omnivore_other_count": omnivores,
            "species_counts": species_counts,
            "species_breakdown": species_breakdown,
        }
