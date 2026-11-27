"""
Dress / Clothing Colour Detection Module for Task 4.

Uses OpenCV perceptual CIELAB color space analysis on the upper torso ROI.
Includes explicit quality gates returning 'Unable to determine' whenever:
1. The torso ROI is outside frame bounds or has fewer than 400 valid non-skin pixels.
2. The image exhibits extreme underexposure, highlight clipping, or low contrast.
3. The dominant color accounts for less than 40% of the candidate garment pixels (e.g. complex patterns).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2

from .constants import COLOR_PALETTE_CIELAB

MIN_TORSO_PIXELS = 400
MIN_DOMINANCE_THRESHOLD = 0.40


@dataclass
class DressColorResult:
    """Result of upper-torso dress/clothing color analysis."""
    color_name: str
    confidence: float
    hex_code: Optional[str]
    is_determined: bool
    reason: Optional[str] = None
    torso_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    pixel_count: int = 0
    secondary_color: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color_name": self.color_name,
            "confidence": round(self.confidence, 4),
            "hex_code": self.hex_code,
            "is_determined": self.is_determined,
            "reason": self.reason,
            "torso_box": self.torso_box,
            "pixel_count": self.pixel_count,
            "secondary_color": self.secondary_color,
        }


class DressColorAnalyzer:
    """
    Perceptual CIELAB color analyzer for clothing regions below detected faces.
    """

    @classmethod
    def get_torso_roi_box(
        cls,
        face_box: Tuple[int, int, int, int],
        img_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Projects anatomical upper torso bounding box below face:
        x: face_x - 0.25*w to face_x + 1.25*w
        y: face_y + 1.05*h to face_y + 2.80*h
        Clamped to image bounds.
        """
        img_h, img_w = img_shape[:2]
        fx, fy, fw, fh = face_box

        x1 = max(0, int(fx - 0.25 * fw))
        x2 = min(img_w, int(fx + 1.25 * fw))
        y1 = min(img_h, int(fy + 1.05 * fh))
        y2 = min(img_h, int(fy + 2.80 * fh))

        w = x2 - x1
        h = y2 - y1

        if w < 20 or h < 20:
            return None
        return (x1, y1, w, h)

    @classmethod
    def _create_skin_mask(cls, bgr_crop: np.ndarray) -> np.ndarray:
        """
        Creates an HSV mask of skin tones (neck/collarbone) to exclude from dress analysis.
        """
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 30, 50], dtype=np.uint8)
        upper_skin = np.array([25, 180, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        # Morphological dilation to ensure collar skin edges are masked
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    @classmethod
    def analyze(
        cls,
        image: np.ndarray,
        face_box: Tuple[int, int, int, int]
    ) -> DressColorResult:
        """
        Analyzes clothing color in upper torso ROI.
        Returns DressColorResult with 'Unable to determine' fallback if quality gates fail.
        """
        if image is None or image.size == 0:
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason="Input image is empty or invalid.",
            )

        torso_box = cls.get_torso_roi_box(face_box, image.shape[:2])
        if torso_box is None:
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason="Torso region is outside image frame.",
            )

        tx, ty, tw, th = torso_box
        torso_crop = image[ty : ty + th, tx : tx + tw]
        if torso_crop.size == 0:
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason="Torso crop is empty.",
                torso_box=torso_box,
            )

        # Gate 1: Mask out skin pixels (neck / throat)
        skin_mask = cls._create_skin_mask(torso_crop)
        valid_mask = (skin_mask == 0)
        valid_pixels_bgr = torso_crop[valid_mask]

        if len(valid_pixels_bgr) < MIN_TORSO_PIXELS:
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason=f"Insufficient non-skin torso visibility ({len(valid_pixels_bgr)} pixels < {MIN_TORSO_PIXELS} threshold).",
                torso_box=torso_box,
                pixel_count=len(valid_pixels_bgr),
            )

        # Convert valid pixels to CIELAB
        valid_bgr_2d = valid_pixels_bgr.reshape(-1, 1, 3)
        valid_lab_2d = cv2.cvtColor(valid_bgr_2d, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)

        # OpenCV Lab ranges: L in [0..255] (scaled from 0..100), a in [0..255] (128 center), b in [0..255] (128 center)
        # Rescale OpenCV Lab to standard CIE Lab (L in [0, 100], a in [-128, 127], b in [-128, 127])
        L_std = valid_lab_2d[:, 0] * (100.0 / 255.0)
        a_std = valid_lab_2d[:, 1] - 128.0
        b_std = valid_lab_2d[:, 2] - 128.0

        # Gate 2: Extreme lighting / low contrast check
        mean_L = float(np.mean(L_std))
        std_L = float(np.std(L_std))
        mean_chroma = float(np.mean(np.sqrt(a_std**2 + b_std**2)))

        if mean_L < 8.0 and mean_chroma < 4.0:
            # Extreme underexposure / black hole sensor noise
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason="Severe underexposure / Low lighting in torso region.",
                torso_box=torso_box,
                pixel_count=len(valid_pixels_bgr),
            )

        if mean_L > 98.0 and std_L < 2.0:
            # Clipped highlights / blown out white background bleed
            return DressColorResult(
                color_name="Unable to determine",
                confidence=0.0,
                hex_code=None,
                is_determined=False,
                reason="Severe highlight blowout / Overexposed torso region.",
                torso_box=torso_box,
                pixel_count=len(valid_pixels_bgr),
            )

        # Calculate distances to standard palette anchors
        # Prepare palette arrays
        palette_names = list(COLOR_PALETTE_CIELAB.keys())
        palette_anchors = np.array([
            [COLOR_PALETTE_CIELAB[name]["L"], COLOR_PALETTE_CIELAB[name]["a"], COLOR_PALETTE_CIELAB[name]["b"]]
            for name in palette_names
        ], dtype=np.float32)

        # Pixel-wise CIE Delta-E Euclidean distances
        std_pixels = np.column_stack([L_std, a_std, b_std])  # (N, 3)
        diffs = std_pixels[:, np.newaxis, :] - palette_anchors[np.newaxis, :, :]  # (N, P, 3)
        delta_e = np.sqrt(np.sum(diffs**2, axis=2))  # (N, P)
        nearest_palette_idx = np.argmin(delta_e, axis=1)  # (N,)

        # Compute cluster frequencies
        counts = np.bincount(nearest_palette_idx, minlength=len(palette_names))
        total_valid = len(std_pixels)
        ratios = counts / float(total_valid)

        sorted_indices = np.argsort(counts)[::-1]
        top_idx = sorted_indices[0]
        top_ratio = float(ratios[top_idx])
        top_color_name = palette_names[top_idx]

        # Gate 3: Minimum Dominance Threshold (40%)
        if top_ratio < MIN_DOMINANCE_THRESHOLD:
            return DressColorResult(
                color_name="Unable to determine",
                confidence=top_ratio,
                hex_code=None,
                is_determined=False,
                reason=f"Low color dominance ({top_ratio*100:.1f}% < 40% threshold) due to multi-color pattern or background noise.",
                torso_box=torso_box,
                pixel_count=total_valid,
            )

        # Secondary color identification if second cluster represents >= 20%
        secondary_color_str = None
        if len(sorted_indices) > 1:
            sec_idx = sorted_indices[1]
            sec_ratio = float(ratios[sec_idx])
            if sec_ratio >= 0.20:
                secondary_color_str = f"{palette_names[sec_idx]} ({sec_ratio*100:.1f}%)"

        hex_code = COLOR_PALETTE_CIELAB[top_color_name]["hex"]

        return DressColorResult(
            color_name=top_color_name,
            confidence=top_ratio,
            hex_code=hex_code,
            is_determined=True,
            reason=None,
            torso_box=torso_box,
            pixel_count=total_valid,
            secondary_color=secondary_color_str,
        )
