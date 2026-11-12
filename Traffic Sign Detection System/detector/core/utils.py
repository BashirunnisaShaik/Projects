"""
Computer Vision Utility Functions for Traffic Sign Detection.
Handles visual overlays, bounding box annotations, ROI cropping,
and coordinate formatting.
"""

import cv2
import numpy as np
import os


def crop_roi(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Safely crop the Region of Interest (ROI) defined by (x, y, w, h).
    Includes a slight padding when possible.
    """
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]

    # Add 5% padding around the sign
    pad_x = int(w * 0.05)
    pad_y = int(h * 0.05)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    crop = image[y1:y2, x1:x2]
    return crop


def draw_detection_overlay(
    image: np.ndarray,
    detections: list,
    box_color=(0, 200, 0),
    thickness=3
) -> np.ndarray:
    """
    Draw professional bounding boxes and formatted label badges over detected signs.
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]

    # Scale font size based on image resolution
    font_scale = max(0.5, min(img_w, img_h) / 800.0)
    font_thickness = max(1, int(font_scale * 2))

    for item in detections:
        x, y, w, h = item['bbox']
        name = item.get('sign_name', 'Traffic Sign')
        conf = item.get('confidence', 90.0)
        label = f"{name} ({conf:.1f}%)"

        # Determine color based on sign category
        cat = item.get('category', 'regulatory')
        if cat == 'regulatory':
            color = (40, 40, 230)      # Vivid Red (BGR)
            badge_bg = (30, 30, 200)
        elif cat == 'mandatory':
            color = (230, 120, 20)     # Vivid Blue (BGR)
            badge_bg = (180, 90, 15)
        elif cat == 'warning':
            color = (15, 200, 235)     # Amber/Yellow (BGR)
            badge_bg = (10, 160, 200)
        else:
            color = (0, 220, 80)       # Green (BGR)
            badge_bg = (0, 160, 60)

        # Draw outer rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

        # Draw corner accents for high-tech HUD look
        corner_len = max(8, int(min(w, h) * 0.15))
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x, y), (x, y + corner_len), (255, 255, 255), thickness + 1)
        # Top-right
        cv2.line(annotated, (x + w, y), (x + w - corner_len, y), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x + w, y), (x + w, y + corner_len), (255, 255, 255), thickness + 1)
        # Bottom-left
        cv2.line(annotated, (x, y + h), (x + corner_len, y + h), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x, y + h), (x, y + h - corner_len), (255, 255, 255), thickness + 1)
        # Bottom-right
        cv2.line(annotated, (x + w, y + h), (x + w - corner_len, y + h), (255, 255, 255), thickness + 1)
        cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_len), (255, 255, 255), thickness + 1)

        # Calculate label size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Label background position
        label_y1 = max(0, y - text_h - 12)
        label_y2 = y
        label_x1 = x
        label_x2 = min(img_w, x + text_w + 14)

        # Draw filled label background
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), badge_bg, -1)
        cv2.rectangle(annotated, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), 1)

        # Draw white label text
        text_pos = (label_x1 + 6, label_y2 - 6)
        cv2.putText(
            annotated,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return annotated


def format_bbox_str(bbox: tuple) -> str:
    """Format (x, y, w, h) as standard coordinate string."""
    if not bbox or len(bbox) != 4:
        return ""
    return f"x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}"
