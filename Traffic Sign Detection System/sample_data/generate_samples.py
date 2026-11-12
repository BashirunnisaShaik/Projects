"""
Utility script to generate realistic synthetic traffic sign images
for testing, demonstration, and sample gallery.
"""

import cv2
import numpy as np
import os

def create_road_background(width=640, height=480):
    """Generate a realistic scenic road background (sky, trees, asphalt road)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (light blue to cyan)
    for y in range(int(height * 0.45)):
        ratio = y / (height * 0.45)
        b = int(220 - ratio * 40)
        g = int(180 - ratio * 30)
        r = int(135 - ratio * 20)
        img[y, :] = (b, g, r)

    # Road asphalt & horizon
    road_top = int(height * 0.55)
    for y in range(road_top, height):
        ratio = (y - road_top) / (height - road_top)
        gray = int(70 - ratio * 20)
        img[y, :] = (gray, gray, gray)

    # Green roadside grass / trees
    cv2.rectangle(img, (0, int(height * 0.40)), (width, road_top), (40, 110, 50), -1)

    # Road lanes
    road_pts = np.array([
        [int(width * 0.45), road_top],
        [int(width * 0.55), road_top],
        [int(width * 0.95), height],
        [int(width * 0.05), height]
    ], np.int32)
    cv2.fillPoly(img, [road_pts], (60, 60, 65))

    # Center dashed line
    cv2.line(img, (int(width * 0.50), road_top), (int(width * 0.50), height), (220, 220, 220), 4)

    # Signpost pole on right shoulder
    pole_x = int(width * 0.72)
    pole_y1 = int(height * 0.20)
    pole_y2 = int(height * 0.70)
    cv2.line(img, (pole_x, pole_y1), (pole_x, pole_y2), (160, 160, 160), 8)

    return img, pole_x, pole_y1


def generate_stop_sign(output_path):
    img, px, py = create_road_background()
    center = (px, py + 20)
    radius = 65

    # 8-sided regular polygon (Octagon)
    angles = np.linspace(0, 2 * np.pi, 9)[:-1] + (np.pi / 8)
    pts = np.array([[int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a))] for a in angles], np.int32)

    # Red fill
    cv2.fillPoly(img, [pts], (25, 25, 215))
    # White inner border
    inner_pts = np.array([[int(center[0] + (radius - 5) * np.cos(a)), int(center[1] + (radius - 5) * np.sin(a))] for a in angles], np.int32)
    cv2.polylines(img, [inner_pts], True, (255, 255, 255), 3)

    # Text "STOP"
    text = "STOP"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.9, 3)
    cv2.putText(img, text, (center[0] - tw // 2, center[1] + th // 2), font, 0.9, (255, 255, 255), 3, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_speed_limit(output_path, speed="60"):
    img, px, py = create_road_background()
    center = (px, py + 20)
    radius = 65

    # Red outer circle
    cv2.circle(img, center, radius, (20, 20, 220), -1)
    # White inner circle
    cv2.circle(img, center, radius - 14, (255, 255, 255), -1)

    # Black speed number
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(speed, font, 1.2, 4)
    cv2.putText(img, speed, (center[0] - tw // 2, center[1] + th // 2), font, 1.2, (20, 20, 20), 4, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_no_entry(output_path):
    img, px, py = create_road_background()
    center = (px, py + 20)
    radius = 65

    # Red circle
    cv2.circle(img, center, radius, (20, 20, 225), -1)
    # Thin white border
    cv2.circle(img, center, radius - 2, (255, 255, 255), 2)

    # Horizontal white bar
    bar_w = int(radius * 1.4)
    bar_h = int(radius * 0.35)
    x1 = center[0] - bar_w // 2
    y1 = center[1] - bar_h // 2
    cv2.rectangle(img, (x1, y1), (x1 + bar_w, y1 + bar_h), (255, 255, 255), -1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_turn_sign(output_path, direction="left"):
    img, px, py = create_road_background()
    center = (px, py + 20)
    radius = 65

    # Blue circle
    cv2.circle(img, center, radius, (215, 110, 15), -1)
    cv2.circle(img, center, radius - 3, (255, 255, 255), 2)

    # Arrow
    arrow_pts = []
    if direction == "left":
        # Left curve or arrow
        cv2.arrowedLine(img, (center[0] + 25, center[1] + 20), (center[0] - 25, center[1] - 5), (255, 255, 255), 8, tipLength=0.45)
    else:
        # Right curve or arrow
        cv2.arrowedLine(img, (center[0] - 25, center[1] + 20), (center[0] + 25, center[1] - 5), (255, 255, 255), 8, tipLength=0.45)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_yield_sign(output_path):
    img, px, py = create_road_background()
    center = (px, py + 20)
    size = 75

    # Inverted equilateral triangle
    p1 = (center[0] - size, center[1] - int(size * 0.6))
    p2 = (center[0] + size, center[1] - int(size * 0.6))
    p3 = (center[0], center[1] + size)
    pts = np.array([p1, p2, p3], np.int32)

    cv2.fillPoly(img, [pts], (20, 20, 220))

    # Inner white inverted triangle
    inner_size = int(size * 0.68)
    ip1 = (center[0] - inner_size, center[1] - int(size * 0.45))
    ip2 = (center[0] + inner_size, center[1] - int(size * 0.45))
    ip3 = (center[0], center[1] + int(size * 0.7))
    ipts = np.array([ip1, ip2, ip3], np.int32)
    cv2.fillPoly(img, [ipts], (255, 255, 255))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


def generate_plain_road(output_path):
    img, _, _ = create_road_background()
    # Remove pole for purely clear road
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(220):
        img[y, :] = (230 - int(y * 0.3), 190, 140)
    for y in range(220, 480):
        img[y, :] = (70, 70, 75)
    cv2.line(img, (320, 220), (320, 480), (220, 220, 220), 4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, 'static', 'images', 'sample_signs')
    os.makedirs(target_dir, exist_ok=True)

    print(f"Generating sample traffic sign images in: {target_dir}")
    generate_stop_sign(os.path.join(target_dir, 'sample_stop_sign.jpg'))
    generate_speed_limit(os.path.join(target_dir, 'sample_speed_limit_60.jpg'), "60")
    generate_no_entry(os.path.join(target_dir, 'sample_no_entry.jpg'))
    generate_turn_sign(os.path.join(target_dir, 'sample_turn_left.jpg'), "left")
    generate_turn_sign(os.path.join(target_dir, 'sample_turn_right.jpg'), "right")
    generate_yield_sign(os.path.join(target_dir, 'sample_yield_sign.jpg'))
    generate_plain_road(os.path.join(target_dir, 'sample_empty_road.jpg'))
    print("Done! Generated 7 test traffic sign images successfully.")
