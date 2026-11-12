"""
Traffic Sign Detection and Classification Module.
Implements color thresholding (HSV), edge detection (Canny),
contour geometry analysis, shape classification, and symbol recognition.
"""

import cv2
import numpy as np
import time
from .preprocessing import preprocess_pipeline, apply_morphology
from .utils import crop_roi, draw_detection_overlay, format_bbox_str


CATEGORY_EXPLANATIONS = {
    'prohibitory': 'This sign indicates that a particular action is not allowed.',
    'regulatory': 'This sign gives rules that drivers must follow.',
    'warning': 'This sign warns drivers about possible danger ahead.',
    'mandatory': 'This sign indicates an action that drivers must follow.',
    'informative': 'This sign provides useful information to drivers.',
    'information': 'This sign provides useful information to drivers.',
    'none': 'No recognizable traffic sign identified.',
}

CATEGORY_DISPLAY_NAMES = {
    'prohibitory': 'Prohibited',
    'regulatory': 'Regulatory',
    'warning': 'Warning',
    'mandatory': 'Mandatory',
    'informative': 'Information',
    'none': 'None',
}

# Database of sign metadata, categories, road safety meanings and driver actions
SIGN_CATALOG = {
    'Unknown / Unsupported Sign': {
        'category': 'warning',
        'shape': 'Unknown',
        'dominant_color': 'Variable',
        'meaning': 'The detected sign is either not in the supported catalog or cannot be identified with high confidence.',
        'driver_action': 'Proceed with caution and observe standard road rules.',
        'description': 'Unrecognized or unsupported road sign.'
    },
    'Stop Sign': {
        'category': 'regulatory',
        'shape': 'Octagon',
        'dominant_color': 'Red & White',
        'meaning': 'Vehicles must stop before proceeding.',
        'driver_action': 'Stop the vehicle before moving ahead.',
        'description': 'Mandatory stop sign: Vehicles must stop completely before proceeding.'
    },
    'No Entry': {
        'category': 'prohibitory',
        'shape': 'Circle',
        'dominant_color': 'Red & White',
        'meaning': 'Entry is not allowed for any vehicle.',
        'driver_action': 'Do not enter this road or area.',
        'description': 'Prohibitory sign: Entry is strictly forbidden for all vehicular traffic.'
    },
    'No Parking': {
        'category': 'prohibitory',
        'shape': 'Circle',
        'dominant_color': 'Red & Blue',
        'meaning': 'Parking is not allowed in this area.',
        'driver_action': 'Do not park your vehicle here.',
        'description': 'Prohibitory sign: Parking or leaving vehicles unattended is forbidden.'
    },
    'No Horn': {
        'category': 'prohibitory',
        'shape': 'Circle',
        'dominant_color': 'Red & White',
        'meaning': 'Using the horn is not allowed in this area.',
        'driver_action': 'Do not use the horn in this area.',
        'description': 'Prohibitory sign: Silent zone. Honking is strictly prohibited.'
    },
    'Speed Limit Sign': {
        'category': 'regulatory',
        'shape': 'Circle',
        'dominant_color': 'Red Border & White',
        'meaning': 'The speed limit is 60 km/h.',
        'driver_action': 'Do not drive faster than 60 km/h.',
        'description': 'Speed restriction: Vehicles must not exceed the posted maximum speed limit.'
    },
    'Yield / Give Way': {
        'category': 'regulatory',
        'shape': 'Inverted Triangle',
        'dominant_color': 'Red Border & White',
        'meaning': 'Other vehicles have priority.',
        'driver_action': 'Give way to other vehicles before proceeding.',
        'description': 'Give Way: Slow down and give precedence to cross traffic.'
    },
    'Turn Left': {
        'category': 'mandatory',
        'shape': 'Circle',
        'dominant_color': 'Blue & White',
        'meaning': 'Vehicles must turn left.',
        'driver_action': 'Turn left at the intersection.',
        'description': 'Mandatory sign: Compulsory left turn required for all vehicles.'
    },
    'Turn Right': {
        'category': 'mandatory',
        'shape': 'Circle',
        'dominant_color': 'Blue & White',
        'meaning': 'Vehicles must turn right.',
        'driver_action': 'Turn right at the intersection.',
        'description': 'Mandatory sign: Compulsory right turn required for all vehicles.'
    },
    'Ahead Only': {
        'category': 'mandatory',
        'shape': 'Circle',
        'dominant_color': 'Blue & White',
        'meaning': 'Vehicles must only go straight ahead.',
        'driver_action': 'Drive straight ahead and do not turn.',
        'description': 'Mandatory sign: Traffic must proceed straight ahead only.'
    },
    'Roundabout': {
        'category': 'mandatory',
        'shape': 'Circle',
        'dominant_color': 'Blue & White',
        'meaning': 'A roundabout junction is ahead.',
        'driver_action': 'Enter the roundabout following the traffic flow.',
        'description': 'Mandatory sign: Compulsory circulatory roundabout movement.'
    },
    'Warning / Hazard': {
        'category': 'warning',
        'shape': 'Triangle',
        'dominant_color': 'Red & Yellow',
        'meaning': 'There is danger or a hazard ahead on the road.',
        'driver_action': 'Drive carefully and stay alert.',
        'description': 'Warning sign: Caution required due to approaching road hazard.'
    },
    'Pedestrian Crossing': {
        'category': 'warning',
        'shape': 'Triangle',
        'dominant_color': 'Yellow & Red',
        'meaning': 'A pedestrian crossing is ahead.',
        'driver_action': 'Slow down and watch for pedestrians.',
        'description': 'Warning sign: Pedestrian zebra crossing ahead. Yield to pedestrians.'
    },
    'School Zone Ahead': {
        'category': 'warning',
        'shape': 'Triangle',
        'dominant_color': 'Yellow & Black',
        'meaning': 'A school zone is ahead with children nearby.',
        'driver_action': 'Reduce speed and watch for children crossing.',
        'description': 'Warning sign: School zone. Drive with extra care and reduce speed.'
    },
    'Traffic Signal Ahead': {
        'category': 'warning',
        'shape': 'Triangle',
        'dominant_color': 'Red & Yellow',
        'meaning': 'A traffic signal is approaching ahead.',
        'driver_action': 'Prepare to slow down or stop at the signal.',
        'description': 'Warning sign: Traffic signal approaching ahead.'
    },
    'Hospital Ahead': {
        'category': 'informative',
        'shape': 'Rectangle',
        'dominant_color': 'Blue & White',
        'meaning': 'A hospital is located nearby.',
        'driver_action': 'Drive quietly and watch for ambulances.',
        'description': 'Information sign: Medical emergency facility in the vicinity.'
    },
    'Parking Area': {
        'category': 'informative',
        'shape': 'Rectangle',
        'dominant_color': 'Blue & White',
        'meaning': 'A parking facility is available.',
        'driver_action': 'Park your vehicle in the designated area.',
        'description': 'Information sign: Designated vehicle parking bay.'
    },
}


def create_color_masks(hsv_img: np.ndarray) -> dict:
    """
    Generate HSV color segmentation masks for red, blue, and yellow signs.
    """
    # Red spans both ends of the HSV hue spectrum (0-10 and 160-180)
    lower_red1 = np.array([0, 65, 50], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 65, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Blue (Mandatory signs)
    lower_blue = np.array([95, 65, 50], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Yellow / Amber (Warning signs)
    lower_yellow = np.array([15, 80, 80], dtype=np.uint8)
    upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

    # Green (Permissive mandatory signs, e.g., Canadian MUTCD green circles)
    lower_green = np.array([38, 100, 60], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)
    mask_green = cv2.inRange(hsv_img, lower_green, upper_green)

    # Morphological clean up
    mask_red = apply_morphology(mask_red, op='close', kernel_size=5)
    mask_blue = apply_morphology(mask_blue, op='close', kernel_size=5)
    mask_yellow = apply_morphology(mask_yellow, op='close', kernel_size=5)
    mask_green = apply_morphology(mask_green, op='close', kernel_size=5)

    # Remove isolated pixel noise from non-sign textures (e.g., mortar on brick walls)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_open)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_open)

    return {
        'red': mask_red,
        'blue': mask_blue,
        'yellow': mask_yellow,
        'green': mask_green
    }


def analyze_red_sign(roi_bgr: np.ndarray, contour_poly, circularity: float) -> tuple:
    """
    Differentiate among Red traffic signs:
    - Stop Sign (Octagonal or high red coverage with STOP)
    - No Entry (Circle with central white horizontal bar)
    - Speed Limit (Circle with white center & dark numerals)
    - Parking Area (Circle with white center & letter 'P')
    - Yield / Give Way (Inverted triangle pointing downwards)
    - School Zone Ahead / Pedestrian Crossing (Upright red triangle with child/pedestrian silhouettes)
    - Upright Warning / Unsupported Sign (Upright red triangle, Vienna convention)
    - No Parking (Circle with blue background and red cross/slash)
    """
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return 'Stop Sign', 75.0, 'Octagon'

    num_vertices = len(contour_poly) if contour_poly is not None else 0
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Triangle Check (Yield vs Upright Warning Triangle)
    if num_vertices == 3:
        pts = contour_poly.reshape(-1, 2)
        min_y = np.min(pts[:, 1])
        max_y = np.max(pts[:, 1])
        box_h = max_y - min_y if max_y > min_y else h
        mean_y_ratio = np.mean(pts[:, 1] - min_y) / float(box_h + 1e-5)

        # Inverted Triangle (Yield): 2 points at top (y~0), 1 at bottom (y~1) -> mean_y_ratio < 0.50
        if mean_y_ratio < 0.50:
            # Yield signs have a clear white/light center without dark pictorial icons
            pad_h, pad_w = int(h * 0.25), int(w * 0.25)
            center_gray = gray_roi[pad_h:h-pad_h, pad_w:w-pad_w]
            dark_ratio = np.sum(center_gray < 80) / float(center_gray.size + 1e-5) if center_gray.size > 0 else 0
            if dark_ratio < 0.15:
                return 'Yield / Give Way', 92.5, 'Inverted Triangle'
            return 'Unknown / Unsupported Sign', 75.0, 'Inverted Triangle'
        else:
            # Upright Triangle: Danger / Warning Sign (Vienna Convention)
            # Analyze inner dark silhouettes
            pad_h, pad_w = int(h * 0.20), int(w * 0.20)
            center_gray = gray_roi[pad_h:h-pad_h, pad_w:w-pad_w]
            ch, cw = center_gray.shape
            dark_ratio = np.sum(center_gray < 80) / float(center_gray.size + 1e-5) if center_gray.size > 0 else 0

            if dark_ratio > 0.08:
                _, thresh = cv2.threshold(center_gray, 90, 255, cv2.THRESH_BINARY_INV)
                cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Check for heads (School Zone has 2 children figures side-by-side)
                heads = []
                for c in cnts:
                    a = cv2.contourArea(c)
                    if 150 < a < 6000:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        ar = bw / float(bh)
                        if 0.5 < ar < 1.7 and by < ch * 0.65:
                            peri = cv2.arcLength(c, True)
                            circ = 4.0 * np.pi * a / (peri * peri + 1e-5)
                            if circ > 0.35:
                                heads.append((bx, by, bw, bh))

                if len(heads) >= 2:
                    return 'School Zone Ahead', 92.0, 'Triangle'
                elif len(heads) == 1:
                    return 'Pedestrian Crossing', 91.0, 'Triangle'
                elif dark_ratio >= 0.15:
                    mid_dark = np.sum(center_gray[int(ch*0.35):int(ch*0.75), :] < 80) / float(ch*0.4*cw + 1e-5)
                    if mid_dark > 0.25:
                        return 'School Zone Ahead', 89.0, 'Triangle'
                return 'Unknown / Unsupported Sign', 85.0, 'Triangle'
            return 'Warning / Hazard', 80.0, 'Triangle'

    # Central region analysis
    pad_h = int(h * 0.25)
    pad_w = int(w * 0.25)
    center_roi = roi_bgr[pad_h:h-pad_h, pad_w:w-pad_w]
    if center_roi.size == 0:
        return None, 0.0, 'Unknown'
    center_hsv = hsv_roi[pad_h:h-pad_h, pad_w:w-pad_w]
    center_gray = gray_roi[pad_h:h-pad_h, pad_w:w-pad_w]

    lower_white = np.array([0, 0, 160], dtype=np.uint8)
    upper_white = np.array([180, 60, 255], dtype=np.uint8)
    center_white_mask = cv2.inRange(center_hsv, lower_white, upper_white)
    center_white_ratio = np.sum(center_white_mask > 0) / float(center_roi.shape[0] * center_roi.shape[1] + 1e-5)

    # Blue center (No Parking)
    lower_blue = np.array([95, 50, 50], dtype=np.uint8)
    upper_blue = np.array([135, 255, 255], dtype=np.uint8)
    center_blue_mask = cv2.inRange(center_hsv, lower_blue, upper_blue)
    blue_ratio = np.sum(center_blue_mask > 0) / float(center_roi.shape[0] * center_roi.shape[1] + 1e-5)
    if blue_ratio > 0.15:
        return 'No Parking', 88.0, 'Circle'

    # Red masks
    lower_red1 = np.array([0, 65, 50], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 65, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask_roi = cv2.bitwise_or(
        cv2.inRange(hsv_roi, lower_red1, upper_red1),
        cv2.inRange(hsv_roi, lower_red2, upper_red2)
    )
    red_in_center = np.sum(red_mask_roi[pad_h:h-pad_h, pad_w:w-pad_w] > 0) / float(center_roi.size // 3 + 1e-5)

    # Check horizontal white bar for "No Entry" (strictly thick solid horizontal bar)
    mid_y1 = int(h * 0.38)
    mid_y2 = int(h * 0.62)
    mid_strip = roi_bgr[mid_y1:mid_y2, int(w*0.15):int(w*0.85)]
    if mid_strip.size > 0:
        mid_strip_hsv = cv2.cvtColor(mid_strip, cv2.COLOR_BGR2HSV)
        white_strip_mask = cv2.inRange(mid_strip_hsv, lower_white, upper_white)
        white_bar_ratio = np.sum(white_strip_mask > 0) / float(mid_strip.shape[0] * mid_strip.shape[1] + 1e-5)
        top_strip = roi_bgr[int(h*0.1):int(h*0.3), int(w*0.2):int(w*0.8)]
        top_white_ratio = 0.0
        if top_strip.size > 0:
            top_strip_hsv = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV)
            top_white_mask = cv2.inRange(top_strip_hsv, lower_white, upper_white)
            top_white_ratio = np.sum(top_white_mask > 0) / float(top_strip.shape[0] * top_strip.shape[1] + 1e-5)

        if white_bar_ratio >= 0.55 and top_white_ratio < 0.22:
            return 'No Entry', 94.5, 'Circle'

    # Check for symbols in white center (Letter "P" vs Speed Limit digits)
    dark_in_center = np.sum(center_gray < 80) / float(center_gray.size + 1e-5)
    if center_white_ratio > 0.25 and dark_in_center > 0.04:
        border_h = max(4, int(h * 0.15))
        outer_top = hsv_roi[:border_h, :]
        red_outer = np.sum(cv2.inRange(outer_top, lower_red1, upper_red1) > 0) + np.sum(cv2.inRange(outer_top, lower_red2, upper_red2) > 0)
        if (red_outer / float(outer_top.shape[0]*outer_top.shape[1] + 1e-5)) > 0.08:
            # If there is a diagonal red prohibition slash in the center: No Horn
            if red_in_center > 0.15:
                return 'No Horn', 92.0, 'Circle'

            # Segment the dark symbol to distinguish "P" (Parking) from digits (Speed Limit)
            _, thresh = cv2.threshold(center_gray, 90, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(c)
                if bh >= 15 and bw >= 8:
                    sym = thresh[by:by+bh, bx:bx+bw]
                    sh, sw = sym.shape
                    tl = np.sum(sym[:sh//2, :sw//2] > 0) / float(sh*sw/4 + 1e-5)
                    tr = np.sum(sym[:sh//2, sw//2:] > 0) / float(sh*sw/4 + 1e-5)
                    bl = np.sum(sym[sh//2:, :sw//2] > 0) / float(sh*sw/4 + 1e-5)
                    br = np.sum(sym[sh//2:, sw//2:] > 0) / float(sh*sw/4 + 1e-5)

                    # Letter "P" has loop in TR, stem in TL & BL, and empty BR (< 0.28)
                    is_letter_p = (tl > 0.35 and tr > 0.35 and bl > 0.35 and br < 0.28 and (bw/float(bh)) < 0.95)
                    if is_letter_p:
                        return 'Parking Area', 94.0, 'Circle'
            return 'Speed Limit Sign', 92.0, 'Circle'

    # Stop Sign: octagon or circular red sign with predominant red center
    if num_vertices == 8 or (num_vertices >= 7 and circularity > 0.70):
        if red_in_center > 0.38:
            return 'Stop Sign', 96.0, 'Octagon'
    if red_in_center > 0.52:
        return 'Stop Sign', 90.0, 'Octagon'
    if circularity > 0.65 and red_in_center > 0.38:
        return 'Stop Sign', 85.0, 'Octagon'

    # If it is clearly a circular/polygonal red road sign but uncatalogued symbol
    if circularity > 0.60:
        return 'Unknown / Unsupported Sign', 75.0, 'Circle'

    return None, 0.0, 'Unknown'


def analyze_directional_arrow(inner_mask: np.ndarray) -> tuple:
    """
    Directional arrow classifier based on motion gradient delta_x = top_x - bot_x:
    - Turn Left: arrow curves and points left (delta_x < -0.12 or leftmost tip < 0.10)
    - Turn Right: arrow curves and points right (delta_x > +0.12 or rightmost tip > 0.90)
    - Ahead Only: straight arrow (|delta_x| <= 0.12)
    - Roundabout: 3 circular arrows (all 4 quadrants occupied and hollow center)
    """
    ih, iw = inner_mask.shape
    total_w = np.sum(inner_mask > 0)
    if total_w < 25:
        return None, 0.0, 'Unknown'

    # Check 4 quadrants for Roundabout
    q_tl = np.sum(inner_mask[:ih//2, :iw//2] > 0)
    q_tr = np.sum(inner_mask[:ih//2, iw//2:] > 0)
    q_bl = np.sum(inner_mask[ih//2:, :iw//2] > 0)
    q_br = np.sum(inner_mask[ih//2:, iw//2:] > 0)

    center_core = inner_mask[int(ih*0.35):int(ih*0.65), int(iw*0.35):int(iw*0.65)]
    center_density = np.sum(center_core > 0) / float(center_core.size + 1e-5)
    min_q_ratio = min(q_tl, q_tr, q_bl, q_br) / float(total_w + 1e-5)

    if center_density < 0.12 and min_q_ratio > 0.10:
        return 'Roundabout', 91.0, 'Circle'

    # Stroke motion gradient from bottom to top
    bot = inner_mask[int(ih*0.65):, :]
    bot_pts = np.where(bot > 0)
    bot_x = np.median(bot_pts[1]) if len(bot_pts[1]) > 0 else iw / 2.0

    top = inner_mask[:int(ih*0.50), :]
    top_pts = np.where(top > 0)
    top_x = np.median(top_pts[1]) if len(top_pts[1]) > 0 else iw / 2.0

    delta_x = (top_x - bot_x) / float(iw)

    top_cnts, _ = cv2.findContours(top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_x, max_x = 0.5, 0.5
    if top_cnts:
        pts = max(top_cnts, key=cv2.contourArea).reshape(-1, 2)
        min_x = np.min(pts[:, 0]) / float(iw)
        max_x = np.max(pts[:, 0]) / float(iw)

    if delta_x < -0.12 or (min_x < 0.10 and delta_x < 0.05):
        return 'Turn Left', 93.0, 'Circle'
    elif delta_x > 0.12 or (max_x > 0.90 and delta_x > -0.05):
        return 'Turn Right', 93.0, 'Circle'
    else:
        return 'Ahead Only', 89.5, 'Circle'


def analyze_blue_sign(roi_bgr: np.ndarray) -> tuple:
    """
    Analyze Blue directional/mandatory traffic signs:
    - Turn Left
    - Turn Right
    - Ahead Only
    - Roundabout
    """
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20:
        return None, 0.0, 'Unknown'

    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150], dtype=np.uint8)
    upper_white = np.array([180, 70, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

    pad_h = int(h * 0.15)
    pad_w = int(w * 0.15)
    inner_mask = white_mask[pad_h:h-pad_h, pad_w:w-pad_w]

    return analyze_directional_arrow(inner_mask)


def analyze_green_sign(roi_bgr: np.ndarray, circularity: float = 0.85) -> tuple:
    """
    Analyze Green permissive circular traffic signs (e.g. Canadian MUTCD turn left / turn right).
    """
    h, w = roi_bgr.shape[:2]
    if h < 20 or w < 20 or circularity < 0.60:
        return None, 0.0, 'Unknown'

    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    pad_h = int(h * 0.15)
    pad_w = int(w * 0.15)
    inner_gray = gray_roi[pad_h:h-pad_h, pad_w:w-pad_w]
    _, inner_dark = cv2.threshold(inner_gray, 100, 255, cv2.THRESH_BINARY_INV)

    return analyze_directional_arrow(inner_dark)


def analyze_yellow_sign(roi_bgr: np.ndarray, num_vertices: int, approx_poly=None) -> tuple:
    """
    Analyze Yellow caution/warning traffic signs:
    - Pedestrian Crossing
    - Warning / Hazard
    - School Zone Ahead
    """
    h, w = roi_bgr.shape[:2]
    if h < 25 or w < 25:
        return None, 0.0, 'Unknown'

    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    lower_yellow = np.array([15, 80, 75], dtype=np.uint8)
    upper_yellow = np.array([38, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / float(h * w + 1e-5)
    if yellow_ratio < 0.20:
        return None, 0.0, 'Unknown'

    inner_pad_y = int(h * 0.20)
    inner_pad_x = int(w * 0.20)
    inner_gray = gray_roi[inner_pad_y:h-inner_pad_y, inner_pad_x:w-inner_pad_x]
    if inner_gray.size == 0:
        return None, 0.0, 'Unknown'

    dark_pixels = np.sum(inner_gray < 75) / float(inner_gray.size + 1e-5)
    if dark_pixels < 0.04:
        return None, 0.0, 'Unknown'

    n_verts = len(approx_poly) if approx_poly is not None else num_vertices
    is_true_triangle = (n_verts == 3)
    perimeter = cv2.arcLength(approx_poly, True) if approx_poly is not None and len(approx_poly) > 0 else 0
    area = w * h
    aspect = w / float(h)
    circularity_approx = 4.0 * np.pi * area / (perimeter * perimeter + 1e-5) if perimeter > 0 else 0.0
    is_diamond = (n_verts == 4 and circularity_approx > 0.55 and 0.75 <= aspect <= 1.30)
    is_rect = (0.40 <= aspect <= 2.50 and (aspect < 0.75 or aspect > 1.30 or not is_diamond))

    if not is_true_triangle and not is_diamond and not is_rect:
        return None, 0.0, 'Unknown'

    shape_name = 'Triangle' if is_true_triangle else ('Diamond' if is_diamond else 'Rectangle')
    ih, iw = inner_gray.shape

    # 1. Roundabout Check (3 circular arrows in a ring with hollow center, e.g. MUTCD W16-1 / Canadian Roundabout Ahead)
    q_tl = np.sum(inner_gray[:ih//2, :iw//2] < 75)
    q_tr = np.sum(inner_gray[:ih//2, iw//2:] < 75)
    q_bl = np.sum(inner_gray[ih//2:, :iw//2] < 75)
    q_br = np.sum(inner_gray[ih//2:, iw//2:] < 75)
    total_dark_pts = q_tl + q_tr + q_bl + q_br

    center_core = (inner_gray[int(ih*0.35):int(ih*0.65), int(iw*0.35):int(iw*0.65)] < 75)
    center_density = np.sum(center_core) / float(center_core.size + 1e-5)
    min_q = min(q_tl, q_tr, q_bl, q_br) / float(total_dark_pts + 1e-5)

    if center_density < 0.10 and min_q > 0.12 and total_dark_pts > 40:
        return 'Roundabout', 92.0, shape_name

    top_zone = inner_gray[:int(ih*0.35), :]
    mid_zone = inner_gray[int(ih*0.35):int(ih*0.70), :]
    bot_zone = inner_gray[int(ih*0.70):, :]

    dark_top = np.sum(top_zone < 75) / float(top_zone.size + 1e-5)
    dark_mid = np.sum(mid_zone < 75) / float(mid_zone.size + 1e-5)
    dark_bot = np.sum(bot_zone < 75) / float(bot_zone.size + 1e-5)

    # 2. Pedestrian Crossing: walking figure
    ped_silhouette = (
        0.03 <= dark_top <= 0.25
        and dark_mid > 0.08
        and dark_mid > dark_top * 1.25
        and dark_pixels >= 0.09
        and dark_pixels < 0.45
    )
    if ped_silhouette:
        return 'Pedestrian Crossing', 91.0, shape_name

    # 3. School zone: high density figures (must have solid central content, diamond or triangle warning signs)
    if (is_diamond or is_true_triangle) and dark_pixels >= 0.22 and (dark_mid > 0.20 or dark_bot > 0.20):
        if center_density >= 0.10:
            return 'School Zone Ahead', 90.0, shape_name

    # 4. General warning / hazard / directional chevron
    if dark_pixels >= 0.08:
        return 'Warning / Hazard', 86.0, shape_name

    return 'Unknown / Unsupported Sign', 78.0, shape_name


def detect_traffic_signs(image: np.ndarray) -> dict:
    """
    Main detection pipeline:
    1. Input validation & color channel normalization
    2. Preprocessing (Resize, noise reduction, color spaces)
    3. Color segmentation into Red, Blue, Yellow candidate regions
    4. Contour detection and aspect-ratio / area filtering
    5. Shape classification & geometry estimation
    6. Inner feature analysis & recognition
    7. Visual overlay generation & confidence scoring
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid image provided for traffic sign detection: image array is None or empty.")

    # Ensure 3-channel BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must have 3 color channels (BGR) for traffic sign detection.")

    start_time = time.time()

    # Preprocessing
    pipeline = preprocess_pipeline(image)
    resized_img = pipeline['resized']
    hsv_img = pipeline['hsv']
    edges = pipeline['edges']

    img_h, img_w = resized_img.shape[:2]
    img_area = img_h * img_w
    min_area = max(500, int(img_area * 0.002))       # at least 0.2% of image
    max_area = int(img_area * 0.65)                  # at most 65% of image

    color_masks = create_color_masks(hsv_img)
    detections = []

    # Process each color channel
    for color_name, mask in color_masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Bounding box & aspect ratio
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 0.45 or aspect_ratio > 2.30:
                continue

            # Polygon approximation
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            approx = cv2.approxPolyDP(cnt, 0.035 * perimeter, True)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            num_vertices = len(approx)

            # Crop Region of Interest
            roi = resized_img[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            # Classify based on detected color & shape
            if color_name == 'red':
                sign_name, conf, shape = analyze_red_sign(roi, approx, circularity)
            elif color_name == 'blue':
                sign_name, conf, shape = analyze_blue_sign(roi)
            elif color_name == 'yellow':
                sign_name, conf, shape = analyze_yellow_sign(roi, num_vertices, approx)
            elif color_name == 'green':
                sign_name, conf, shape = analyze_green_sign(roi, circularity)
            else:
                continue

            if sign_name is None:
                continue

            # Adjust confidence based on circularity/shape fidelity
            if shape in ['Circle', 'Octagon'] and circularity > 0.60:
                conf = min(98.5, conf + 3.0)
            elif shape == 'Triangle' and num_vertices == 3:
                conf = min(97.0, conf + 4.0)

            sign_meta = SIGN_CATALOG.get(sign_name, {
                'category': 'regulatory',
                'shape': shape,
                'dominant_color': color_name.capitalize(),
                'meaning': 'Standard road traffic sign regulations apply.',
                'driver_action': 'Observe road conditions and proceed with caution.',
                'description': 'Standard road traffic sign.'
            })

            detections.append({
                'sign_name': sign_name,
                'category': sign_meta.get('category', 'regulatory'),
                'shape': shape,
                'dominant_color': sign_meta.get('dominant_color', color_name.capitalize()),
                'meaning': sign_meta.get('meaning', 'Standard road traffic sign regulations apply.'),
                'driver_action': sign_meta.get('driver_action', 'Observe road conditions and proceed with caution.'),
                'confidence': round(conf, 1),
                'bbox': (x, y, w, h),
                'bbox_str': format_bbox_str((x, y, w, h)),
                'area': area,
                'description': sign_meta.get('description', 'Standard road traffic sign.'),
                'roi': crop_roi(resized_img, (x, y, w, h))
            })

    # Fallback Hough Circle detection if no color contour passed the threshold
    if not detections:
        # Try finding circular signs via HoughCircles on blurred grayscale
        blurred = pipeline['blurred_gray']
        min_radius = int(min(img_w, img_h) * 0.03)
        max_radius = int(min(img_w, img_h) * 0.45)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min(img_w, img_h) * 0.15),
            param1=100,
            param2=50,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :3]:  # Top 3 circles
                cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                x = max(0, cx - r)
                y = max(0, cy - r)
                w = min(img_w - x, 2 * r)
                h = min(img_h - y, 2 * r)
                if w < 20 or h < 20:
                    continue

                roi = resized_img[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                # Check dominant color in this circular ROI
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                red_cnt = np.sum(cv2.inRange(hsv_roi, np.array([0, 70, 50]), np.array([12, 255, 255])) > 0) + \
                          np.sum(cv2.inRange(hsv_roi, np.array([160, 70, 50]), np.array([180, 255, 255])) > 0)
                blue_cnt = np.sum(cv2.inRange(hsv_roi, np.array([95, 70, 50]), np.array([135, 255, 255])) > 0)
                green_cnt = np.sum(cv2.inRange(hsv_roi, np.array([35, 60, 40]), np.array([85, 255, 255])) > 0)

                total_pix = roi.shape[0] * roi.shape[1] + 1e-5
                if red_cnt / total_pix > 0.12:
                    sign_name, conf, shape = analyze_red_sign(roi, np.array([]), 0.85)
                elif blue_cnt / total_pix > 0.15:
                    sign_name, conf, shape = analyze_blue_sign(roi)
                elif green_cnt / total_pix > 0.12:
                    sign_name, conf, shape = analyze_green_sign(roi)
                else:
                    continue

                if sign_name is None:
                    continue

                sign_meta = SIGN_CATALOG.get(sign_name, {
                    'category': 'regulatory',
                    'shape': 'Circle',
                    'dominant_color': 'Red & White',
                    'meaning': 'Standard road traffic sign regulations apply.',
                    'driver_action': 'Observe road conditions and proceed with caution.',
                    'description': 'Standard road traffic sign.'
                })

                detections.append({
                    'sign_name': sign_name,
                    'category': sign_meta.get('category', 'regulatory'),
                    'shape': shape,
                    'dominant_color': sign_meta.get('dominant_color', 'Red & White'),
                    'meaning': sign_meta.get('meaning', 'Standard road traffic sign regulations apply.'),
                    'driver_action': sign_meta.get('driver_action', 'Observe road conditions and proceed with caution.'),
                    'confidence': round(conf, 1),
                    'bbox': (x, y, w, h),
                    'bbox_str': format_bbox_str((x, y, w, h)),
                    'area': float(w * h),
                    'description': sign_meta.get('description', 'Standard road traffic sign.'),
                    'roi': crop_roi(resized_img, (x, y, w, h))
                })

    # Sort detections by area (descending) to prioritize the most prominent sign
    detections.sort(key=lambda d: d['area'], reverse=True)

    # Filter out duplicate overlapping detections (Non-Maximum Suppression)
    final_detections = []
    for det in detections:
        x1, y1, w1, h1 = det['bbox']
        overlap = False
        for kept in final_detections:
            x2, y2, w2, h2 = kept['bbox']
            # Intersection over area
            ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            intersection = ix * iy
            if intersection > 0.4 * min(w1 * h1, w2 * h2):
                overlap = True
                break
        if not overlap:
            final_detections.append(det)

    elapsed_time = round(time.time() - start_time, 3)

    if final_detections:
        primary = final_detections[0]
        annotated_img = draw_detection_overlay(resized_img, final_detections)

        return {
            'detected': True,
            'primary_sign': primary['sign_name'],
            'category': primary['category'],
            'shape': primary['shape'],
            'dominant_color': primary['dominant_color'],
            'meaning': primary['meaning'],
            'driver_action': primary['driver_action'],
            'confidence': primary['confidence'],
            'bbox': primary['bbox'],
            'bbox_str': primary['bbox_str'],
            'description': primary['description'],
            'processing_time': elapsed_time,
            'annotated_image': annotated_img,
            'primary_roi': primary['roi'],
            'all_detections': final_detections,
            'intermediate': {
                'gray': pipeline['gray'],
                'blurred': pipeline['blurred_gray'],
                'enhanced': pipeline['enhanced_gray'],
                'edges': edges,
            }
        }
    else:
        return {
            'detected': False,
            'primary_sign': 'No Sign Detected',
            'category': 'none',
            'shape': 'None',
            'dominant_color': 'None',
            'meaning': 'Traffic sign could not be identified confidently.',
            'driver_action': 'Observe roadway conditions and proceed with caution.',
            'confidence': 0.0,
            'bbox': (0, 0, 0, 0),
            'bbox_str': '',
            'description': 'Traffic sign could not be identified confidently. Please ensure the sign is clear, visible, and well-lit.',
            'processing_time': elapsed_time,
            'annotated_image': resized_img,
            'primary_roi': None,
            'all_detections': [],
            'intermediate': {
                'gray': pipeline['gray'],
                'blurred': pipeline['blurred_gray'],
                'enhanced': pipeline['enhanced_gray'],
                'edges': edges,
            }
        }
