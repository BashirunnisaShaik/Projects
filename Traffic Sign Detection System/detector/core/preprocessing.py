"""
Image Preprocessing Module for Traffic Sign Detection System.
Provides modular image enhancement, safe decoding from bytes and Windows paths,
color space transformation, noise reduction, and morphological operations.
"""

import cv2
import numpy as np
import os
import io


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Safely decode raw image bytes into a valid 3-channel BGR OpenCV NumPy array.
    Supports JPG, JPEG, and PNG images.
    Uses numpy.frombuffer() and cv2.imdecode(), with PIL fallback
    for non-standard JPEG/PNG encodings (e.g., CMYK JPEGs, EXIF orientation).
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("The uploaded image file is empty (0 bytes). Please upload a valid image.")

    img = None

    # Step 1: Decode using OpenCV and numpy.frombuffer
    try:
        np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
    except Exception:
        img = None

    # Step 2: Handle color channels if OpenCV successfully decoded
    if img is not None:
        if len(img.shape) == 2:
            # Grayscale -> 3-channel BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                # 4-channel BGRA (PNG with transparency) -> 3-channel BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:
                # 1-channel -> 3-channel BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                # Standard 3-channel BGR
                pass
            else:
                img = None
        else:
            img = None

    # Step 3: Fallback using PIL (Pillow) if OpenCV returned None
    # (Handles CMYK JPEGs, unusual EXIF metadata, progressive JPEGs, etc.)
    if img is None:
        try:
            from PIL import Image, ImageOps
            pil_image = Image.open(io.BytesIO(file_bytes))

            # Auto-orient using EXIF orientation tag if present
            try:
                pil_image = ImageOps.exif_transpose(pil_image)
            except Exception:
                pass

            # Convert to standard RGB mode
            pil_image = pil_image.convert('RGB')
            rgb_arr = np.array(pil_image)
            if rgb_arr is not None and rgb_arr.size > 0:
                img = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        except Exception:
            img = None

    # Step 4: Validate decoded image
    if img is None or img.size == 0 or len(img.shape) < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(
            "Unable to decode the uploaded image. Please ensure the file is a valid, uncorrupted JPG, JPEG, or PNG image."
        )

    return img


def load_image(image_source) -> np.ndarray:
    """
    Safely load and decode an image from:
    - Raw bytes
    - File path string or Path object (handles Windows paths with spaces or Unicode)
    - File-like objects (Django FieldFile, UploadedFile, BytesIO)
    """
    file_bytes = None

    # Case 1: Raw bytes
    if isinstance(image_source, (bytes, bytearray)):
        file_bytes = bytes(image_source)

    # Case 2: File-like object (Django FieldFile, UploadedFile, BytesIO)
    elif hasattr(image_source, 'read'):
        try:
            if hasattr(image_source, 'seek'):
                image_source.seek(0)
            file_bytes = image_source.read()
            if hasattr(image_source, 'seek'):
                image_source.seek(0)
        except Exception:
            file_bytes = None

    # Case 3: File path string or Path object
    if file_bytes is None and isinstance(image_source, (str, os.PathLike)):
        file_path = os.fspath(image_source)
        file_path = os.path.normpath(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file does not exist at: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
        except Exception as err:
            raise IOError(f"Could not read image file from disk: {err}")

    if not file_bytes:
        raise ValueError("Could not read image data from the provided source. Please re-upload your image.")

    return load_image_from_bytes(file_bytes)


def encode_image_to_bytes(img: np.ndarray, extension: str = '.jpg') -> bytes:
    """
    Encode an OpenCV BGR image into bytes for saving into Django ContentFile.
    """
    if img is None or img.size == 0:
        return b''
    if not extension.startswith('.'):
        extension = '.' + extension
    success, encoded_img = cv2.imencode(extension, img)
    if success and encoded_img is not None:
        return encoded_img.tobytes()
    return b''


def save_image(img: np.ndarray, output_path: str = None, extension: str = '.jpg'):
    """
    Safely save an image to disk or return encoded image bytes.
    - If output_path is provided: writes to disk and returns bool (True on success).
    - If output_path is None: encodes image with the specified extension and returns bytes.
    """
    if img is None or img.size == 0:
        return False if output_path else b''

    if output_path is not None:
        norm_path = os.path.normpath(output_path)
        parent = os.path.dirname(norm_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        ext = os.path.splitext(norm_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            ext = '.jpg'

        success, encoded_img = cv2.imencode(ext, img)
        if success and encoded_img is not None:
            with open(norm_path, 'wb') as f:
                f.write(encoded_img.tobytes())
            return True
        return False

    return encode_image_to_bytes(img, extension=extension)


def resize_image(image: np.ndarray, max_width: int = 1200, max_height: int = 900) -> np.ndarray:
    """
    Resize image to a maximum dimension while strictly maintaining aspect ratio.
    If image is already smaller than max bounds, it remains unchanged.
    """
    h, w = image.shape[:2]
    if w <= max_width and h <= max_height:
        return image

    scale = min(max_width / float(w), max_height / float(h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def convert_color_spaces(image: np.ndarray) -> dict:
    """
    Convert BGR image to common computer vision color formats:
    - RGB: for web display & PIL
    - Grayscale: for edge detection & contours
    - HSV: for robust color segmentation (Hue, Saturation, Value)
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        'bgr': image,
        'rgb': rgb,
        'gray': gray,
        'hsv': hsv
    }


def reduce_noise(image: np.ndarray, method: str = 'gaussian', ksize: int = 5) -> np.ndarray:
    """
    Apply noise reduction to smooth road texture and artifacts.
    Methods:
    - 'gaussian': Standard Gaussian blur (default)
    - 'bilateral': Preserves sharp edges while smoothing flat surfaces
    - 'median': Effective against salt-and-pepper noise
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == 'median':
        k = ksize if ksize % 2 == 1 else ksize + 1
        return cv2.medianBlur(image, k)
    else:
        k = (ksize, ksize) if isinstance(ksize, int) else ksize
        if k[0] % 2 == 0:
            k = (k[0] + 1, k[1] + 1)
        return cv2.GaussianBlur(image, k, 0)


def enhance_contrast(gray_image: np.ndarray) -> np.ndarray:
    """
    Enhance local contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Especially helpful for traffic signs in shadows or overcast lighting.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def apply_morphology(mask: np.ndarray, op: str = 'close', kernel_size: int = 5) -> np.ndarray:
    """
    Apply mathematical morphology to binary masks:
    - 'close': Closes holes inside sign regions
    - 'open': Removes small stray noise specks
    - 'dilate': Expands sign boundary slightly
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if op == 'close':
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif op == 'open':
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif op == 'dilate':
        return cv2.dilate(mask, kernel, iterations=1)
    elif op == 'erode':
        return cv2.erode(mask, kernel, iterations=1)
    return mask


def preprocess_pipeline(image: np.ndarray) -> dict:
    """
    Full preprocessing pipeline returning intermediate stages for visualization.
    """
    # 1. Resize if required
    resized = resize_image(image)

    # 2. Convert color spaces
    colors = convert_color_spaces(resized)

    # 3. Blur for noise reduction
    blurred_gray = reduce_noise(colors['gray'], method='gaussian', ksize=5)
    blurred_bgr = reduce_noise(resized, method='gaussian', ksize=5)

    # 4. Enhance contrast
    enhanced_gray = enhance_contrast(blurred_gray)

    # 5. Canny edge detection
    edges = cv2.Canny(enhanced_gray, threshold1=50, threshold2=150)

    return {
        'resized': resized,
        'gray': colors['gray'],
        'hsv': colors['hsv'],
        'blurred_gray': blurred_gray,
        'blurred_bgr': blurred_bgr,
        'enhanced_gray': enhanced_gray,
        'edges': edges,
    }
