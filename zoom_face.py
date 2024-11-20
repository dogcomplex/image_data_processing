import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def detect_face_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect a single face in image and return its bounding box (x, y, w, h)."""
    cascade_paths = [
        'haarcascade_frontalface_default.xml',
        str(Path(cv2.__file__).parent / 'data' / 'haarcascade_frontalface_default.xml'),
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]
    
    face_cascade = None
    for path in cascade_paths:
        if Path(path).exists():
            face_cascade = cv2.CascadeClassifier(path)
            if not face_cascade.empty():
                break
    
    if face_cascade is None or face_cascade.empty():
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces[0] if len(faces) == 1 else None

def zoom_and_crop_face(
    image_path: Path,
    output_path: Path,
    zoom_factor: float = 1.5
) -> Tuple[bool, Tuple[int, int]]:
    """
    Zoom in on detected face and crop. Returns (success, (width, height))
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return False, (0, 0)
    
    face_rect = detect_face_region(image)
    if face_rect is None:
        return False, (0, 0)
    
    x, y, w, h = face_rect
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate zoom region
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    
    # Calculate crop region
    left = max(0, center_x - new_w // 2)
    top = max(0, center_y - new_h // 2)
    right = min(image.shape[1], left + new_w)
    bottom = min(image.shape[0], top + new_h)
    
    # Adjust if crop would go out of bounds
    if right - left < new_w:
        left = max(0, image.shape[1] - new_w)
    if bottom - top < new_h:
        top = max(0, image.shape[0] - new_h)
    
    cropped = image[top:bottom, left:right]
    cv2.imwrite(str(output_path), cropped)
    
    return True, (right - left, bottom - top)

def process_folder(
    input_folder: str | Path,
    output_folder: str = "zoomed_faces",
    zoom_factor: float = 1.5
) -> None:
    """Process all images in the input folder."""
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    processed = 0
    successful = 0
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        processed += 1
        print(f"\rProcessing image {processed}: {img_path.name}", end="")
        
        success, _ = zoom_and_crop_face(
            img_path,
            output_path / img_path.name,
            zoom_factor
        )
        if success:
            successful += 1
    
    print(f"\nSuccessfully processed {successful} out of {processed} images")
