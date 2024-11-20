import cv2
import numpy as np
from pathlib import Path
import shutil

def detect_face_count(image_path: Path) -> int:
    """Return the number of faces detected in an image."""
    # Try multiple possible cascade file locations
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
                print(f"Loaded cascade from: {path}")
                break
    
    if face_cascade is None or face_cascade.empty():
        print("Warning: Could not load face cascade classifier")
        return 0
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return 0
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Debug: Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Save debug image
    debug_path = image_path.parent / "debug_faces"
    debug_path.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_path / image_path.name), image)
    
    return len(faces)

def filter_single_face_images(input_folder: str | Path, output_folder: str = "single_face") -> None:
    """Copy images containing exactly one face to the output folder."""
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    processed = 0
    single_face_count = 0
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        processed += 1
        print(f"\rProcessing image {processed}: {img_path.name}", end="")
        
        face_count = detect_face_count(img_path)
        if face_count == 1:
            single_face_count += 1
            shutil.copy2(img_path, output_path / img_path.name)
    
    print(f"\nFound {single_face_count} images with single faces out of {processed} processed images")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python single_face.py <input_folder>")
        sys.exit(1)
    
    filter_single_face_images(sys.argv[1])
