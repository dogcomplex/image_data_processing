import cv2
import numpy as np
import os
import sys
from pathlib import Path

def detect_faces(image):
    """Detect faces in image and return average center point."""
    # Try multiple possible paths for the cascade file
    cascade_paths = [
        'haarcascade_frontalface_default.xml',  # Local directory
        os.path.join(sys.prefix, 'Library', 'etc', 'haarcascades', 'haarcascade_frontalface_default.xml'),  # Windows
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',  # Unix-like
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',  # Unix-like alternative
    ]
    
    face_cascade = None
    cascade_found = False
    for cascade_path in cascade_paths:
        if os.path.exists(cascade_path):
            print(f"Found cascade file at: {cascade_path}")  # Debug info
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if not face_cascade.empty():
                cascade_found = True
                break
    
    if not cascade_found:
        print("Warning: Could not load face cascade classifier. Using image center.")
        return (image.shape[1] // 2, image.shape[0] // 2)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("No faces detected in image. Using image center.")
        return (image.shape[1] // 2, image.shape[0] // 2)
    
    # Calculate average center point of all detected faces
    centers = []
    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        centers.append((center_x, center_y))
        print(f"Found face at: ({center_x}, {center_y})")  # Debug info
    
    avg_x = int(sum(x for x, _ in centers) / len(centers))
    avg_y = int(sum(y for _, y in centers) / len(centers))
    
    print(f"Using center point: ({avg_x}, {avg_y})")  # Debug info
    return (avg_x, avg_y)

def crop_around_point(image, center_x, center_y, target_size):
    """Crop a square region around the given center point."""
    height, width = image.shape[:2]
    
    # Calculate crop dimensions
    half_size = target_size // 2
    left = max(0, center_x - half_size)
    right = min(width, center_x + half_size)
    top = max(0, center_y - half_size)
    bottom = min(height, center_y + half_size)
    
    # Adjust if crop would go out of bounds
    if right - left < target_size:
        if left == 0:
            right = target_size
        else:
            left = width - target_size
    if bottom - top < target_size:
        if top == 0:
            bottom = target_size
        else:
            top = height - target_size
    
    return image[top:bottom, left:right]

def process_folder(input_folder, output_folder="face_cropped", target_size=512, debug=True):
    """Process all images in the input folder."""
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    if debug:
        debug_path = input_path.parent / f"{output_folder}_debug"
        debug_path.mkdir(exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        print(f"\nProcessing {img_path.name}")
        
        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to read {img_path.name}")
            continue
        
        # Make a copy for debugging
        debug_image = image.copy()
        
        # Detect face and get center point
        center_x, center_y = detect_faces(image)
        
        if debug:
            # Draw detection results with dynamic target size
            cv2.circle(debug_image, (center_x, center_y), 10, (0, 255, 0), -1)
            cv2.rectangle(debug_image, 
                         (center_x - target_size//2, center_y - target_size//2),
                         (center_x + target_size//2, center_y + target_size//2),
                         (0, 255, 0), 2)
            debug_file = debug_path / f"debug_{img_path.name}"
            cv2.imwrite(str(debug_file), debug_image)
        
        # Crop image with dynamic target size
        cropped = crop_around_point(image, center_x, center_y, target_size)
        
        # Ensure final size matches target size
        final = cv2.resize(cropped, (target_size, target_size))
        
        # Save result
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), final)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python face_crop.py <input_folder>")
        sys.exit(1)
    
    process_folder(sys.argv[1])
