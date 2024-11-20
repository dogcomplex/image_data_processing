import face_recognition
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from collections import Counter
import shutil
from typing import List, Tuple
import cv2

def load_face_encodings(image_path: Path) -> List[np.ndarray]:
    """Load image and return face encodings if any faces are found."""
    # Load image
    image = face_recognition.load_image_file(str(image_path))
    
    # Find face locations
    face_locations = face_recognition.face_locations(image)
    
    # Get face encodings
    return face_recognition.face_encodings(image, face_locations)

def process_folder(
    input_folder: str | Path,
    output_folder: str = "identified",
    tolerance: float = 0.6,
    min_cluster_size: int = 3
) -> None:
    """
    Process a folder of images and copy images containing the majority face cluster
    to the output folder.
    
    Args:
        input_folder: Path to input folder containing images
        output_folder: Name of output folder for filtered images
        tolerance: DBSCAN distance tolerance (lower = stricter matching)
        min_cluster_size: Minimum cluster size to consider as valid
    """
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    # Collect face encodings and file paths
    encodings = []
    file_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    print("Loading and encoding faces...")
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        print(f"\rProcessing: {img_path.name}", end="")
        
        # Get face encodings for this image
        face_encodings = load_face_encodings(img_path)
        
        # If exactly one face found, add to our dataset
        if len(face_encodings) == 1:
            encodings.append(face_encodings[0])
            file_paths.append(img_path)
    
    if not encodings:
        print("\nNo valid face encodings found!")
        return
    
    print(f"\nFound {len(encodings)} valid images with single faces")
    
    # Cluster faces using DBSCAN
    clustering = DBSCAN(
        eps=tolerance,
        min_samples=min_cluster_size,
        metric="euclidean"
    ).fit(encodings)
    
    # Find the largest cluster (excluding noise labeled as -1)
    labels = clustering.labels_
    valid_labels = labels[labels >= 0]
    
    if len(valid_labels) == 0:
        print("No clusters found! Try adjusting tolerance or min_cluster_size")
        return
    
    # Get the most common label (majority cluster)
    majority_label = Counter(valid_labels).most_common(1)[0][0]
    
    # Copy images from majority cluster
    copied = 0
    for label, file_path in zip(labels, file_paths):
        if label == majority_label:
            shutil.copy2(file_path, output_path / file_path.name)
            copied += 1
    
    print(f"\nCopied {copied} images from majority face cluster")
    print(f"Removed {len(file_paths) - copied} outlier images")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter images to keep only the majority face')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--tolerance', type=float, default=0.6,
                       help='Face matching tolerance (lower = stricter, default: 0.6)')
    parser.add_argument('--min-cluster', type=int, default=3,
                       help='Minimum cluster size (default: 3)')
    
    args = parser.parse_args()
    
    process_folder(
        args.input_dir,
        tolerance=args.tolerance,
        min_cluster_size=args.min_cluster
    )
