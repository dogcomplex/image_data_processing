from pathlib import Path
import face_recognition
import numpy as np
from typing import Dict, List, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging

def load_reference_encodings(reference_dir: Path) -> List[np.ndarray]:
    """Load face encodings from reference directory."""
    encodings = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for ext in image_extensions:
        for img_path in reference_dir.glob(f"*{ext}"):
            try:
                logging.info(f"Processing reference image: {img_path.name}")
                image = face_recognition.load_image_file(str(img_path))
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    logging.info(f"Successfully encoded face from {img_path.name}")
                else:
                    logging.warning(f"No face found in {img_path.name}")
            except Exception as e:
                logging.warning(f"Failed to process reference image {img_path}: {e}")
    
    logging.info(f"Loaded {len(encodings)} reference face encodings")
    return encodings

def compute_similarity(
    image_path: Path,
    reference_encodings: List[np.ndarray]
) -> Tuple[Path, float]:
    """Compute maximum similarity score for an image against reference encodings."""
    try:
        image = face_recognition.load_image_file(str(image_path))
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            logging.warning(f"No faces found in {image_path.name}")
            return image_path, 0.0
            
        # For each face in the image, compare against all reference encodings
        best_similarity = 0.0
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(reference_encodings, face_encoding)
            similarity = 1 - min(distances)  # Convert distance to similarity
            best_similarity = max(best_similarity, similarity)
        
        return image_path, best_similarity
        
    except Exception as e:
        logging.warning(f"Failed to process target image {image_path}: {e}")
        return image_path, 0.0

def process_target_images(
    target_dir: Path,
    reference_encodings: List[np.ndarray],
    output_dir: Path,
    num_threads: int = 4
) -> None:
    """Process all images in target directory using multiple threads."""
    output_dir.mkdir(exist_ok=True)
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # Collect all image paths
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(target_dir.glob(f"*{ext}"))
    
    logging.info(f"Found {len(image_paths)} target images to process")
    
    if not image_paths:
        logging.warning(f"No images found in {target_dir}")
        return
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(compute_similarity, img_path, reference_encodings)
            for img_path in image_paths
        ]
        
        # Collect results
        similarities = [future.result() for future in futures]
    
    # Copy files with similarity score prefix
    processed_count = 0
    for img_path, similarity in similarities:
        score = int(similarity * 100)
        new_name = f"{score:03d}_{img_path.name}"
        shutil.copy2(img_path, output_dir / new_name)
        processed_count += 1
        if processed_count % 10 == 0:
            logging.info(f"Processed {processed_count}/{len(similarities)} images")
    
    logging.info(f"Successfully processed and copied {processed_count} images")

def sort_by_similarity(
    reference_dir: str | Path,
    target_dir: str | Path,
    output_dir: str | Path = None,
    num_threads: int = 4
) -> None:
    """Main function to sort images by similarity to reference faces."""
    reference_path = Path(reference_dir)
    target_path = Path(target_dir)
    output_path = Path(output_dir) if output_dir else target_path / "sorted_similarity"
    
    if not all(p.exists() for p in [reference_path, target_path]):
        raise ValueError("Reference and target directories must exist")
    
    logging.info("Loading reference face encodings...")
    reference_encodings = load_reference_encodings(reference_path)
    
    if not reference_encodings:
        raise ValueError("No valid reference faces found")
    
    logging.info(f"Processing target images into {output_path}...")
    process_target_images(target_path, reference_encodings, output_path, num_threads)
    logging.info("Processing complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort images by face similarity')
    parser.add_argument('reference_dir', help='Directory containing reference images')
    parser.add_argument('target_dir', help='Directory containing images to sort')
    parser.add_argument('--output-dir', help='Directory for sorted output (default: sorted_similarity in target dir)')
    parser.add_argument('--threads', type=int, default=4, help='Number of processing threads')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    try:
        sort_by_similarity(args.reference_dir, args.target_dir, args.output_dir, args.threads)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise
