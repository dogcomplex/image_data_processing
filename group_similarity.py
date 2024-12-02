from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import shutil
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass

@dataclass
class GroupingConfig:
    """Configuration for similarity grouping"""
    hash_size: int = 16  # Size of the perceptual hash
    threshold: int = 10  # Maximum hash difference to consider similar
    min_content_std: float = 15.0  # Minimum standard deviation to filter trivial frames
    max_workers: int = 4  # Number of parallel workers

def compute_image_hash(image_path: Path) -> Tuple[Path, imagehash.ImageHash | None]:
    """Compute perceptual hash for an image, returning None if image is trivial"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check if image has enough variation (not trivial)
            np_img = np.array(img)
            if np.std(np_img) < GroupingConfig.min_content_std:
                logging.warning(f"Skipping trivial image: {image_path.name}")
                return image_path, None
            
            # Compute perceptual hash
            return image_path, imagehash.average_hash(img, hash_size=GroupingConfig.hash_size)
            
    except Exception as e:
        logging.error(f"Failed to process {image_path.name}: {e}")
        return image_path, None

def extract_video_frames(
    video_path: Path,
    sample_interval: int = 30
) -> List[Tuple[str, np.ndarray]]:
    """Extract frames from video at given interval"""
    frames = []
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_name = f"{video_path.stem}_frame_{frame_count:06d}"
                frames.append((frame_name, frame_rgb))
                
            frame_count += 1
            
        cap.release()
        
    except Exception as e:
        logging.error(f"Failed to process video {video_path}: {e}")
        
    return frames

def find_similar_groups(
    hashes: Dict[Path, imagehash.ImageHash],
    config: GroupingConfig
) -> List[Set[Path]]:
    """Group similar images based on hash differences"""
    groups: List[Set[Path]] = []
    processed = set()
    
    for path1, hash1 in hashes.items():
        if path1 in processed:
            continue
            
        # Start new group
        current_group = {path1}
        processed.add(path1)
        
        # Find similar images
        for path2, hash2 in hashes.items():
            if path2 not in processed and hash1 - hash2 <= config.threshold:
                current_group.add(path2)
                processed.add(path2)
        
        if len(current_group) > 1:  # Only keep groups with multiple items
            groups.append(current_group)
    
    return groups

def group_similar_images(
    input_dir: str | Path,
    output_dir: str | Path = None,
    config: GroupingConfig = GroupingConfig()
) -> None:
    """Group similar images/videos into subfolders"""
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path.parent / "grouped_similarity"
    output_path.mkdir(exist_ok=True)
    
    # Supported file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    video_extensions = {'.mp4', '.webm', '.mkv'}
    
    # Process all files
    file_hashes: Dict[Path, imagehash.ImageHash] = {}
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Process images
        image_futures = []
        for ext in image_extensions:
            for img_path in input_path.glob(f"*{ext}"):
                image_futures.append(executor.submit(compute_image_hash, img_path))
        
        # Process videos
        video_frames = []
        for ext in video_extensions:
            for video_path in input_path.glob(f"*{ext}"):
                frames = extract_video_frames(video_path)
                video_frames.extend((video_path, frame_name, frame) 
                                 for frame_name, frame in frames)
        
        # Convert video frames to hashes
        for video_path, frame_name, frame in video_frames:
            pil_image = Image.fromarray(frame)
            hash_value = imagehash.average_hash(pil_image, hash_size=config.hash_size)
            if np.std(frame) >= config.min_content_std:
                file_hashes[video_path] = hash_value
        
        # Collect image results
        for future in image_futures:
            path, hash_value = future.result()
            if hash_value is not None:
                file_hashes[path] = hash_value
    
    # Find groups of similar files
    groups = find_similar_groups(file_hashes, config)
    
    # Copy files to group folders
    for i, group in enumerate(groups, 1):
        group_dir = output_path / f"group_{i:03d}"
        group_dir.mkdir(exist_ok=True)
        
        for file_path in group:
            try:
                shutil.copy2(file_path, group_dir / file_path.name)
            except Exception as e:
                logging.error(f"Failed to copy {file_path}: {e}")
    
    logging.info(f"Created {len(groups)} groups of similar files")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Group similar images and videos')
    parser.add_argument('input_dir', help='Input directory containing images/videos')
    parser.add_argument('--output-dir', help='Output directory (default: grouped_similarity)')
    parser.add_argument('--hash-size', type=int, default=16,
                       help='Size of perceptual hash (default: 16)')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Maximum hash difference threshold (default: 10)')
    parser.add_argument('--min-content-std', type=float, default=15.0,
                       help='Minimum standard deviation to filter trivial frames (default: 15.0)')
    parser.add_argument('--threads', type=int, default=4,
                       help='Number of processing threads (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(message)s')
    
    # Create config
    config = GroupingConfig(
        hash_size=args.hash_size,
        threshold=args.threshold,
        min_content_std=args.min_content_std,
        max_workers=args.threads
    )
    
    try:
        group_similar_images(args.input_dir, args.output_dir, config)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
