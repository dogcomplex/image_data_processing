from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil
from typing import Tuple
from PIL import Image
import io

def compute_frame_changes_webp(video_path: Path) -> Tuple[Path, float]:
    """
    Compute average frame-to-frame changes in WebP animation using PIL.
    Returns tuple of (path, change_score) where change_score is 0-1.
    """
    try:
        with Image.open(video_path) as img:
            if not getattr(img, 'is_animated', False):
                logging.warning(f"{video_path} is not animated")
                return video_path, 0.0

            # Get first frame
            img.seek(0)
            prev_frame = np.array(img.convert('L'))
            total_diff = 0.0
            frame_count = 1

            # Process subsequent frames
            try:
                while True:
                    img.seek(img.tell() + 1)
                    frame = np.array(img.convert('L'))
                    
                    # Compute mean absolute difference between frames
                    diff = np.mean(np.abs(frame.astype(float) - prev_frame.astype(float)))
                    total_diff += diff
                    
                    prev_frame = frame
                    frame_count += 1
            except EOFError:
                pass  # End of frames

            if frame_count < 2:
                return video_path, 0.0

            # Normalize score to 0-1 range (typical diffs are 0-255)
            avg_diff = total_diff / (frame_count - 1)
            normalized_score = min(1.0, avg_diff / 50.0)  # Scale factor of 50 chosen empirically
            
            logging.info(f"{video_path.name}: {frame_count} frames, score {normalized_score:.3f}")
            return video_path, normalized_score

    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return video_path, 0.0

def compute_frame_changes_gif(video_path: Path) -> Tuple[Path, float]:
    """
    Compute average frame-to-frame changes in GIF using PIL.
    Returns tuple of (path, change_score) where change_score is 0-1.
    """
    try:
        with Image.open(video_path) as img:
            if not getattr(img, 'is_animated', False):
                logging.warning(f"{video_path} is not animated")
                return video_path, 0.0

            frames = []
            try:
                while True:
                    frames.append(np.array(img.convert('L')))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if len(frames) < 2:
                return video_path, 0.0

            # Compute differences between consecutive frames
            total_diff = 0.0
            for i in range(len(frames) - 1):
                diff = np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float)))
                total_diff += diff

            # Normalize score
            avg_diff = total_diff / (len(frames) - 1)
            normalized_score = min(1.0, avg_diff / 50.0)

            logging.info(f"{video_path.name}: {len(frames)} frames, score {normalized_score:.3f}")
            return video_path, normalized_score

    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return video_path, 0.0

def compute_frame_changes(video_path: Path) -> Tuple[Path, float]:
    """
    Compute average frame-to-frame changes based on file type.
    Returns tuple of (path, change_score) where change_score is 0-1.
    """
    if video_path.suffix.lower() == '.webp':
        return compute_frame_changes_webp(video_path)
    elif video_path.suffix.lower() == '.gif':
        return compute_frame_changes_gif(video_path)
    else:
        logging.error(f"Unsupported file format: {video_path}")
        return video_path, 0.0

def sort_by_changes(
    target_dir: str | Path,
    output_dir: str | Path = None,
    num_threads: int = 4
) -> None:
    """
    Sort videos by amount of frame-to-frame changes.
    
    Args:
        target_dir: Directory containing videos to analyze
        output_dir: Output directory (default: ./processed/sorted_changed)
        num_threads: Number of parallel processing threads
    """
    target_path = Path(target_dir)
    if not target_path.exists():
        raise ValueError(f"Target directory does not exist: {target_dir}")

    # Set default output to ./processed/sorted_changed relative to target dir
    output_path = Path(output_dir) if output_dir else target_path.parent / "processed" / "sorted_changed"
    output_path.parent.mkdir(exist_ok=True)
    output_path.mkdir(exist_ok=True)

    # Supported video formats
    video_extensions = {'.webp', '.gif'}
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for ext in video_extensions:
            for video_path in target_path.glob(f"*{ext}"):
                futures.append(executor.submit(compute_frame_changes, video_path))

        # Process results and copy files
        processed_count = 0
        for future in futures:
            try:
                video_path, score = future.result()
                
                # Create new filename with score prefix
                score_prefix = f"{int(score * 100000):06d}"
                new_name = f"{score_prefix}_{video_path.name}"
                new_path = output_path / new_name
                
                # Copy file with new name
                shutil.copy2(video_path, new_path)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logging.info(f"Processed {processed_count} videos")
                    
            except Exception as e:
                logging.error(f"Failed to process result: {e}")

    logging.info(f"Successfully processed {processed_count} videos")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sort videos by amount of frame changes')
    parser.add_argument('target_dir', help='Directory containing videos to analyze')
    parser.add_argument('--output-dir', help='Directory for sorted output (default: sorted_changed)')
    parser.add_argument('--threads', type=int, default=4, help='Number of processing threads')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(message)s')
    
    try:
        sort_by_changes(args.target_dir, args.output_dir, args.threads)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
