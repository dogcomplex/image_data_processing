from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging
import shutil
from typing import Tuple, Dict, Literal
from PIL import Image
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim
import imagehash

ScoreMethod = Literal['mad', 'mse', 'ssim', 'phash', 'all']

@dataclass
class ScoringConfig:
    """Configuration for video change scoring"""
    method: ScoreMethod = 'all'
    mad_scale: float = 50.0  # Scale factor for Mean Absolute Difference
    mse_scale: float = 1000.0  # Scale factor for Mean Squared Error
    ssim_threshold: float = 0.05  # Minimum SSIM difference to count as change
    phash_threshold: float = 5  # Maximum perceptual hash difference for similar frames
    drastic_change_penalty: float = 0.5  # Penalty for scene cuts/drastic changes

def compute_frame_scores(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Dict[str, float]:
    """
    Compute various frame difference scores between two frames.
    Returns dictionary of scoring method -> normalized score (0-1).
    """
    # Convert frames to grayscale if needed
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame

    scores = {}
    
    # 1. Mean Absolute Difference (sensitive to small changes)
    mad = np.mean(np.abs(curr_gray.astype(float) - prev_gray.astype(float)))
    scores['mad'] = min(1.0, mad / 50.0)
    
    # 2. Mean Squared Error (more sensitive to larger changes)
    mse = np.mean((curr_gray.astype(float) - prev_gray.astype(float)) ** 2)
    scores['mse'] = min(1.0, mse / 1000.0)
    
    # 3. Structural Similarity (focuses on structural changes)
    similarity = ssim(prev_gray, curr_gray)
    scores['ssim'] = 1.0 - similarity  # Convert to difference score
    
    # 4. Perceptual Hash (good for detecting scene changes)
    prev_img = Image.fromarray(prev_gray)
    curr_img = Image.fromarray(curr_gray)
    prev_hash = imagehash.average_hash(prev_img)
    curr_hash = imagehash.average_hash(curr_img)
    hash_diff = prev_hash - curr_hash
    scores['phash'] = min(1.0, hash_diff / 32.0)  # Normalize to 0-1
    
    return scores

def compute_frame_changes_webp(
    video_path: Path,
    config: ScoringConfig
) -> Tuple[Path, float]:
    """
    Compute frame changes in WebP using multiple scoring methods.
    """
    try:
        with Image.open(video_path) as img:
            if not getattr(img, 'is_animated', False):
                logging.warning(f"{video_path} is not animated")
                return video_path, 0.0

            # Initialize score accumulators
            total_scores = {'mad': 0.0, 'mse': 0.0, 'ssim': 0.0, 'phash': 0.0}
            drastic_changes = 0
            frame_count = 1

            # Get first frame
            img.seek(0)
            prev_frame = np.array(img.convert('RGB'))

            # Process subsequent frames
            try:
                while True:
                    img.seek(img.tell() + 1)
                    curr_frame = np.array(img.convert('RGB'))
                    
                    # Compute all scores for this frame pair
                    scores = compute_frame_scores(prev_frame, curr_frame)
                    
                    # Track drastic changes (potential scene cuts)
                    if scores['phash'] > 0.5:  # High perceptual hash difference
                        drastic_changes += 1
                    
                    # Accumulate scores
                    for method in total_scores:
                        total_scores[method] += scores[method]
                    
                    prev_frame = curr_frame
                    frame_count += 1
            except EOFError:
                pass

            if frame_count < 2:
                return video_path, 0.0

            # Calculate final scores
            avg_scores = {method: score / (frame_count - 1) 
                         for method, score in total_scores.items()}
            
            # Apply drastic change penalty
            drastic_change_ratio = drastic_changes / (frame_count - 1)
            penalty = 1.0 - (drastic_change_ratio * config.drastic_change_penalty)

            # Compute final score based on method
            if config.method == 'mad':
                final_score = avg_scores['mad']
            elif config.method == 'mse':
                final_score = avg_scores['mse']
            elif config.method == 'ssim':
                final_score = avg_scores['ssim']
            elif config.method == 'phash':
                final_score = avg_scores['phash']
            else:  # 'all' - weighted combination
                final_score = (
                    0.3 * avg_scores['mad'] +
                    0.2 * avg_scores['mse'] +
                    0.3 * avg_scores['ssim'] +
                    0.2 * avg_scores['phash']
                )

            # Apply penalty
            final_score *= penalty
            
            logging.info(f"{video_path.name}: {frame_count} frames, "
                        f"scores={avg_scores}, penalty={penalty:.2f}, "
                        f"final={final_score:.3f}")
            
            return video_path, final_score

    except Exception as e:
        logging.error(f"Error processing {video_path}: {e}")
        return video_path, 0.0

def sort_by_changes(
    target_dir: str | Path,
    output_dir: str | Path = None,
    num_threads: int = 4,
    scoring_method: ScoreMethod = 'all'
) -> None:
    """
    Sort videos by amount of frame-to-frame changes.
    
    Args:
        target_dir: Directory containing videos to analyze
        output_dir: Output directory (default: ./processed/sorted_changed)
        num_threads: Number of parallel processing threads
        scoring_method: Scoring method to use ('mad', 'mse', 'ssim', 'phash', or 'all')
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
                futures.append(executor.submit(compute_frame_changes_webp, video_path, ScoringConfig(method=scoring_method)))

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
    parser.add_argument('--scoring-method', type=str, default='all', help='Scoring method to use')
    
    args = parser.parse_args()
    
    # Set logging level based on verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(message)s')
    
    try:
        sort_by_changes(args.target_dir, args.output_dir, args.threads, args.scoring_method)
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main()
