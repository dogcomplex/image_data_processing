from pathlib import Path
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import shutil

@dataclass
class ImageSelectionConfig:
    """Configuration settings for image selection"""
    target_size: int = 512
    output_folder: str = "selections"
    file_pattern: str = "*.jpg"
    prefix_separator: str = "_"

def get_prefix(filename: str, separator: str = "_") -> str:
    """Extract prefix from filename before first separator"""
    return filename.split(separator)[0]

def get_resolution_score(
    width: int, 
    height: int, 
    target: int
) -> Tuple[bool, int]:
    """
    Calculate how close dimensions are to target resolution
    Returns (is_larger_than_target, distance_from_target)
    """
    is_larger = width >= target and height >= target
    distance = abs(width - target) + abs(height - target)
    return (is_larger, distance)

def group_images_by_prefix(
    input_path: Path, 
    config: ImageSelectionConfig
) -> Dict[str, List[Path]]:
    """Group images by their prefix"""
    image_groups = defaultdict(list)
    for img_path in input_path.glob(config.file_pattern):
        prefix = get_prefix(img_path.name, config.prefix_separator)
        image_groups[prefix].append(img_path)
    return dict(image_groups)

def select_best_images(
    input_dir: str | Path, 
    output_path: Path,
    config: ImageSelectionConfig = ImageSelectionConfig()
) -> None:
    """
    Select and copy the best matching images from each prefix group
    """
    input_path = Path(input_dir)
    output_path.mkdir(exist_ok=True)
    
    image_groups = group_images_by_prefix(input_path, config)
    
    # Process each group
    for prefix, images in image_groups.items():
        best_score = (False, float('inf'))
        best_image = None
        
        for img_path in images:
            with Image.open(img_path) as img:
                width, height = img.size
                score = get_resolution_score(width, height, config.target_size)
                
                if (score[0] and not best_score[0]) or \
                   (score[0] == best_score[0] and score[1] < best_score[1]):
                    best_score = score
                    best_image = img_path
        
        if best_image:
            shutil.copy2(best_image, output_path / best_image.name)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Select best matching images from groups')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--target-size', type=int, default=512,
                       help='Target resolution (default: 512)')
    parser.add_argument('--output-folder', default="selections",
                       help='Output folder name (default: selections)')
    parser.add_argument('--file-pattern', default="*.jpg",
                       help='File pattern to match (default: *.jpg)')
    parser.add_argument('--prefix-separator', default="_",
                       help='Character separating prefix from rest of filename (default: _)')
    
    args = parser.parse_args()
    
    config = ImageSelectionConfig(
        target_size=args.target_size,
        output_folder=args.output_folder,
        file_pattern=args.file_pattern,
        prefix_separator=args.prefix_separator
    )
    
    select_best_images(args.input_dir, config)

if __name__ == "__main__":
    main()
