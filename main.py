from pathlib import Path
import argparse
from selecter import select_best_images, ImageSelectionConfig
from resizer import batch_resize_images
from face_crop import process_folder

def create_pipeline_folders(base_path: Path) -> tuple[Path, Path, Path]:
    """Create and return paths for each stage of the pipeline"""
    selected = base_path / "1_selected"
    resized = base_path / "2_resized"
    face_cropped = base_path / "3_face_cropped"
    
    for path in (selected, resized, face_cropped):
        path.mkdir(exist_ok=True)
    
    return selected, resized, face_cropped

def process_images(
    input_dir: str | Path,
    target_size: int = 512,
    file_pattern: str = "*.jpg",
    prefix_separator: str = "_",
    jpeg_quality: int = 95,
    selected_folder: str = "selected",
    resized_folder: str = "resized",
    face_cropped_folder: str = "face_cropped"
) -> None:
    """Run the complete image processing pipeline"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Step 1: Select best images
    print("\n=== Step 1: Selecting best images ===")
    for pattern in file_pattern.split(","):
        pattern = pattern.strip()
        print(f"Processing pattern: {pattern}")
        config = ImageSelectionConfig(
            target_size=target_size,
            output_folder=selected_folder,
            file_pattern=pattern,
            prefix_separator=prefix_separator
        )
        select_best_images(str(input_path), config)
    
    # Count files in selected folder
    selected_path = input_path / selected_folder
    selected_files = list(selected_path.glob("*.*"))
    print(f"Selected {len(selected_files)} files")
    
    # Step 2: Resize images
    print("\n=== Step 2: Resizing images ===")
    batch_resize_images(
        selected_path,
        target_size=target_size,
        quality=jpeg_quality,
        extensions=tuple(p.replace("*", "") for p in file_pattern.split(",")),
        output_folder=resized_folder
    )
    
    # Count files in resized folder
    resized_path = input_path / resized_folder
    resized_files = list(resized_path.glob("*.*"))
    print(f"Resized {len(resized_files)} files")
    
    # Step 3: Face crop
    print("\n=== Step 3: Face cropping ===")
    process_folder(
        resized_path, 
        output_folder=face_cropped_folder,
        target_size=target_size
    )
    
    # Count files in face_cropped folder
    face_cropped_path = input_path / face_cropped_folder
    cropped_files = list(face_cropped_path.glob("*.*"))
    print(f"Face cropped {len(cropped_files)} files")

def main():
    parser = argparse.ArgumentParser(description='Process images through selection, resizing, and face cropping')
    parser.add_argument('input_dir', help='Input directory containing images')
    parser.add_argument('--target-size', type=int, default=512,
                       help='Target resolution (default: 512)')
    parser.add_argument('--file-pattern', default="*.jpg,*.jpeg,*.png",
                       help='File pattern(s) to match (comma-separated, default: *.jpg,*.jpeg,*.png)')
    parser.add_argument('--prefix-separator', default="_",
                       help='Character separating prefix from rest of filename (default: _)')
    parser.add_argument('--jpeg-quality', type=int, default=95,
                       help='JPEG quality for resized images (default: 95)')
    parser.add_argument('--selected-folder', default="selected",
                       help='Output folder for selected images (default: selected)')
    parser.add_argument('--resized-folder', default="resized",
                       help='Output folder for resized images (default: resized)')
    parser.add_argument('--face-cropped-folder', default="face_cropped",
                       help='Output folder for face-cropped images (default: face_cropped)')
    
    args = parser.parse_args()
    
    try:
        # List input files before processing
        input_path = Path(args.input_dir)
        input_files = []
        for pattern in args.file_pattern.split(","):
            pattern = pattern.strip()
            matched_files = list(input_path.glob(pattern))
            input_files.extend(matched_files)
            print(f"Found {len(matched_files)} files matching pattern {pattern}")
        
        if not input_files:
            print("No input files found! Please check your file patterns and input directory.")
            return
            
        process_images(
            args.input_dir,
            target_size=args.target_size,
            file_pattern=args.file_pattern,
            prefix_separator=args.prefix_separator,
            jpeg_quality=args.jpeg_quality,
            selected_folder=args.selected_folder,
            resized_folder=args.resized_folder,
            face_cropped_folder=args.face_cropped_folder
        )
        print("\nImage processing pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise

if __name__ == "__main__":
    main()
