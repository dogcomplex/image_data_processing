from pathlib import Path
import argparse
from selecter import select_best_images, ImageSelectionConfig
from resizer import batch_resize_images
from face_crop import process_folder
from single_face import filter_single_face_images
from zoom_face import process_folder as zoom_process
from filter import filter_by_resolution
from identify import process_folder as identify_process

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
    face_cropped_folder: str = "face_cropped",
    single_face_folder: str = "single_face",
    identified_folder: str = "identified"
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
    
    # Step 4: Filter single face images
    print("\n=== Step 4: Filtering single face images ===")
    filter_single_face_images(
        face_cropped_path,
        output_folder=single_face_folder
    )
    
    # Count files in single_face folder
    single_face_path = input_path / single_face_folder
    single_face_files = list(single_face_path.glob("*.*"))
    print(f"Found {len(single_face_files)} images with single faces")
    
    # Step 5: Filter for matching faces
    print("\n=== Step 5: Filtering for matching faces ===")
    identify_process(
        single_face_path,
        output_folder=identified_folder
    )
    
    # Count files in identified folder
    identified_path = input_path / identified_folder
    identified_files = list(identified_path.glob("*.*"))
    print(f"Kept {len(identified_files)} images with matching faces")


def process_images_zoom_face(
    input_dir: str | Path,
    target_size: int = 512,
    file_pattern: str = "*.jpg",
    prefix_separator: str = "_",
    jpeg_quality: int = 95,
    zoom_factor: float = 1.5,
    selected_folder: str = "1_selected",
    single_face_folder: str = "2_single_face",
    zoomed_folder: str = "3_zoomed",
    filtered_folder: str = "4_filtered",
    final_folder: str = "5_final",
    identified_folder: str = "6_identified"
) -> None:
    """Run the face-zooming image processing pipeline"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Step 1: Select best resolution images
    print("\n=== Step 1: Selecting best resolution images ===")
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
    
    selected_path = input_path / selected_folder
    
    # Step 2: Filter for single face images
    print("\n=== Step 2: Filtering single face images ===")
    filter_single_face_images(
        selected_path,
        output_folder=single_face_folder
    )
    
    single_face_path = input_path / single_face_folder
    
    # Step 3: Zoom into faces
    print("\n=== Step 3: Zooming into faces ===")
    zoom_process(
        single_face_path,
        output_folder=zoomed_folder,
        zoom_factor=zoom_factor
    )
    
    zoomed_path = input_path / zoomed_folder
    
    # Step 4: Filter out low resolution results
    print("\n=== Step 4: Filtering low resolution images ===")
    filter_by_resolution(
        zoomed_path,
        min_size=target_size // 3,
        output_folder=filtered_folder
    )
    
    filtered_path = input_path / filtered_folder
    
    # Step 5: Final resize and crop
    print("\n=== Step 5: Final resize and face crop ===")
    batch_resize_images(
        filtered_path,
        target_size=target_size,
        quality=jpeg_quality,
        output_folder=final_folder
    )
    
    resized_path = input_path / final_folder
    process_folder(
        resized_path,
        output_folder="final",
        target_size=target_size
    )
    
    # Step 6: Filter for matching faces
    print("\n=== Step 6: Filtering for matching faces ===")
    final_path = input_path / "final"
    identify_process(
        final_path,
        output_folder=identified_folder
    )
    
    # Count files in identified folder
    identified_path = input_path / identified_folder
    identified_files = list(identified_path.glob("*.*"))
    print(f"Kept {len(identified_files)} images with matching faces")

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
    parser.add_argument('--single-face-folder', default="single_face",
                       help='Output folder for single-face images (default: single_face)')
    parser.add_argument('--zoom-face', action='store_true',
                       help='Use face-zooming pipeline instead of default')
    parser.add_argument('--zoom-factor', type=float, default=2.5,
                       help='Face zoom factor (default: 2.5)')
    parser.add_argument('--identified-folder', default="identified",
                       help='Output folder for face-matched images (default: identified)')
    
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

        if args.zoom_face:
            process_images_zoom_face(
                args.input_dir,
                target_size=args.target_size,
                file_pattern=args.file_pattern,
                prefix_separator=args.prefix_separator,
                jpeg_quality=args.jpeg_quality,
                zoom_factor=args.zoom_factor,
                identified_folder=args.identified_folder
            )
        else:
            process_images(
                args.input_dir,
                target_size=args.target_size,
                file_pattern=args.file_pattern,
                prefix_separator=args.prefix_separator,
                jpeg_quality=args.jpeg_quality,
                selected_folder=args.selected_folder,
                resized_folder=args.resized_folder,
                face_cropped_folder=args.face_cropped_folder,
                single_face_folder=args.single_face_folder,
                identified_folder=args.identified_folder
            )
        print("\nImage processing pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise

if __name__ == "__main__":
    main()
