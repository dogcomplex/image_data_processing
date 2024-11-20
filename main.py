from pathlib import Path
import argparse
from selecter import select_best_images, ImageSelectionConfig
from resizer import batch_resize_images
from face_crop import process_folder
from single_face import filter_single_face_images
from zoom_face import process_folder as zoom_process
from filter import filter_by_resolution
from identify import process_folder as identify_process
from pipeline_config import PipelineConfig, make_stage_folder

def process_images(
    input_dir: str | Path,
    config: PipelineConfig = PipelineConfig()
) -> None:
    """Run the complete image processing pipeline"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create stage folders with config hashes
    selected_path = make_stage_folder(input_path, "selected", config, input_path)
    resized_path = make_stage_folder(input_path, "resized", config, selected_path)
    face_cropped_path = make_stage_folder(input_path, "face_cropped", config, resized_path)
    single_face_path = make_stage_folder(input_path, "single_face", config, face_cropped_path)
    identified_path = make_stage_folder(input_path, "identified", config, single_face_path)
    
    # Track which stages need processing
    needs_processing = {
        "selected": not any(selected_path.iterdir()),
        "resized": not any(resized_path.iterdir()),
        "face_cropped": not any(face_cropped_path.iterdir()),
        "single_face": not any(single_face_path.iterdir()),
        "identified": not any(identified_path.iterdir())
    }
    
    # Process stages in order, tracking dependencies
    if needs_processing["selected"]:
        print("\n=== Step 1: Selecting best images ===")
        for pattern in config.file_pattern.split(","):
            pattern = pattern.strip()
            print(f"Processing pattern: {pattern}")
            selection_config = ImageSelectionConfig(
                target_size=config.target_size,
                file_pattern=pattern,
                prefix_separator=config.prefix_separator
            )
            select_best_images(input_path, selected_path, selection_config)
        # Invalidate all downstream stages
        needs_processing.update({k: True for k in ["resized", "face_cropped", "single_face", "identified"]})
    
    if needs_processing["resized"]:
        print("\n=== Step 2: Resizing images ===")
        batch_resize_images(
            selected_path,
            target_size=config.target_size,
            quality=config.jpeg_quality,
            output_path=resized_path
        )
        # Invalidate downstream stages
        needs_processing.update({k: True for k in ["face_cropped", "single_face", "identified"]})
    
    if needs_processing["face_cropped"]:
        print("\n=== Step 3: Face cropping ===")
        process_folder(
            resized_path, 
            output_path=face_cropped_path,
            target_size=config.target_size
        )
        # Invalidate downstream stages
        needs_processing.update({k: True for k in ["single_face", "identified"]})
    
    if needs_processing["single_face"]:
        print("\n=== Step 4: Filtering single face images ===")
        filter_single_face_images(
            face_cropped_path,
            output_path=single_face_path
        )
        # Invalidate downstream stage
        needs_processing["identified"] = True
    
    if needs_processing["identified"]:
        print("\n=== Step 5: Filtering for matching faces ===")
        identify_process(
            single_face_path,
            output_path=identified_path,
            tolerance=config.face_tolerance,
            min_cluster_size=config.min_cluster_size
        )
    
    # Print final statistics
    print("\nPipeline completed!")
    print(f"Selected: {len(list(selected_path.glob('*.*')))} images")
    print(f"Resized: {len(list(resized_path.glob('*.*')))} images")
    print(f"Face cropped: {len(list(face_cropped_path.glob('*.*')))} images")
    print(f"Single face: {len(list(single_face_path.glob('*.*')))} images")
    print(f"Identified: {len(list(identified_path.glob('*.*')))} images")

def process_images_zoom_face(
    input_dir: str | Path,
    config: PipelineConfig = PipelineConfig()
) -> None:
    """Run the face-zooming image processing pipeline"""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create stage folders sequentially, waiting for each stage
    selected_path = make_stage_folder(input_path, "1_selected", config, input_path)
    if not any(selected_path.iterdir()):
        print("\n=== Step 1: Selecting best resolution images ===")
        for pattern in config.file_pattern.split(","):
            pattern = pattern.strip()
            selection_config = ImageSelectionConfig(
                target_size=config.target_size,
                file_pattern=pattern,
                prefix_separator=config.prefix_separator
            )
            select_best_images(input_path, selected_path, selection_config)
    
    single_face_path = make_stage_folder(input_path, "2_single_face", config, selected_path)
    if not any(single_face_path.iterdir()):
        print("\n=== Step 2: Filtering single face images ===")
        filter_single_face_images(selected_path, single_face_path)
    
    zoomed_path = make_stage_folder(input_path, "3_zoomed", config, single_face_path)
    if not any(zoomed_path.iterdir()):
        print("\n=== Step 3: Zooming into faces ===")
        zoom_process(single_face_path, zoomed_path, zoom_factor=config.zoom_factor)
    
    filtered_path = make_stage_folder(input_path, "4_filtered", config, zoomed_path)
    if not any(filtered_path.iterdir()):
        print("\n=== Step 4: Filtering low resolution images ===")
        filter_by_resolution(zoomed_path, filtered_path, config.target_size // 3)
    
    final_path = make_stage_folder(input_path, "5_final", config, filtered_path)
    if not any(final_path.iterdir()):
        print("\n=== Step 5: Final resize and face crop ===")
        batch_resize_images(
            filtered_path,
            target_size=config.target_size,
            quality=config.jpeg_quality,
            output_path=final_path
        )
    
    identified_path = make_stage_folder(input_path, "6_identified", config, final_path)
    if not any(identified_path.iterdir()):
        print("\n=== Step 6: Filtering for matching faces ===")
        identify_process(
            final_path,
            output_path=identified_path,
            tolerance=config.face_tolerance,
            min_cluster_size=config.min_cluster_size
        )
    
    # Print final statistics
    print("\nPipeline completed!")
    print(f"Selected: {len(list(selected_path.glob('*.*')))} images")
    print(f"Single face: {len(list(single_face_path.glob('*.*')))} images")
    print(f"Zoomed: {len(list(zoomed_path.glob('*.*')))} images")
    print(f"Filtered: {len(list(filtered_path.glob('*.*')))} images")
    print(f"Resized: {len(list(final_path.glob('*.*')))} images")
    print(f"Identified: {len(list(identified_path.glob('*.*')))} images")

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
    parser.add_argument('--legacy-pipeline', action='store_true',
                       help='Use legacy pipeline instead of face-zooming pipeline')
    parser.add_argument('--zoom-factor', type=float, default=2.5,
                       help='Face zoom factor (default: 2.5)')
    parser.add_argument('--identified-folder', default="identified",
                       help='Output folder for face-matched images (default: identified)')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = PipelineConfig(
        target_size=args.target_size,
        file_pattern=args.file_pattern,
        prefix_separator=args.prefix_separator,
        jpeg_quality=args.jpeg_quality,
        zoom_factor=args.zoom_factor,
        face_tolerance=0.6,
        min_cluster_size=3
    )
    
    try:
        if args.legacy_pipeline:
            process_images(args.input_dir, config)
        else:
            process_images_zoom_face(args.input_dir, config)
            
        print("\nImage processing pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        raise

if __name__ == "__main__":
    main()
