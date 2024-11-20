from pathlib import Path
import cv2
import shutil

def filter_by_resolution(
    input_folder: str | Path,
    min_size: int,
    output_folder: str = "filtered"
) -> None:
    """
    Copy images that meet minimum resolution requirements to output folder.
    """
    input_path = Path(input_folder)
    output_path = input_path.parent / output_folder
    output_path.mkdir(exist_ok=True)
    
    image_extensions = {'.jpg', '.jpeg', '.png'}
    processed = 0
    passed = 0
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        processed += 1
        print(f"\rChecking image {processed}: {img_path.name}", end="")
        
        # Read image dimensions
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        height, width = image.shape[:2]
        
        # Check if image meets minimum size requirement
        if width >= min_size and height >= min_size:
            passed += 1
            shutil.copy2(img_path, output_path / img_path.name)
    
    print(f"\nKept {passed} images out of {processed} that met minimum size of {min_size}px")
