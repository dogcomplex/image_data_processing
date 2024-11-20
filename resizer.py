from pathlib import Path
from PIL import Image
import os

def create_output_directory(input_path: Path) -> Path:
    """Create and return path to output directory next to input directory."""
    output_path = input_path.parent / "resized"
    output_path.mkdir(exist_ok=True)
    return output_path

def calculate_new_dimensions(width: int, height: int, target_size: int) -> tuple[int, int]:
    """Calculate new dimensions maintaining aspect ratio."""
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return new_width, new_height

def resize_image(
    image_path: Path, 
    output_path: Path, 
    target_size: int = 512,
    quality: int = 95
) -> None:
    """Resize single image maintaining aspect ratio."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            new_width, new_height = calculate_new_dimensions(width, height, target_size)
            
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            output_file = output_path / image_path.name
            resized_img.save(output_file, quality=quality)
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def batch_resize_images(
    input_folder: str | Path,
    output_path: Path,
    target_size: int = 512,
    quality: int = 95,
    extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')
) -> None:
    """Resize all images in a folder"""
    input_path = Path(input_folder)
    output_path.mkdir(exist_ok=True)
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in extensions:
            resize_image(image_file, output_path, target_size, quality)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python resizer.py <input_folder>")
        sys.exit(1)
        
    input_folder = sys.argv[1]
    batch_resize_images(
        input_folder,
        target_size=512,  # Minimum dimension size
        quality=95,       # JPEG quality (ignored for PNG)
        extensions=('.jpg', '.jpeg', '.png', '.webp')
    )
