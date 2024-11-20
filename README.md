# Image Processing Pipeline

A Python-based image processing pipeline for batch processing images through six stages:

1. **Image Selection**: Selects best quality images from groups of similar files
2. **Single Face Filter**: Filters for images containing exactly one face
3. **Face Zooming**: Zooms in on detected faces with configurable zoom factor
4. **Resolution Filter**: Filters out low-resolution results
5. **Final Resize**: Resizes images to target dimensions
6. **Face Matching**: Groups similar faces together (optional)

## Features

- Smart image grouping by filename prefixes
- Maintains aspect ratios during resizing
- OpenCV-based face detection and centering
- Debug visualizations for face detection
- Multi-format support (JPG, PNG, JPEG, WEBP)
- Configurable output quality and target sizes

## Installation

```bash
# Clone the repository
git clone git@github.com:dogcomplex/image_data_processing.git

# Install dependencies
pip install pillow opencv-python numpy

# Download face detector
cd image_data_processing
curl -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## Usage

Basic usage:
```bash
python main.py <input_directory>
```

Advanced usage with options:
```bash
python main.py <input_directory> \
    --target-size 512 \
    --file-pattern "*.jpg,*.jpeg,*.png,*.webp" \
    --prefix-separator "_" \
    --jpeg-quality 95 \
    --zoom-factor 2.5 \
    --legacy-pipeline  # Use old pipeline without face zooming
```

### Parameters

- `input_directory`: Source images folder
- `--target-size`: Target resolution (default: 512)
- `--file-pattern`: File patterns to match (default: "*.jpg,*.jpeg,*.png")
- `--prefix-separator`: Character separating prefix from filename (default: "_")
- `--jpeg-quality`: Output JPEG quality (default: 95)
- `--zoom-factor`: Face zoom multiplier (default: 2.5)
- `--legacy-pipeline`: Use original pipeline without face zooming

## Pipeline Stages

### Default Pipeline (with face zooming)
1. **Selection**: Groups similar images by prefix and selects best resolution match
2. **Single Face Filter**: Filters for images with exactly one face
3. **Face Zooming**: Zooms in on detected faces
4. **Resolution Filter**: Removes low-quality results
5. **Final Resize**: Resizes to target dimensions
6. **Face Matching**: Groups similar faces together (optional)

### Legacy Pipeline
Available via --legacy-pipeline flag
1. **Selection**: Groups and selects best images
2. **Resizing**: Basic image resizing
3. **Face Cropping**: Center-crop around faces
4. **Single Face Filter**: Optional single-face filtering
5. **Face Matching**: Groups similar faces together (optional)

## Output Structure

The default pipeline creates these folders:
- `1_selected/`: Best images from each group
- `2_single_face/`: Images with exactly one face
- `3_zoomed/`: Face-zoomed versions
- `4_filtered/`: Resolution-filtered subset
- `5_final/`: Final resized images
- `6_identified/`: Optional face-matched groups

## LoRA Training Integration

For preparing images for LoRA training:

1. Collect source images:
   - Use browser plugins like AutoScroller and DownThemAll
   - Save to input folder

2. Process images:
   - Run pipeline for selection, resizing, and face cropping
   - Optionally filter to single-face subset
   - Target size typically 512 or 768

3. Train LoRA:
   - Install ComfyUI node from [sd-lora-trainer](https://github.com/edenartlab/sd-lora-trainer)
   - Use workflow: [train_lora.json](https://github.com/edenartlab/sd-lora-trainer/blob/main/ComfyUI_workflows/train_lora.json)
   - Configure training parameters (e.g., 800 steps, checkpoint every 50)

Copyright (c) 2024 Warren Koch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
