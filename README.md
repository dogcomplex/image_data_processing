# Image Processing Pipeline

A Python-based image processing pipeline that handles batch processing of images through three main stages:

1. **Image Selection**: Selects the best quality images from groups of similar files based on naming prefixes and target resolution
2. **Image Resizing**: Resizes selected images while maintaining aspect ratio
3. **Face-Centered Cropping**: Detects faces in images and crops around them to create consistent square outputs

## Features

- Groups images by filename prefixes for smart selection
- Maintains aspect ratios during resizing
- Uses OpenCV for face detection
- Creates debug visualizations of face detection
- Processes multiple image formats (JPG, PNG, JPEG)
- Configurable output quality and target sizes

## Installation

```bash
# Clone the repository
git clone git@github.com:dogcomplex/image_data_processing.git

# Install dependencies
pip install pillow opencv-python numpy

# download face detector for cropping
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
python main.py <input_directory> \\
    --target-size 512 \\
    --file-pattern "*.jpg,*.jpeg,*.png" \\
    --prefix-separator "_" \\
    --jpeg-quality 95
```

Using this for prepping for LoRA training:
```
process:
- go to webpage with images and view all, scroll down with autoscroller plugin till all on the same page
- DownThemAll plugin to scrape to folder 
- run some image selection (pruning alt sizes), resizing (to target of e.g. 768 or 512), then face detect and crop to square of each
- https://github.com/dogcomplex/image_data_processing (scripts just generated on the fly for all that by claude lol)
- (optional) pick some best subset manually
- Install ComfyUI node using manager.  git repo: https://github.com/edenartlab/sd-lora-trainer?tab=readme-ov-file
- open workflow: https://github.com/edenartlab/sd-lora-trainer/blob/main/ComfyUI_workflows/train_lora.json
- configure lora trainer according to docs (did defaults, checkpoint every 50, 800 steps total, disable ti, dreamshaper-8 SD1.5 model, entered lora name and input dir, set for face recognition)
- queue, let bake 25ish mins on 3090rtx
```


### Parameters

- `input_directory`: Folder containing source images
- `--target-size`: Target resolution (default: 512)
- `--file-pattern`: File patterns to match (default: "*.jpg,*.jpeg,*.png")
- `--prefix-separator`: Character separating prefix from filename (default: "_")
- `--jpeg-quality`: Output JPEG quality (default: 95)

## Pipeline Stages

1. **Selection** (`selecter.py`): Groups images by prefix and selects the best resolution match from each group
2. **Resizing** (`resizer.py`): Resizes images to target size while maintaining aspect ratio
3. **Face Cropping** (`face_crop.py`): Detects faces and crops images to center them

## Output Structure

The pipeline creates three folders:
- `1_selected/`: Best images selected from each group
- `2_resized/`: Resized versions of selected images
- `3_face_cropped/`: Final square crops centered on detected faces

## Debug Output

When face detection is running, debug images are saved to a `face_cropped_debug` folder showing:
- Detected face centers (green dots)
- Planned crop regions (green rectangles)

## License

MIT License

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
