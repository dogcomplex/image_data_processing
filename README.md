# Image Processing Pipeline

Batch image dataset preparation with two pipelines (default face-zoom and legacy) and several utility scripts for grouping and sorting. Outputs are cached per stage using short hashes of both the input contents and the current configuration for reproducibility.

## Features

- Smart selection by filename prefix groups (e.g., `prefix_001.jpg` → group `prefix`)
- Aspect-ratio preserving resize and face-centric crops/zooms
- OpenCV-based Haar cascade face detection with debug visuals
- Optional majority-face filtering via `face_recognition` + DBSCAN
- Multi-format support where applicable (JPG/JPEG/PNG; WEBP supported in resizer; GIF/WEBP for animated sorting)
- Stage outputs cached into sibling `processed/` using config/input hashes

## Installation

Windows 10/11 (CMD):

```bat
:: From your repo root
py -m venv .venv
.venv\Scripts\activate
pip install -U pip setuptools wheel
pip install pillow opencv-python numpy face-recognition scikit-learn scikit-image ImageHash

:: Download Haar cascade (required for face detection)
curl -L -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

WSL/Ubuntu (optional):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
sudo apt update && sudo apt install -y libgl1 libglib2.0-0
pip install pillow opencv-python numpy face-recognition scikit-learn scikit-image ImageHash
curl -L -o haarcascade_frontalface_default.xml https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

Notes:
- `face_recognition` is required by the `identified` stage and some utilities. On Windows, ensure a compatible Python version and C++ build tools if needed; otherwise consider WSL.
- The Haar cascade file is searched in common OpenCV locations; placing it next to the repo root also works.

## Usage (default pipeline: face zoom)

```bat
python main.py <input_directory>
```

Common options (defaults shown):

```bat
python main.py <input_directory> ^
  --target-size 512 ^
  --file-pattern "*.jpg,*.jpeg,*.png" ^
  --prefix-separator "_" ^
  --jpeg-quality 95 ^
  --zoom-factor 2.5
```

- `--legacy-pipeline` switches to the legacy (non-zoom) pipeline.
- To include WEBP during selection, add it to `--file-pattern` (downstream single-face/zoom currently process JPG/JPEG/PNG).

### What gets created

Outputs are written to a sibling `processed/` directory next to your input folder. Each stage folder is suffixed with an 8-char hash derived from the stage config and input contents to enable cache reuse and invalidation.

Default face-zoom pipeline stages:
- `1_selected_<hash>/`: Best image from each prefix group
- `2_single_face_<hash>/`: Subset with exactly one detected face
- `3_zoomed_<hash>/`: Face-zoomed crops
- `4_filtered_<hash>/`: Resolution filter (keeps images ≥ target_size/3 on both sides)
- `5_final_<hash>/`: Final resize to target resolution
- `6_identified_<hash>/`: Majority-face cluster filtered

## Legacy pipeline

Enable with `--legacy-pipeline`. Stages (also under `processed/`):
- `selected_<hash>/` → `resized_<hash>/` → `face_cropped_<hash>/` → `single_face_<hash>/` → `identified_<hash>/`

The legacy pipeline performs selection → resize → face-centered crop → optional single-face → majority-face filtering.

## Utilities and standalone scripts

All commands below default to Windows CMD. For WSL/Linux, drop the `^` line continuations and use `\` or newlines.

- Selection and resizing are embedded in the main pipeline. Additional helpers:

- Resizer (`resizer.py`):
  ```bat
  python resizer.py <input_folder>
  ```

- Face crop with debug (`face_crop.py`):
  ```bat
  python face_crop.py <input_folder>
  ```

- Single-face filter (`single_face.py`):
  ```bat
  python single_face.py <input_folder>
  ```

- Majority-face filter (`identify.py`):
  ```bat
  python identify.py <input_dir> ^
    --tolerance 0.6 ^
    --min-cluster 3
  ```

- Group visually similar images/videos (`group_similarity.py`):
  ```bat
  python group_similarity.py <input_dir> ^
    --output-dir <optional_out_dir> ^
    --hash-size 16 ^
    --threshold 10 ^
    --min-content-std 15.0 ^
    --threads 4
  ```

- Sort animated GIF/WEBP by amount of change (`sort_changed.py`):
  ```bat
  python sort_changed.py <target_dir> ^
    --output-dir <optional_out_dir> ^
    --threads 4 ^
    --scoring-method all
  ```

- Sort images by similarity to reference faces (`sort_similarity.py`):
  ```bat
  python sort_similarity.py <reference_dir> <target_dir> ^
    --output-dir <optional_out_dir> ^
    --threshold 0.0 ^
    --threads 4
  ```

## Configuration and caching

Stage folders are created via `pipeline_config.PipelineConfig` and `make_stage_folder`. A short hash is computed from:
- The config values (e.g., target size, zoom factor, quality)
- The stage name
- A content hash of the input folder (filenames + file sizes)

If the input contents change, downstream stages are invalidated; otherwise, prior stage outputs are reused.

## LoRA training workflow (optional)

1. Collect images (browser helpers etc.) into an input folder
2. Run the default pipeline with your `--target-size` (commonly 512 or 768)
3. Use `5_final_<hash>/` or `6_identified_<hash>/` as your curated set
4. Train with your preferred toolchain

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
