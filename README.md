# Human Preprocessing Kit

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)

## Overview

The **Human Preprocessing Kit** is a comprehensive video-to-human data pipeline designed to extract high-fidelity 2D poses, instance segmentation masks, and 3D poses from input videos. It integrates several state-of-the-art methods for human-centric computer vision tasks, enabling researchers and developers to preprocess video data for applications in animation, AR/VR, human activity recognition, and more.

This toolkit supports modular components for flexibility:
- **Instance Segmentation**: Generates per-instance masks for human bodies using advanced segmentation models.
- **2D Pose Estimation**: Detects 2D keypoints for body, hands, and face landmarks.
- **3D Pose Estimation**: Reconstructs temporally consistent 3D poses and meshes, capturing body shape and motion dynamics.

The pipeline processes videos frame-by-frame or in batches, outputting structured data (e.g., JSON for keypoints, PNG for masks, OBJ for meshes) that can be easily integrated into downstream workflows.

**Key Features**:
- Support for multiple SOTA methods with easy switching via configuration.
- Temporal consistency for smooth video-based reconstructions.
- Extensible architecture for adding custom models.
- Tested on Ubuntu 20.04+ with Python 3.8+ and PyTorch 1.10+.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Components](#components)
  - [Instance Segmentation](#instance-segmentation)
  - [2D Pose Estimation](#2d-pose-estimation)
  - [3D Pose Estimation](#3d-pose-estimation)
- [Testing](#testing)
- [Outputs](#outputs)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [References](#references)

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Human-Preprocessing-Kit.git
   cd Human-Preprocessing-Kit
   ```

2. Install dependencies (see [Installation](#installation) for details):
   ```bash
   pip install -r requirements.txt
   ```

3. Run a test on a sample video:
   ```bash
   python tests/preprocess_video.py --input_video sample.mp4 --method mediapipe --output_dir results/
   ```

4. Visualize outputs:
   ```bash
   python scripts/visualize.py --input_json results/poses.json --video sample.mp4
   ```

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10+ with CUDA (for GPU acceleration; CPU-only mode supported)
- FFmpeg for video handling: `sudo apt install ffmpeg`
- Git for cloning dependencies

Install the core package in editable mode:
```bash
pip install -e .
```

Component-specific installations are detailed in their respective sections below. Ensure you have sufficient disk space (~5GB) for pre-trained models.

### Environment Setup
Create a virtual environment:
```bash
python -m venv hpk_env
source hpk_env/bin/activate  # On Linux/Mac
# pip install -r requirements.txt
```

## Configuration

Configurations are managed via YAML files in the `configs/` directory. Key files:
- `default.yaml`: Global settings (e.g., device, batch size).
- Method-specific: e.g., `vibe.yaml` for VIBE parameters.

Example snippet for `vibe.yaml`:
```yaml
model:
  checkpoint: 'path/to/vibe_checkpoint.pth'
data:
  video_path: 'input.mp4'
output:
  keypoints: true
  mesh: true
```

Load configs in scripts: `from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/vibe.yaml')`.

## Usage

The toolkit provides a unified CLI interface for preprocessing:
```bash
python scripts/preprocess.py --input_video <path> --output_dir <dir> --component <seg|2d|3d> --method <method_name> [--cfg <config.yaml>]
```

- `--component`: `seg` (segmentation), `2d` (2D pose), `3d` (3D pose).
- `--method`: Specific model (e.g., `maskrcnn`, `mediapipe`, `vibe`).
- Outputs are saved in `<output_dir>/<method>/` with timestamps.

For batch processing:
```bash
python scripts/batch_process.py --video_list videos.txt --method vibe
```

Custom integration:
```python
from hpk.pipeline import Pipeline
pipe = Pipeline(component='3d', method='mediapipe')
results = pipe.process_video('input.mp4')
```

## Components

### Instance Segmentation

This module extracts pixel-level masks for human instances in video frames, supporting multi-person scenarios.

#### Supported Methods
1. **Mask R-CNN**: Instance segmentation with bounding box detection. Backbone: ResNet-50-FPN.
2. **DeepLabv3+**: Semantic segmentation refined for instances. Backbone: ResNet-101.
3. **Segment Anything (SAM)**: Zero-shot segmentation prompted by bounding boxes or points.

#### Installation
Install Detectron2 (required for Mask R-CNN and DeepLabv3+):
```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
For SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

#### Usage
```bash
python scripts/segment.py --video input.mp4 --method maskrcnn --output_dir masks/
```

### 2D Pose Estimation

This module detects 2D keypoints for body, hands, and facial landmarks, serving as input for 3D lifting.

#### Supported Methods
- **MediaPipe**: Holistic 2D/3D estimation with body (33 keypoints), hands (42 total), and face (468 keypoints). Total: 543 keypoints per person.
- **OpenPose**: Multi-person 2D detection using BODY_25 model: body (25), hands (42), face (70). Total: 137 keypoints per person.

#### Installation
For MediaPipe:
```bash
pip install mediapipe
```

For OpenPose (detailed below):
Follow the steps in the [OpenPose subsection](#openpose).

#### Usage
```bash
python scripts/pose2d.py --video input.mp4 --method openpose --output_dir poses2d/
```

### 3D Pose Estimation

This module lifts 2D detections to 3D poses and meshes, emphasizing temporal consistency for video inputs.

#### Supported Methods

##### MediaPipe
MediaPipe provides real-time, on-device 3D estimation:
- Body: 33 keypoints
- Hands (left/right): 42 keypoints (21 each)
- Face: 468 keypoints
- **Total Output**: 543 3D keypoints per person

**Installation**: Included in base requirements.

##### VIBE (Video Inference for Body Pose and Shape Estimation)
VIBE excels in video tasks by modeling temporal dependencies, robust to occlusions and fast motions.
- Body Pose: 3D positions of 49 keypoints; rotations for 23 joints
- Hands (left/right): 42 keypoints (21 each)
- Shape: SMPL-based 3D mesh with 6890 vertices
- **Total Output**: 91 keypoints + 6890 vertices per person

**Installation**:
- Install STAF dependency: Clone from [STAF GitHub](https://github.com/soulslicer/STAF) and follow its setup.
- Download VIBE checkpoint: Place in `models/vibe/`.

##### TCMR (Temporally Consistent Mesh Recovery)
TCMR builds on VIBE for enhanced temporal smoothness in 3D pose and shape from monocular videos (CVPR 2021).
- Body Pose: 3D positions of 49 keypoints; rotations for 23 joints (SMPL model)
- Shape: 10 shape parameters for body variation
- Mesh: Full-body 3D mesh with 6890 vertices
- **Total Output**: 49 keypoints + 6890 vertices per person (body-focused; hands via extension)

**Installation**:
- Run setup script: `source scripts/install_pip.sh` (requires sudo).
- Download SMPL models (male/female/neutral) from [SMPL website](https://smpl.is.tue.mpg.de) and place in `data/base_data/`.
- Download pre-trained weights: `source scripts/get_pretrained.sh`.

##### OpenPose
See [2D Pose Estimation](#2d-pose-estimation) for 2D details. For 3D, use `--3d` flag to enable calibration-based lifting (outputs 25 3D body keypoints).

###### Installation Steps
1. Update system and install dependencies:
   ```bash
   sudo apt update
   sudo apt install -y build-essential cmake libopencv-dev libopenblas-dev \
       libopenmpi-dev libomp-dev libboost-all-dev libgflags-dev libgoogle-glog-dev \
       libhdf5-serial-dev protobuf-compiler
   ```

2. Install CUDA and cuDNN for GPU (optional): Follow [this guide](https://www.enablegeek.com/blog/setting-up-cuda-and-cudnn-on-ubuntu-20-04/).

3. Clone OpenPose:
   ```bash
   git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
   cd openpose
   ```

4. Download models:
   ```bash
   cd models/
   ./getModels.sh
   cd ..
   ```
   Alternative: Download from [Google Drive](https://drive.google.com/file/d/1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh).

5. Build with CMake (CPU-only example; adjust for GPU):
   ```bash
   mkdir build && cd build
   cmake -DBUILD_PYTHON=ON -DBUILD_3D=ON -DGPU_MODE=CPU_ONLY ..
   make -j$(nproc)
   ```

6. Add to environment:
   ```bash
   echo 'export PYTHONPATH=$PYTHONPATH:/path/to/openpose/build/python' >> ~/.bashrc
   source ~/.bashrc
   ```

#### Usage (3D-specific)
```bash
python scripts/pose3d.py --video input.mp4 --method vibe --cfg configs/vibe.yaml
```

## Testing

Run unit tests for each component:
```bash
# Instance Segmentation
python -m pytest tests/segmentation/test_maskrcnn.py
python -m pytest tests/segmentation/test_deeplabv3.py
python -m pytest tests/segmentation/test_sam.py

# 2D/3D Pose
python -m pytest tests/pose_estimation/test_mediapipe.py
python -m pytest tests/pose_estimation/test_vibe.py
python -m pytest tests/pose_estimation/test_tcmr.py
python -m pytest tests/pose_estimation/test_openpose.py
```

Integration test:
```bash
python -m pytest tests/integration/test_pipeline.py
```

## Outputs

- **Segmentation**: Per-frame masks as PNG (instance ID encoded).
- **2D Poses**: JSON with keypoints {frame: {person_id: [x,y,conf]}} .
- **3D Poses**: JSON with 3D keypoints/meshes; OBJ files for visualization.
- All outputs include metadata (e.g., timestamps, method version).

## Limitations

- Multi-person handling varies by method (e.g., OpenPose excels here).
- GPU recommended for real-time; CPU fallback is slower.
- No built-in tracking; pair with trackers like DeepSORT for persistent IDs.
- SMPL-based methods assume neutral clothing; performance drops on loose attire.

## Contributing

We welcome contributions! Please:
1. Fork the repo and create a feature branch.
2. Add tests for new features.
3. Submit a PR with detailed description.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on open-source libraries: Detectron2, MediaPipe, OpenPose, VIBE, TCMR.
- Thanks to the original authors for their foundational work.

## References

- Mask R-CNN: He et al., ICCV 2017.
- VIBE: Kocabas et al., CVPR 2020.
- TCMR: Choi et al., CVPR 2021.
- OpenPose: Cao et al., TPAMI 2019.
- MediaPipe: Lugaresi et al., arXiv 2019.

For issues or questions, open a GitHub issue.