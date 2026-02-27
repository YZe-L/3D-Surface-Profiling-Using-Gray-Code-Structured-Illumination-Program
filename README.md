# 3D Surface Profiling Using Gray-Code Structured Illumination: Program

A Python-based structured light 3D scanning system using Grey-code pattern projection and stereo triangulation. Developed as part of the Optics and Photonics MSc at Imperial College London.

## Overview

This system reconstructs 3D surface geometry by projecting a sequence of Grey-code binary patterns from a calibrated projector onto a target object, capturing the deformed patterns with a calibrated camera, and recovering depth through stereo triangulation. The pipeline includes Nayar's direct/global illumination separation for robust decoding under inter-reflections and specular highlights.

## System Requirements

### Hardware

- **Projector**: LCD or DLP projector (tested with Apeman LC350, 1920×1080)
- **Camera**: Machine vision camera (tested with Thorlabs CS165CU/M, 1440×1080)
- **Calibration target**: Printed checkerboard pattern

### Software

- Python 3.8+
- NumPy
- OpenCV (`opencv-python`)
- tqdm (optional, for progress bars)

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py                 # Main entry point: calibration and reconstruction CLI
├── structured_light.py     # Grey-code encoding/decoding and Nayar light separation
├── scan3d.py               # Stereo triangulation and point cloud generation
├── calibration_data.py     # Calibration data I/O (YAML and MATLAB formats)
└── README.md
```

### Module Descriptions

| Module | Functionality |
|--------|--------------|
| `main.py` | CLI interface, image loading, chessboard corner detection, stereo calibration orchestration, reconstruction pipeline |
| `structured_light.py` | Grey-to-binary conversion, pattern decoding from captured image pairs, direct/global light estimation (Nayar's algorithm), robust bit classification with saturation and dark-pixel rescue |
| `scan3d.py` | Batch ray–ray intersection (vectorised), stereo triangulation, point cloud storage, PLY export, surface normal computation, adaptive outlier filtering |
| `calibration_data.py` | `CalibrationData` class for camera/projector intrinsics, extrinsics, and distortion coefficients; save/load in OpenCV YAML and MATLAB `.m` formats |

## Usage

### 1. Calibration

Prepare calibration image sets: for each checkerboard pose, capture the full Grey-code projection sequence. Organise images into numbered subdirectories under a root folder:

```
calibration_data/
├── set_001/
│   ├── 000.png   # White reference
│   ├── 001.png   # Black reference
│   ├── 002.png   # Column bit-plane 0, normal
│   ├── 003.png   # Column bit-plane 0, inverted
│   └── ...       # Remaining bit-plane pairs
├── set_002/
│   └── ...
└── ...            # 5+ poses recommended
```

Run calibration:

```bash
python main.py calibrate ./calibration_data --corners 6,9 --size 24,24
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corners` | `6,9` | Inner corner count (cols, rows) |
| `--size` | `24,24` | Square size in mm (width, height) |
| `--threshold` | `25` | Shadow/contrast threshold |
| `--b` | `0.5` | Nayar reflectance parameter *b* ∈ [0, 1) |
| `--m` | `5` | Minimum direct light for valid classification |
| `--window` | `100` | Local homography window size (px) |
| `--no-parallel` | — | Disable multiprocessing |

Output: `calibration.yml` (OpenCV YAML) and `calibration.m` (MATLAB-compatible).

### 2. Reconstruction

Capture the Grey-code sequence on the target object (same projection order as calibration). Place all images in a single folder with a `projector_info.txt` containing the projector resolution:

```
1920 1080
```

Run reconstruction:

```bash
python main.py reconstruct ./scan_001 ./calibration_data/calibration.yml --threshold 25 --b 0.5
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--threshold` | `25` | Contrast threshold (higher = stricter filtering) |
| `--b` | `0.5` | Nayar reflectance parameter |
| `--m` | `5` | Minimum direct light threshold |
| `--max-dist` | `100.0` | Maximum ray–ray distance for valid triangulation (mm) |
| `--output`, `-o` | `pointcloud.ply` | Output PLY file path |

Output: PLY point cloud, viewable in [CloudCompare](https://www.cloudcompare.org/), MeshLab, or similar software.

## Algorithm Pipeline

```
Image capture (46 frames per scan)
        │
        ▼
  Nayar direct/global light separation ──► Ld, Lg per pixel
        │
        ▼
  Grey-code bit-plane decoding ──► (px, py) correspondence map
        │
        ▼
  Stereo calibration parameters (K, Kp, R, t, distortion)
        │
        ▼
  Undistortion + ray construction
        │
        ▼
  Batch ray–ray intersection ──► 3D point cloud (.ply)
```

### Key Implementation Details

- **Vectorised operations**: All decoding, triangulation, and filtering use NumPy batch operations rather than per-pixel loops, achieving significant speedup on large images.
- **Robust bit classification**: Extends Nayar's method with rescue mechanisms for saturated pixels (specular surfaces) and dark pixels (low-reflectance surfaces), improving coverage on challenging materials.
- **Adaptive outlier removal**: Neighbourhood-based filtering with locally adaptive thresholds removes spurious triangulation results while preserving fine geometric features.
- **Multiprocessing**: Calibration sets are processed in parallel across CPU cores.

## Calibration File Format

The `calibration.yml` file stores all parameters in OpenCV YAML format:

```yaml
cam_K:    # Camera intrinsic matrix (3×3)
cam_kc:   # Camera distortion [k1, k2, p1, p2, k3]
proj_K:   # Projector intrinsic matrix (3×3)
proj_kc:  # Projector distortion [k1, k2, p1, p2, k3]
R:        # Rotation matrix, projector relative to camera (3×3)
T:        # Translation vector (3×1, in mm)
```

## Acknowledgements

The core reconstruction algorithm is ported from the [Brown University scan3d-capture project](http://mesh.brown.edu/calibration/software.html), originally developed by Daniel Moreno and Gabriel Taubin. This implementation adds NumPy vectorisation, multiprocessing, Nayar-based robust decoding with saturation/dark-pixel rescue, and adaptive outlier filtering.

## References

1. G. Taubin, D. Moreno, and D. Lanman, "3D scanning for personal 3D printing: build your own desktop 3D scanner," *ACM SIGGRAPH 2014 Studio*, 2014.
2. D. Moreno and G. Taubin, "Simple, accurate, and robust projector-camera calibration," *3DIMPVT*, IEEE, 2012.
3. S. K. Nayar, G. Krishnan, M. D. Grossberg, and R. Raskar, "Fast separation of direct and global components of a scene using high frequency illumination," *ACM SIGGRAPH*, 2006.

## Licence

This project is released for academic and educational purposes.