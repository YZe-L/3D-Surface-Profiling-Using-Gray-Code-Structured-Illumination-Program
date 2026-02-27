"""
main.py - Structured Light 3D Scanner Main Program

Ported from Brown University scan3d-capture project
Original authors: Daniel Moreno and Gabriel Taubin

Optimized with:
- Multiprocessing for parallel calibration set processing
- Vectorized homography computation
- Progress tracking

Usage:
    # Calibration mode:
    python main.py calibrate <root_dir> [options]

    # Reconstruction mode (requires calibration file):
    python main.py reconstruct <image_folder> <calibration_file> [options]

Examples:
    python main.py calibrate ./calibration_data --corners 11,7 --size 21,21
    python main.py reconstruct ./scan_001 ./calibration.yml --threshold 25 --b 0.5

Required packages:
    pip install numpy opencv-python

Optional packages (for better performance):
    pip install tqdm  # Progress bars
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

from calibration_data import CalibrationData
import structured_light as sl
import scan3d

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Default parameters
DEFAULT_CORNER_X = 6 # Column number of coners
DEFAULT_CORNER_Y = 9 # Row number of coners
DEFAULT_CORNER_WIDTH = 24 # Coner width in mm
DEFAULT_CORNER_HEIGHT = 24 # Coner height in mm
DEFAULT_THRESHOLD = 25 # Threshold
DEFAULT_B = 0.5 # b
DEFAULT_M = 5 # m
DEFAULT_HOMOGRAPHY_WINDOW = 100 # homography window
DEFAULT_MAX_DIST = 100.0 # max distance in mm
DEFAULT_PROJECTOR_WIDTH = 1920 # projector resolution width in pixel
DEFAULT_PROJECTOR_HEIGHT = 1080 # projector resolution height in pixel

# Number of parallel workers
NUM_WORKERS = max(1, mp.cpu_count() - 1)


def get_image_files(folder: str) -> List[str]:
    """Get sorted list of image files in folder"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))

    return sorted(list(set(files)))


def get_projector_size(folder: str) -> Tuple[int, int]:
    """Read projector size from projector_info.txt"""
    info_file = os.path.join(folder, "projector_info.txt")
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r') as f:
                parts = f.read().strip().split()
                if len(parts) >= 2:
                    width = int(parts[0])
                    height = int(parts[1])
                    print(f"Projector info loaded: {width}x{height}")
                    return (width, height)
        except:
            pass

    print(f"Using default projector size: {DEFAULT_PROJECTOR_WIDTH}x{DEFAULT_PROJECTOR_HEIGHT}")
    return (DEFAULT_PROJECTOR_WIDTH, DEFAULT_PROJECTOR_HEIGHT)


def get_chessboard_world_coords(corner_count: Tuple[int, int],
                                corner_size: Tuple[float, float]) -> np.ndarray:
    """Generate world coordinates for chessboard corners (vectorized)"""
    cols, rows = corner_count
    w_coords = np.tile(np.arange(cols), rows) * corner_size[0]
    h_coords = np.repeat(np.arange(rows), cols) * corner_size[1]
    return np.column_stack([w_coords, h_coords, np.zeros(cols * rows)]).astype(np.float32)


def extract_chessboard_corners(image_file: str, corner_count: Tuple[int, int],
                               image_scale: int = 1) -> Optional[np.ndarray]:
    """Extract chessboard corners from image"""
    gray_image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        return None

    # Scale down for faster detection
    if image_scale > 1:
        small_img = cv2.resize(gray_image,
                               (gray_image.shape[1] // image_scale,
                                gray_image.shape[0] // image_scale))
    else:
        small_img = gray_image

    # Find corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small_img, corner_count, flags)

    if found:
        # Scale back
        corners = corners * image_scale

        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

        return corners.reshape(-1, 2)

    return None


def decode_gray_set(folder: str, b: float, m: int,
                    projector_size: Tuple[int, int],
                    verbose: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Decode Gray code patterns from image set"""
    image_files = get_image_files(folder)

    total_images = len(image_files)
    if verbose:
        print(f"  Found {total_images} images")

    if total_images < 10:
        print(f"ERROR: Too few images in {folder}")
        return None, None

    # Calculate pattern structure
    total_patterns = total_images // 2 - 1
    total_bits = total_patterns // 2
    direct_light_count = 4
    direct_light_offset = 4

    if total_patterns < direct_light_count + direct_light_offset:
        print("ERROR: Too few pattern images")
        return None, None

    # Select high-frequency images for direct light estimation
    direct_images = []
    for i in range(direct_light_count):
        col_idx = 2 + 2 * (total_bits - direct_light_offset + i)
        row_idx = 2 + 2 * total_bits + 2 * (total_bits - direct_light_offset + i)

        if col_idx + 1 < total_images:
            img1 = sl.get_gray_image(image_files[col_idx])
            img2 = sl.get_gray_image(image_files[col_idx + 1])
            if img1 is not None:
                direct_images.append(img1)
            if img2 is not None:
                direct_images.append(img2)

        if row_idx + 1 < total_images:
            img1 = sl.get_gray_image(image_files[row_idx])
            img2 = sl.get_gray_image(image_files[row_idx + 1])
            if img1 is not None:
                direct_images.append(img1)
            if img2 is not None:
                direct_images.append(img2)

    if verbose:
        print("  Estimating direct and global light components...")
    direct_light = sl.estimate_direct_light(direct_images, b)

    # Decode pattern
    if verbose:
        print("  Decoding pattern...")
    pattern_image, min_max_image = sl.decode_pattern(
        image_files, projector_size,
        robust=True, gray_pattern=True,
        direct_light=direct_light, m=m
    )

    return pattern_image, min_max_image


def compute_homography_corners_vectorized(cam_corners: np.ndarray,
                                          pattern_image: np.ndarray,
                                          min_max_image: np.ndarray,
                                          threshold: int,
                                          window_size: int) -> Optional[np.ndarray]:
    """Compute projector corners using local homography (optimized)"""
    proj_corners = []
    half_window = window_size // 2
    height, width = pattern_image.shape[:2]

    # Pre-compute validity mask for entire image
    pattern_col = pattern_image[:, :, 0]
    pattern_row = pattern_image[:, :, 1]
    diff = min_max_image[:, :, 1].astype(np.int32) - min_max_image[:, :, 0].astype(np.int32)

    valid_mask = (
        ~np.isnan(pattern_col) &
        ~np.isnan(pattern_row) &
        (diff >= threshold)
    )

    for corner in cam_corners:
        x, y = int(corner[0]), int(corner[1])

        # Check bounds
        x_min = max(0, x - half_window)
        x_max = min(width, x + half_window)
        y_min = max(0, y - half_window)
        y_max = min(height, y + half_window)

        if x_max - x_min < half_window or y_max - y_min < half_window:
            return None

        # Extract window
        window_valid = valid_mask[y_min:y_max, x_min:x_max]
        window_col = pattern_col[y_min:y_max, x_min:x_max]
        window_row = pattern_row[y_min:y_max, x_min:x_max]

        # Get valid points
        valid_h, valid_w = np.where(window_valid)

        if len(valid_h) < 10:
            return None

        # Build correspondence arrays
        img_points = np.column_stack([
            valid_w + x_min,
            valid_h + y_min
        ]).astype(np.float32)

        proj_points = np.column_stack([
            window_col[valid_h, valid_w],
            window_row[valid_h, valid_w]
        ]).astype(np.float32)

        # Find homography
        H, _ = cv2.findHomography(img_points, proj_points, cv2.RANSAC)

        if H is None:
            return None

        # Transform corner
        pt = np.array([[corner[0]], [corner[1]], [1.0]])
        q = H @ pt
        q = q / q[2]

        proj_corners.append([q[0, 0], q[1, 0]])

    return np.array(proj_corners, dtype=np.float32)


def process_calibration_set(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single calibration set (for parallel execution)

    Args:
        args: dictionary with folder, corner_count, corner_size, b, m, threshold, window, image_scale

    Returns:
        Dictionary with world_corners, cam_corners, proj_corners or None
    """
    folder = args['folder']
    corner_count = args['corner_count']
    corner_size = args['corner_size']
    b = args['b']
    m = args['m']
    threshold = args['threshold']
    window = args['window']
    image_scale = args['image_scale']
    folder_name = os.path.basename(folder)

    try:
        image_files = get_image_files(folder)
        if len(image_files) < 10:
            return {'folder': folder_name, 'error': 'not enough images'}

        # Extract camera corners
        cam_corners = extract_chessboard_corners(
            image_files[1], corner_count, image_scale
        )

        if cam_corners is None:
            return {'folder': folder_name, 'error': 'chessboard not found'}

        # Decode patterns
        proj_size = get_projector_size(folder)
        pattern_image, min_max_image = decode_gray_set(folder, b, m, proj_size, verbose=False)

        if pattern_image is None:
            return {'folder': folder_name, 'error': 'decode failed'}

        # Compute projector corners
        proj_corners = compute_homography_corners_vectorized(
            cam_corners, pattern_image, min_max_image,
            threshold, window
        )

        if proj_corners is None:
            return {'folder': folder_name, 'error': 'homography failed'}

        # Generate world coordinates
        world_corners = get_chessboard_world_coords(corner_count, corner_size)

        return {
            'folder': folder_name,
            'world_corners': world_corners,
            'cam_corners': cam_corners,
            'proj_corners': proj_corners,
            'n_corners': len(cam_corners)
        }

    except Exception as e:
        return {'folder': folder_name, 'error': str(e)}


def calibrate(root_dir: str, corner_count: Tuple[int, int],
              corner_size: Tuple[float, float],
              threshold: int, b: float, m: int,
              homography_window: int,
              parallel: bool = True) -> Optional[CalibrationData]:
    """
    Perform camera-projector stereo calibration

    Optimized with parallel processing of calibration sets.
    """
    print("=" * 60)
    print("Starting Calibration")
    print("=" * 60)
    print(f"Root directory: {root_dir}")
    print(f"Corner count: {corner_count}")
    print(f"Corner size: {corner_size} mm")
    print(f"Parallel processing: {parallel} (workers: {NUM_WORKERS})")

    start_time = time.time()

    # Find all calibration set folders
    folders = sorted([d for d in os.listdir(root_dir)
                      if os.path.isdir(os.path.join(root_dir, d))])

    if len(folders) < 3:
        print("ERROR: Need at least 3 calibration sets")
        return None

    print(f"Found {len(folders)} calibration sets")

    # Get image size from first set
    first_folder = os.path.join(root_dir, folders[0])
    image_files = get_image_files(first_folder)
    if not image_files:
        print(f"ERROR: No images in {first_folder}")
        return None

    first_image = cv2.imread(image_files[1], cv2.IMREAD_GRAYSCALE)
    image_size = (first_image.shape[1], first_image.shape[0])
    image_scale = max(1, int(image_size[0] / 1024))

    projector_size = get_projector_size(first_folder)

    print(f"Image size: {image_size}")
    print(f"Projector size: {projector_size}")
    print(f"Image scale: {image_scale}")

    # Prepare arguments for parallel processing
    process_args = []
    for folder_name in folders:
        process_args.append({
            'folder': os.path.join(root_dir, folder_name),
            'corner_count': corner_count,
            'corner_size': corner_size,
            'b': b,
            'm': m,
            'threshold': threshold,
            'window': homography_window,
            'image_scale': image_scale
        })

    # Process calibration sets
    results = []

    if parallel and len(folders) > 1:
        print(f"\nProcessing {len(folders)} sets in parallel...")

        # Use ThreadPoolExecutor (better for I/O bound tasks like image loading)
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(process_calibration_set, args): args['folder']
                      for args in process_args}

            if HAS_TQDM:
                pbar = tqdm(total=len(futures), desc="Processing sets")

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if HAS_TQDM:
                    pbar.update(1)
                    if 'error' in result:
                        pbar.set_postfix_str(f"Skip: {result['folder']}")
                    else:
                        pbar.set_postfix_str(f"OK: {result['folder']}")
                else:
                    if 'error' in result:
                        print(f"  [{result['folder']}] Skipped: {result['error']}")
                    else:
                        print(f"  [{result['folder']}] OK - {result['n_corners']} corners")

            if HAS_TQDM:
                pbar.close()
    else:
        print(f"\nProcessing {len(folders)} sets sequentially...")
        for i, args in enumerate(process_args):
            print(f"[{i+1}/{len(folders)}] {os.path.basename(args['folder'])}...")
            result = process_calibration_set(args)
            results.append(result)

            if 'error' in result:
                print(f"  Skipped: {result['error']}")
            else:
                print(f"  OK - {result['n_corners']} corners")

    # Collect valid results
    world_corners_all = []
    camera_corners_all = []
    projector_corners_all = []

    for result in results:
        if 'error' not in result:
            world_corners_all.append(result['world_corners'])
            camera_corners_all.append(result['cam_corners'])
            projector_corners_all.append(result['proj_corners'])

    if len(world_corners_all) < 3:
        print("\nERROR: Need at least 3 valid sets for calibration")
        return None

    print(f"\nUsing {len(world_corners_all)} valid sets for calibration")

    # Calibration flags
    cal_flags = cv2.CALIB_FIX_K3

    # Calibrate camera
    print("\nCalibrating camera...")
    cam_error, cam_K, cam_kc, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
        world_corners_all, camera_corners_all, image_size,
        None, None, flags=cal_flags
    )
    print(f"Camera reprojection error: {cam_error:.4f}")

    # Calibrate projector
    print("Calibrating projector...")
    proj_error, proj_K, proj_kc, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        world_corners_all, projector_corners_all, projector_size,
        None, None, flags=cal_flags
    )
    print(f"Projector reprojection error: {proj_error:.4f}")

    # Stereo calibration
    print("Calibrating stereo...")
    stereo_error, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        world_corners_all, camera_corners_all, projector_corners_all,
        cam_K, cam_kc, proj_K, proj_kc, image_size,
        flags=cv2.CALIB_FIX_INTRINSIC + cal_flags
    )
    print(f"Stereo reprojection error: {stereo_error:.4f}")

    from visualize_calibration import save_pose_data
    save_pose_data(
        os.path.join(root_dir, "calibration_poses.npz"),
        cam_rvecs, cam_tvecs, proj_rvecs, proj_tvecs,
        cam_K, cam_kc, proj_K, proj_kc, R, T,
        world_corners_all, camera_corners_all, projector_corners_all
    )

    # Create calibration data
    calib = CalibrationData()
    calib.cam_K = cam_K
    calib.cam_kc = cam_kc
    calib.proj_K = proj_K
    calib.proj_kc = proj_kc
    calib.R = R
    calib.T = T
    calib.cam_error = cam_error
    calib.proj_error = proj_error
    calib.stereo_error = stereo_error

    # Save calibration
    yml_file = os.path.join(root_dir, "calibration.yml")
    calib.save_calibration_yml(yml_file)
    print(f"\nCalibration saved: {yml_file}")

    m_file = os.path.join(root_dir, "calibration.m")
    calib.save_calibration_matlab(m_file)
    print(f"Calibration saved (MATLAB): {m_file}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Calibration Complete! (Time: {elapsed:.1f}s)")
    print("=" * 60)
    calib.display()

    return calib


def reconstruct(image_folder: str, calibration_file: str,
                threshold: int, b: float, m: int, max_dist: float,
                output_file: str = None) -> Optional[scan3d.Pointcloud]:
    """
    Reconstruct 3D model from captured images
    """
    print("=" * 60)
    print("Starting Reconstruction")
    print("=" * 60)

    start_time = time.time()

    # Load calibration
    calib = CalibrationData()
    if not calib.load_calibration(calibration_file):
        print(f"ERROR: Failed to load calibration from {calibration_file}")
        return None

    print("Calibration loaded:")
    calib.display()

    # Get projector size
    projector_size = get_projector_size(image_folder)

    # Decode patterns
    print("\nDecoding patterns...")
    pattern_image, min_max_image = decode_gray_set(image_folder, b, m, projector_size)

    if pattern_image is None:
        print("ERROR: Decode failed")
        return None

    # Compute actual projector resolution from decoded pattern
    pattern_col = pattern_image[:, :, 0]
    pattern_row = pattern_image[:, :, 1]
    valid_mask = ~np.isnan(pattern_col) & ~np.isnan(pattern_row)
    if np.any(valid_mask):
        actual_proj_width  = int(np.nanmax(pattern_col[valid_mask])) + 1
        actual_proj_height = int(np.nanmax(pattern_row[valid_mask])) + 1
        actual_projector_size = (actual_proj_width, actual_proj_height)
    else:
        actual_projector_size = projector_size

    # Load color image (first image)
    image_files = get_image_files(image_folder)
    color_image = cv2.imread(image_files[0])

    # Reconstruct using actual projector resolution
    print("\nReconstructing 3D model...")
    pointcloud = scan3d.Pointcloud()
    scan3d.reconstruct_model(
        pointcloud, calib,
        pattern_image, min_max_image, color_image,
        actual_projector_size, threshold, max_dist
    )

    # Save PLY
    if output_file is None:
        output_file = os.path.join(image_folder, "pointcloud.ply")

    if pointcloud.save_ply(output_file, save_colors=True):
        print(f"\nPoint cloud saved: {output_file}")
    else:
        print("ERROR: Failed to save point cloud")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Reconstruction Complete! (Time: {elapsed:.1f}s)")
    print("=" * 60)
    print(f"Configured projector size : {projector_size[0]} x {projector_size[1]}")
    print(f"Actual projector resolution: {actual_projector_size[0]} x {actual_projector_size[1]}")

    return pointcloud


def main():
    parser = argparse.ArgumentParser(
        description="Structured Light 3D Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py calibrate ./calibration_data --corners 11,7 --size 21,21
    python main.py reconstruct ./scan_001 ./calibration.yml --b 0.5 -o output.ply
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Calibrate camera-projector system")
    cal_parser.add_argument("root_dir", help="Root directory containing calibration sets")
    cal_parser.add_argument("--corners", default=f"{DEFAULT_CORNER_X},{DEFAULT_CORNER_Y}",
                            help=f"Corner count (cols,rows), default: {DEFAULT_CORNER_X},{DEFAULT_CORNER_Y}")
    cal_parser.add_argument("--size", default=f"{DEFAULT_CORNER_WIDTH},{DEFAULT_CORNER_HEIGHT}",
                            help=f"Corner size in mm (width,height), default: {DEFAULT_CORNER_WIDTH},{DEFAULT_CORNER_HEIGHT}")
    cal_parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                            help=f"Shadow threshold, default: {DEFAULT_THRESHOLD}")
    cal_parser.add_argument("--b", type=float, default=DEFAULT_B,
                            help=f"Surface reflectance (b), default: {DEFAULT_B}")
    cal_parser.add_argument("--m", type=int, default=DEFAULT_M,
                            help=f"Min direct light (m), default: {DEFAULT_M}")
    cal_parser.add_argument("--window", type=int, default=DEFAULT_HOMOGRAPHY_WINDOW,
                            help=f"Homography window size, default: {DEFAULT_HOMOGRAPHY_WINDOW}")
    cal_parser.add_argument("--no-parallel", action="store_true",
                            help="Disable parallel processing")

    # Reconstruct command
    rec_parser = subparsers.add_parser("reconstruct", help="Reconstruct 3D model")
    rec_parser.add_argument("image_folder", help="Folder containing captured images")
    rec_parser.add_argument("calibration_file", help="Calibration file (.yml)")
    rec_parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                            help=f"Shadow threshold, default: {DEFAULT_THRESHOLD}")
    rec_parser.add_argument("--b", type=float, default=DEFAULT_B,
                            help=f"Surface reflectance (b), default: {DEFAULT_B}")
    rec_parser.add_argument("--m", type=int, default=DEFAULT_M,
                            help=f"Min direct light (m), default: {DEFAULT_M}")
    rec_parser.add_argument("--max-dist", type=float, default=DEFAULT_MAX_DIST,
                            help=f"Max ray distance, default: {DEFAULT_MAX_DIST}")
    rec_parser.add_argument("--output", "-o", help="Output PLY file")

    args = parser.parse_args()

    if args.command == "calibrate":
        # Parse corners
        corners = tuple(map(int, args.corners.split(",")))
        size = tuple(map(float, args.size.split(",")))

        calibrate(args.root_dir, corners, size,
                  args.threshold, args.b, args.m, args.window,
                  parallel=not args.no_parallel)

    elif args.command == "reconstruct":
        reconstruct(args.image_folder, args.calibration_file,
                    args.threshold, args.b, args.m, args.max_dist,
                    args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()