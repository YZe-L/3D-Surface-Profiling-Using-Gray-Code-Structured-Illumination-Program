"""
scan3d.py - 3D Reconstruction from Structured Light

Ported from Brown University scan3d-capture project
Original authors: Daniel Moreno and Gabriel Taubin

This module handles:
- Stereo triangulation
- Point cloud reconstruction (Adaptive Contrast)
- Normal computation
- Projector view generation

Optimized with NumPy vectorization + batch processing for better performance.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List

from calibration_data import CalibrationData
import structured_light as sl


class Pointcloud:
    """Point cloud data structure"""

    def __init__(self):
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None

    def clear(self):
        self.points = None
        self.colors = None
        self.normals = None

    def init_points(self, rows: int, cols: int):
        self.points = np.full((rows, cols, 3), np.nan, dtype=np.float32)

    def init_color(self, rows: int, cols: int):
        self.colors = np.full((rows, cols, 3), 255, dtype=np.uint8)

    def init_normals(self, rows: int, cols: int):
        self.normals = np.full((rows, cols, 3), np.nan, dtype=np.float32)

    def save_ply(self, filename: str, save_normals: bool = False, save_colors: bool = True,
                 binary: bool = False) -> bool:
        """Save point cloud to PLY file (optimized)"""
        if self.points is None:
            return False

        # Vectorized: find all valid points at once
        valid_mask = ~np.isnan(self.points[:, :, 0])
        valid_indices = np.where(valid_mask)

        num_points = len(valid_indices[0])
        if num_points == 0:
            return False

        # Extract valid data using advanced indexing
        valid_points = self.points[valid_indices]
        valid_colors = self.colors[valid_indices] if self.colors is not None and save_colors else None
        valid_normals = self.normals[valid_indices] if self.normals is not None and save_normals else None

        try:
            with open(filename, 'wb' if binary else 'w') as fp:
                # Write header
                header = "ply\n"
                header += "format binary_little_endian 1.0\n" if binary else "format ascii 1.0\n"
                header += f"element vertex {num_points}\n"
                header += "property float x\nproperty float y\nproperty float z\n"

                if valid_normals is not None:
                    header += "property float nx\nproperty float ny\nproperty float nz\n"

                if valid_colors is not None:
                    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"

                header += "end_header\n"

                if binary:
                    fp.write(header.encode())
                    # Batch write for binary
                    for i in range(num_points):
                        fp.write(valid_points[i].astype(np.float32).tobytes())
                        if valid_normals is not None:
                            fp.write(valid_normals[i].astype(np.float32).tobytes())
                        if valid_colors is not None:
                            c = valid_colors[i]
                            fp.write(np.array([c[2], c[1], c[0]], dtype=np.uint8).tobytes())
                else:
                    fp.write(header)
                    # Build all lines at once then write
                    lines = []
                    for i in range(num_points):
                        pt = valid_points[i]
                        line = f"{pt[0]} {pt[1]} {pt[2]}"
                        if valid_normals is not None:
                            n = valid_normals[i]
                            line += f" {n[0]} {n[1]} {n[2]}"
                        if valid_colors is not None:
                            c = valid_colors[i]
                            line += f" {c[2]} {c[1]} {c[0]}"
                        lines.append(line)
                    fp.write("\n".join(lines) + "\n")

            return True
        except Exception as e:
            print(f"Error saving PLY: {e}")
            return False


# ============================================================================
# Vectorized Ray Intersection (batch processing)
# ============================================================================

def approximate_ray_intersection_batch(v1: np.ndarray, q1: np.ndarray,
                                        v2: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate intersection of multiple ray pairs (fully vectorized)

    Args:
        v1: Nx3 direction vectors of rays 1
        q1: Nx3 origin points of rays 1
        v2: Nx3 direction vectors of rays 2
        q2: Nx3 origin points of rays 2

    Returns:
        points: Nx3 intersection points
        distances: N distances between rays
    """
    # Dot products (vectorized)
    v1tv1 = np.sum(v1 * v1, axis=1)  # N
    v2tv2 = np.sum(v2 * v2, axis=1)  # N
    v1tv2 = np.sum(v1 * v2, axis=1)  # N

    detV = v1tv1 * v2tv2 - v1tv2 * v1tv2

    # Handle parallel rays
    parallel_mask = np.abs(detV) < 1e-10
    detV[parallel_mask] = 1.0  # Avoid division by zero

    q2_q1 = q2 - q1
    Q1 = np.sum(v1 * q2_q1, axis=1)
    Q2 = -np.sum(v2 * q2_q1, axis=1)

    lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) / detV
    lambda2 = (v1tv2 * Q1 + v1tv1 * Q2) / detV

    # Compute points on each ray
    p1 = lambda1[:, np.newaxis] * v1 + q1
    p2 = lambda2[:, np.newaxis] * v2 + q2

    # Midpoint and distance
    points = 0.5 * (p1 + p2)
    distances = np.linalg.norm(p2 - p1, axis=1)

    # Handle parallel rays
    points[parallel_mask] = 0.5 * (q1[parallel_mask] + q2[parallel_mask])
    distances[parallel_mask] = np.linalg.norm(q2[parallel_mask] - q1[parallel_mask], axis=1)

    return points, distances


def triangulate_stereo_batch(K1: np.ndarray, kc1: np.ndarray,
                              K2: np.ndarray, kc2: np.ndarray,
                              Rt: np.ndarray, T: np.ndarray,
                              cam_points: np.ndarray, proj_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate multiple 3D points at once (batch processing)

    Args:
        cam_points: Nx2 array of camera image points
        proj_points: Nx2 array of projector image points

    Returns:
        points_3d: Nx3 array of 3D points
        distances: N array of ray intersection distances
    """
    n_points = cam_points.shape[0]

    if n_points == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float64)

    # Undistort all points at once (OpenCV is already optimized for batch)
    cam_pts = cam_points.reshape(-1, 1, 2).astype(np.float64)
    proj_pts = proj_points.reshape(-1, 1, 2).astype(np.float64)

    cam_undist = cv2.undistortPoints(cam_pts, K1, kc1).reshape(-1, 2)
    proj_undist = cv2.undistortPoints(proj_pts, K2, kc2).reshape(-1, 2)

    # Convert to normalized camera coordinates
    u1 = np.column_stack([cam_undist, np.ones(n_points)])  # Nx3
    u2 = np.column_stack([proj_undist, np.ones(n_points)])  # Nx3

    # Transform to world coordinates
    w1 = u1  # Camera is at origin
    # w2 = Rt @ (u2.T - T) for each point
    w2 = (Rt @ (u2.T - T)).T  # Nx3

    # World rays
    v1 = w1
    v2 = (Rt @ u2.T).T  # Nx3

    # Compute ray-ray intersection for all points
    points_3d, distances = approximate_ray_intersection_batch(v1, w1, v2, w2)

    return points_3d.astype(np.float32), distances


# ============================================================================
# Single point triangulation (for compatibility)
# ============================================================================

def approximate_ray_intersection(v1: np.ndarray, q1: np.ndarray,
                                 v2: np.ndarray, q2: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute approximate intersection of two rays (single point)"""
    v1 = v1.flatten()
    v2 = v2.flatten()
    q1 = q1.flatten()
    q2 = q2.flatten()

    v1tv1 = np.dot(v1, v1)
    v2tv2 = np.dot(v2, v2)
    v1tv2 = np.dot(v1, v2)

    detV = v1tv1 * v2tv2 - v1tv2 * v1tv2

    if abs(detV) < 1e-10:
        return (q1 + q2) / 2, np.linalg.norm(q2 - q1)

    q2_q1 = q2 - q1
    Q1 = np.dot(v1, q2_q1)
    Q2 = -np.dot(v2, q2_q1)

    lambda1 = (v2tv2 * Q1 + v1tv2 * Q2) / detV
    lambda2 = (v1tv2 * Q1 + v1tv1 * Q2) / detV

    p1 = lambda1 * v1 + q1
    p2 = lambda2 * v2 + q2
    p = 0.5 * (p1 + p2)
    distance = np.linalg.norm(p2 - p1)

    return p, distance


def triangulate_stereo(K1: np.ndarray, kc1: np.ndarray,
                       K2: np.ndarray, kc2: np.ndarray,
                       Rt: np.ndarray, T: np.ndarray,
                       p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """Triangulate single 3D point"""
    inp1 = np.array([[[p1[0], p1[1]]]], dtype=np.float64)
    inp2 = np.array([[[p2[0], p2[1]]]], dtype=np.float64)

    outp1 = cv2.undistortPoints(inp1, K1, kc1)
    outp2 = cv2.undistortPoints(inp2, K2, kc2)

    u1 = np.array([outp1[0, 0, 0], outp1[0, 0, 1], 1.0])
    u2 = np.array([outp2[0, 0, 0], outp2[0, 0, 1], 1.0])

    w1 = u1
    w2 = Rt @ (u2.reshape(3, 1) - T)
    w2 = w2.flatten()

    v1 = w1
    v2 = (Rt @ u2.reshape(3, 1)).flatten()

    return approximate_ray_intersection(v1, w1, v2, w2)


# ============================================================================
# Optimized Reconstruction
# ============================================================================

def reconstruct_model(pointcloud: Pointcloud, calib: CalibrationData,
                      pattern_image: np.ndarray, min_max_image: np.ndarray,
                      color_image: np.ndarray, projector_size: Tuple[int, int],
                      threshold: int, max_dist: float,
                      verbose: bool = True) -> None:
    """
    Reconstruct 3D model (Patch Center Method) with Adaptive Contrast
    Fully vectorized + batch triangulation
    """
    if pattern_image is None or min_max_image is None or not calib.is_valid():
        print("[reconstruct_model] ERROR: Invalid input")
        return

    height, width = pattern_image.shape[:2]

    # Init point cloud (projector resolution)
    scale_factor_x = 1
    # scale_factor_y = 1 if projector_size[0] > projector_size[1] else 2  # original: halves Y for portrait projectors
    scale_factor_y = 1  # Full Y resolution for denser point cloud
    out_cols = projector_size[0] // scale_factor_x
    out_rows = projector_size[1] // scale_factor_y

    pointcloud.clear()
    pointcloud.init_points(out_rows, out_cols)
    pointcloud.init_color(out_rows, out_cols)

    if verbose:
        print("Collecting valid points (vectorized)...")

    # ========== Vectorized validity check ==========
    min_contrast_ratio = threshold / 100.0

    # Extract channels
    pattern_col = pattern_image[:, :, 0]
    pattern_row = pattern_image[:, :, 1]
    vmin = min_max_image[:, :, 0].astype(np.int32)
    vmax = min_max_image[:, :, 1].astype(np.int32)
    diff = vmax - vmin

    # Create validity mask
    # Two-tier: strict for bright regions, relaxed for dark regions
    base_valid = (
        ~np.isnan(pattern_col) &
        ~np.isnan(pattern_row) &
        (pattern_col >= 0) & (pattern_col < projector_size[0]) &
        (pattern_row >= 0) & (pattern_row < projector_size[1])
    )

    # Standard quality gate
    standard_valid = base_valid & (diff >= 3) & (
        (diff / np.maximum(vmax, 1)) >= min_contrast_ratio
    )

    # Relaxed gate for dark regions: lower absolute noise floor,
    # but require higher contrast RATIO (dark surfaces still modulate)
    dark_region = (vmax < 40)  # very dark surface
    dark_valid = base_valid & dark_region & (diff >= 1) & (
        (diff / np.maximum(vmax, 1)) >= min_contrast_ratio * 0.3
    )

    valid_mask = standard_valid | dark_valid

    if verbose:
        n_dark = np.sum(dark_valid & ~standard_valid)
        if n_dark > 0:
            print(f"Dark pixel rescue: {n_dark} extra pixels from low-reflectance regions")

    # ========== Spatial consistency filter ==========
    # Catches decode errors from specular reflections: if a pixel's decoded
    # projector coordinate jumps drastically compared to its camera neighbors,
    # it's likely a misdecode from saturated/reflective regions.
    # To disable: comment out this block.
    max_proj_jump = 3.0  # max allowed projector-pixel jump between adjacent camera pixels
    for ch in range(2):  # column and row channels
        pat_ch = pattern_image[:, :, ch].copy()
        pat_ch[~valid_mask] = np.nan
        # Compare with 4-neighbors
        d_left  = np.abs(pat_ch[:, 1:] - pat_ch[:, :-1])
        d_right = np.abs(pat_ch[:, :-1] - pat_ch[:, 1:])
        d_up    = np.abs(pat_ch[1:, :] - pat_ch[:-1, :])
        d_down  = np.abs(pat_ch[:-1, :] - pat_ch[1:, :])
        # A pixel is inconsistent if ALL valid neighbors have large jumps
        pad_inf = np.full_like(pat_ch, np.nan)
        jump_l = pad_inf.copy(); jump_l[:, 1:] = d_left
        jump_r = pad_inf.copy(); jump_r[:, :-1] = d_right
        jump_u = pad_inf.copy(); jump_u[1:, :] = d_up
        jump_d = pad_inf.copy(); jump_d[:-1, :] = d_down
        jumps = np.stack([jump_l, jump_r, jump_u, jump_d], axis=2)
        with np.errstate(all='ignore'):
            min_jump = np.nanmin(jumps, axis=2)
        # If even the smallest jump to any neighbor is too large, it's an outlier
        bad = (min_jump > max_proj_jump) & valid_mask
        valid_mask = valid_mask & ~bad
    n_consistency_removed = np.sum(bad) if 'bad' in dir() else 0
    if verbose and n_consistency_removed > 0:
        print(f"Spatial consistency: removed {n_consistency_removed} mismatched pixels")
    # ========== End spatial consistency filter ==========

    # Get valid pixel coordinates
    valid_h, valid_w = np.where(valid_mask)
    n_valid = len(valid_h)

    if verbose:
        print(f"Found {n_valid} valid pixels")

    if n_valid == 0:
        return

    # Get projector coordinates for valid pixels
    proj_x = (pattern_col[valid_h, valid_w] / scale_factor_x).astype(np.int32)
    proj_y = (pattern_row[valid_h, valid_w] / scale_factor_y).astype(np.int32)

    # Defaults for when crop block is commented out
    crop_x_min = 0
    crop_y_min = 0

    # ========== Crop to camera-visible region ==========
    # Trims warped edges where projector illuminates but camera can't properly observe.
    # Adjust margins independently per edge (in projector pixels).
    # To disable: comment out from here to 'End crop block'.
    crop_left   = 0
    crop_right  = 0
    crop_top    = 0
    crop_bottom = 0

    crop_x_min = int(proj_x.min()) + crop_left
    crop_x_max = int(proj_x.max()) - crop_right
    crop_y_min = int(proj_y.min()) + crop_top
    crop_y_max = int(proj_y.max()) - crop_bottom

    crop_mask = (
        (proj_x >= crop_x_min) & (proj_x <= crop_x_max) &
        (proj_y >= crop_y_min) & (proj_y <= crop_y_max)
    )
    valid_h = valid_h[crop_mask]
    valid_w = valid_w[crop_mask]
    proj_x = proj_x[crop_mask]
    proj_y = proj_y[crop_mask]

    out_cols = crop_x_max - crop_x_min + 1
    out_rows = crop_y_max - crop_y_min + 1
    pointcloud.clear()
    pointcloud.init_points(out_rows, out_cols)
    pointcloud.init_color(out_rows, out_cols)

    if verbose:
        print(f"Crop: L={crop_left} R={crop_right} T={crop_top} B={crop_bottom}, "
              f"output={out_cols}x{out_rows}, pixels={len(valid_h)}")
    if len(valid_h) == 0:
        return
    # ========== End crop block ==========

    # ========== Group by projector pixel ==========
    store_x = proj_x - crop_x_min
    store_y = proj_y - crop_y_min
    proj_indices = store_y * out_cols + store_x
    unique_proj_indices, inverse_indices = np.unique(proj_indices, return_inverse=True)

    if verbose:
        print(f"Found {len(unique_proj_indices)} unique projector pixels")
        print("Computing camera centers...")

    # Compute center of camera points for each unique projector pixel
    n_unique = len(unique_proj_indices)

    cam_sum_x = np.bincount(inverse_indices, weights=valid_w.astype(np.float64), minlength=n_unique)
    cam_sum_y = np.bincount(inverse_indices, weights=valid_h.astype(np.float64), minlength=n_unique)
    cam_count = np.bincount(inverse_indices, minlength=n_unique)

    # Accumulate ORIGINAL projector coords for triangulation
    proj_orig_sum_x = np.bincount(inverse_indices, weights=proj_x.astype(np.float64), minlength=n_unique)
    proj_orig_sum_y = np.bincount(inverse_indices, weights=proj_y.astype(np.float64), minlength=n_unique)

    # Avoid division by zero
    cam_count[cam_count == 0] = 1

    cam_center_x = cam_sum_x / cam_count
    cam_center_y = cam_sum_y / cam_count

    # Storage coordinates for pointcloud array indexing
    proj_px = unique_proj_indices % out_cols
    proj_py = unique_proj_indices // out_cols

    # ORIGINAL projector coordinates for triangulation
    orig_px = proj_orig_sum_x / cam_count
    orig_py = proj_orig_sum_y / cam_count

    # Prepare batch arrays - use ORIGINAL proj coords for geometry!
    cam_points = np.column_stack([cam_center_x, cam_center_y])
    proj_points = np.column_stack([orig_px * scale_factor_x, orig_py * scale_factor_y])

    if verbose:
        print("Triangulating (batch)...")

    # ========== Batch triangulation ==========
    Rt = calib.R.T
    points_3d, distances = triangulate_stereo_batch(
        calib.cam_K, calib.cam_kc,
        calib.proj_K, calib.proj_kc,
        Rt, calib.T,
        cam_points, proj_points
    )

    # ========== Store results ==========
    good_mask = distances < max_dist
    good_count = np.sum(good_mask)

    if verbose:
        print(f"Storing {good_count} good points...")

    # Store points
    good_py = proj_py[good_mask]
    good_px = proj_px[good_mask]
    pointcloud.points[good_py, good_px] = points_3d[good_mask]

    # Store colors
    if color_image is not None:
        good_cam_y = np.clip(cam_center_y[good_mask].astype(np.int32), 0, color_image.shape[0] - 1)
        good_cam_x = np.clip(cam_center_x[good_mask].astype(np.int32), 0, color_image.shape[1] - 1)
        pointcloud.colors[good_py, good_px] = color_image[good_cam_y, good_cam_x]

    # ========== Remove outliers ==========
    remove_outliers(pointcloud, max_neighbor_dist=8.0, min_neighbors=1, adaptive=True, adaptive_factor=3.0)

    if verbose:
        print(f"Reconstructed points: {good_count}")


def remove_outliers(pointcloud: Pointcloud, max_neighbor_dist: float = 2.0, min_neighbors: int = 2,
                    adaptive: bool = True, adaptive_factor: float = 3.0):
    """
    Filter flying pixels based on neighbor distance (Vectorized).

    Two modes:
      adaptive=False: Original fixed-threshold mode.
      adaptive=True:  Uses local adaptive threshold based on median neighbor
                      distance * adaptive_factor. Preserves sharp tips/edges
                      where spacing is naturally larger but consistent.

    Args:
        max_neighbor_dist: Absolute upper bound distance.
        min_neighbors: Minimum valid neighbors to keep a point.
        adaptive: Enable adaptive thresholding.
        adaptive_factor: Multiplier for local median (higher = keeps more sharp features).
    """
    if pointcloud.points is None:
        return

    mode_str = f"adaptive(factor={adaptive_factor})" if adaptive else f"fixed(dist={max_neighbor_dist})"
    print(f"Filtering outliers ({mode_str}, min_neighbors={min_neighbors})...")

    points = pointcloud.points
    rows, cols = points.shape[:2]

    # Calculate distance to 4 neighbors
    left_shift = np.roll(points, 1, axis=1)
    right_shift = np.roll(points, -1, axis=1)
    up_shift = np.roll(points, 1, axis=0)
    down_shift = np.roll(points, -1, axis=0)

    dist_left = np.linalg.norm(points - left_shift, axis=2)
    dist_right = np.linalg.norm(points - right_shift, axis=2)
    dist_up = np.linalg.norm(points - up_shift, axis=2)
    dist_down = np.linalg.norm(points - down_shift, axis=2)

    # Fix boundaries
    dist_left[:, 0] = np.inf
    dist_right[:, -1] = np.inf
    dist_up[0, :] = np.inf
    dist_down[-1, :] = np.inf

    if adaptive:
        # Stack all 4 distances: (rows, cols, 4)
        all_dists = np.stack([dist_left, dist_right, dist_up, dist_down], axis=2)
        all_dists_clean = np.where(np.isinf(all_dists), np.nan, all_dists)

        # Local median of each point's neighbor distances
        with np.errstate(all='ignore'):
            local_median = np.nanmedian(all_dists_clean, axis=2)

        # Fallback for all-NaN points
        global_median = np.nanmedian(all_dists_clean)
        local_median = np.where(np.isnan(local_median), global_median, local_median)

        # Adaptive threshold: local_median * factor, capped by absolute max
        threshold = np.minimum(local_median * adaptive_factor, max_neighbor_dist)

        valid_left  = (dist_left  < threshold).astype(np.int8)
        valid_right = (dist_right < threshold).astype(np.int8)
        valid_up    = (dist_up    < threshold).astype(np.int8)
        valid_down  = (dist_down  < threshold).astype(np.int8)
    else:
        valid_left  = (dist_left  < max_neighbor_dist).astype(np.int8)
        valid_right = (dist_right < max_neighbor_dist).astype(np.int8)
        valid_up    = (dist_up    < max_neighbor_dist).astype(np.int8)
        valid_down  = (dist_down  < max_neighbor_dist).astype(np.int8)

    neighbor_count = valid_left + valid_right + valid_up + valid_down
    keep_mask = neighbor_count >= min_neighbors

    removed = np.sum(~np.isnan(points[:,:,0]) & ~keep_mask)
    total = np.sum(~np.isnan(points[:,:,0]))
    print(f"Outlier removal: removed {removed}/{total} points ({100*removed/max(total,1):.1f}%)")

    # Apply mask
    pointcloud.points = np.where(keep_mask[:, :, None], points, np.nan).astype(np.float32)


def reconstruct_model_simple(pointcloud: Pointcloud, calib: CalibrationData,
                             pattern_image: np.ndarray, min_max_image: np.ndarray,
                             color_image: np.ndarray, projector_size: Tuple[int, int],
                             threshold: int, max_dist: float,
                             verbose: bool = True) -> None:
    """
    Reconstruct 3D model (Simple Method) - Fully vectorized
    """
    if pattern_image is None or min_max_image is None or not calib.is_valid():
        print("[reconstruct_model] ERROR: Invalid input")
        return

    height, width = pattern_image.shape[:2]

    pointcloud.clear()
    pointcloud.init_points(height, width)
    pointcloud.init_color(height, width)

    if verbose:
        print("Collecting valid points (vectorized)...")

    # Vectorized validity check
    min_contrast_ratio = threshold / 100.0

    pattern_col = pattern_image[:, :, 0]
    pattern_row = pattern_image[:, :, 1]
    vmin = min_max_image[:, :, 0].astype(np.int32)
    vmax = min_max_image[:, :, 1].astype(np.int32)
    diff = vmax - vmin

    valid_mask = (
        ~np.isnan(pattern_col) &
        ~np.isnan(pattern_row) &
        (pattern_col >= 0) & (pattern_col < projector_size[0]) &
        (pattern_row >= 0) & (pattern_row < projector_size[1]) &
        (diff >= 5) &
        ((diff / np.maximum(vmax, 1)) >= min_contrast_ratio)
    )

    valid_h, valid_w = np.where(valid_mask)
    n_valid = len(valid_h)

    if verbose:
        print(f"Found {n_valid} valid pixels")

    if n_valid == 0:
        return

    # Prepare batch arrays
    cam_points = np.column_stack([valid_w, valid_h]).astype(np.float64)
    proj_points = np.column_stack([
        pattern_col[valid_h, valid_w],
        pattern_row[valid_h, valid_w]
    ]).astype(np.float64)

    if verbose:
        print("Triangulating (batch)...")

    # Batch triangulation
    Rt = calib.R.T
    points_3d, distances = triangulate_stereo_batch(
        calib.cam_K, calib.cam_kc,
        calib.proj_K, calib.proj_kc,
        Rt, calib.T,
        cam_points, proj_points
    )

    # Store results
    good_mask = distances < max_dist
    good_count = np.sum(good_mask)

    good_h = valid_h[good_mask]
    good_w = valid_w[good_mask]
    pointcloud.points[good_h, good_w] = points_3d[good_mask]

    if color_image is not None:
        pointcloud.colors[good_h, good_w] = color_image[good_h, good_w]

    if verbose:
        print(f"Reconstructed points: {good_count}")


def compute_normals(pointcloud: Pointcloud) -> None:
    """Compute surface normals (vectorized)"""
    if pointcloud.points is None:
        return

    rows, cols = pointcloud.points.shape[:2]
    pointcloud.init_normals(rows, cols)

    pts = pointcloud.points

    # Compute differences using slicing (much faster than loop)
    dx = pts[1:-1, 2:, :] - pts[1:-1, :-2, :]  # horizontal diff
    dy = pts[2:, 1:-1, :] - pts[:-2, 1:-1, :]  # vertical diff

    # Cross product for normals
    normals = np.cross(dy, dx)

    # Normalize
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0] = 1  # Avoid division by zero
    normals = normals / norm

    # Check for valid neighbors
    valid = (
        ~np.isnan(pts[1:-1, :-2, 0]) &
        ~np.isnan(pts[1:-1, 2:, 0]) &
        ~np.isnan(pts[:-2, 1:-1, 0]) &
        ~np.isnan(pts[2:, 1:-1, 0])
    )

    # Store normals
    pointcloud.normals[1:-1, 1:-1] = np.where(valid[:, :, np.newaxis], normals, np.nan)


def make_projector_view(pattern_image: np.ndarray, min_max_image: np.ndarray,
                        color_image: np.ndarray, projector_size: Tuple[int, int],
                        threshold: int) -> Optional[np.ndarray]:
    """Create projector view image (vectorized)"""
    if pattern_image is None or min_max_image is None:
        return None

    scale_factor_x = 1
    scale_factor_y = 1 if projector_size[0] > projector_size[1] else 2
    out_cols = projector_size[0] // scale_factor_x
    out_rows = projector_size[1] // scale_factor_y

    projector_image = np.full((out_rows, out_cols, 3), 255, dtype=np.uint8)

    # Vectorized validity check
    min_contrast_ratio = threshold / 100.0

    pattern_col = pattern_image[:, :, 0]
    pattern_row = pattern_image[:, :, 1]
    vmin = min_max_image[:, :, 0].astype(np.int32)
    vmax = min_max_image[:, :, 1].astype(np.int32)
    diff = vmax - vmin

    valid_mask = (
        ~np.isnan(pattern_col) &
        ~np.isnan(pattern_row) &
        (pattern_col >= 0) & (pattern_col < projector_size[0]) &
        (pattern_row >= 0) & (pattern_row < projector_size[1]) &
        (diff >= 5) &
        ((diff / np.maximum(vmax, 1)) >= min_contrast_ratio)
    )

    valid_h, valid_w = np.where(valid_mask)

    proj_x = (pattern_col[valid_h, valid_w] / scale_factor_x).astype(np.int32)
    proj_y = (pattern_row[valid_h, valid_w] / scale_factor_y).astype(np.int32)

    if color_image is not None:
        projector_image[proj_y, proj_x] = color_image[valid_h, valid_w]

    return projector_image