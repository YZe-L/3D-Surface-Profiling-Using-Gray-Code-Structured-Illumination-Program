"""
structured_light.py - Gray Code Pattern Decoding

Ported from Brown University scan3d-capture project
Original authors: Daniel Moreno and Gabriel Taubin

This module handles:
- Gray code to binary conversion
- Pattern decoding from captured images
- Direct/global light separation (Nayar's algorithm)
- Robust bit classification

Optimized with NumPy vectorization for better performance.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

# Constants
PIXEL_UNCERTAIN = np.nan
BIT_UNCERTAIN = 0xffff


def INVALID(value) -> bool:
    """Check if value is invalid (NaN)"""
    if isinstance(value, (float, np.floating)):
        return np.isnan(value)
    elif isinstance(value, np.ndarray):
        return np.any(np.isnan(value))
    return False


def binary_to_gray(num):
    """Convert binary to Gray code (supports scalar and array)"""
    return (num >> 1) ^ num


def gray_to_binary(num, num_bits: int = 32):
    """Convert Gray code to binary (supports scalar and array)"""
    result = np.asarray(num).copy()
    shift = 1
    while shift < num_bits:
        result ^= (result >> shift)
        shift <<= 1
    return result


def get_gray_image(filename: str) -> Optional[np.ndarray]:
    """Load image and convert to grayscale"""
    rgb_image = cv2.imread(filename)
    if rgb_image is not None and rgb_image.shape[0] > 0:
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        return gray_image
    return None


def get_robust_bit_vectorized(value1: np.ndarray, value2: np.ndarray,
                               Ld: np.ndarray, Lg: np.ndarray, m: int) -> np.ndarray:
    """
    Robust bit classification (fully vectorized)

    Returns:
        result: array with 0, 1, or BIT_UNCERTAIN
    """
    result = np.full(value1.shape, BIT_UNCERTAIN, dtype=np.uint16)

    # Condition 1: Ld >= m (valid pixels)
    valid = Ld >= m

    # Condition 2: Ld > Lg -> simple comparison
    simple_mask = valid & (Ld > Lg)
    result[simple_mask] = np.where(value1[simple_mask] > value2[simple_mask], 1, 0)

    # Condition 3: Ld <= Lg -> complex comparison
    complex_mask = valid & (Ld <= Lg)

    # value1 <= Ld and value2 >= Lg -> 0
    zero_mask = complex_mask & (value1 <= Ld) & (value2 >= Lg)
    result[zero_mask] = 0

    # value1 >= Lg and value2 <= Ld -> 1
    one_mask = complex_mask & (value1 >= Lg) & (value2 <= Ld)
    result[one_mask] = 1

    return result


def estimate_direct_light(images: List[np.ndarray], b: float,
                          saturation_thresh: int = 250) -> Optional[np.ndarray]:
    """
    Estimate direct and global light components using Nayar's algorithm (vectorized)
    Enhanced with saturation detection for specular/reflective surfaces.

    Args:
        images: list of grayscale images (high-frequency patterns)
        b: surface reflectance parameter (0-1)
        saturation_thresh: pixel values above this are considered saturated (0-255)

    Returns:
        direct_light: 3-channel image [Ld, Lg, saturation_flag]
                      saturation_flag: 0=normal, 1=partially saturated, 2=fully saturated
    """
    MAX_COUNT = 10

    count = len(images)
    if count < 1:
        return None

    print(" --- estimate_direct_light START ---")

    if count > MAX_COUNT:
        count = MAX_COUNT
        print(f"WARNING: Using only {MAX_COUNT} of {len(images)}")

    # Verify all images are grayscale
    for i in range(count):
        if len(images[i].shape) != 2:
            print("Gray images required")
            return None

    height, width = images[0].shape

    # Stack images for vectorized operations
    stack = np.stack(images[:count], axis=0).astype(np.float32)

    Lmax = np.max(stack, axis=0)
    Lmin = np.min(stack, axis=0)

    # ========== Saturation detection ==========
    # Count how many images have saturated pixels at each location
    sat_count = np.sum(stack >= saturation_thresh, axis=0)  # (H, W)
    # 0=normal, 1=partial saturation (some images saturated), 2=full (all saturated)
    sat_flag = np.zeros((height, width), dtype=np.uint8)
    sat_flag[sat_count > 0] = 1           # some images saturated
    sat_flag[sat_count >= count * 0.8] = 2  # most/all images saturated (strong specular)

    n_partial = np.sum(sat_flag == 1)
    n_full = np.sum(sat_flag == 2)
    if n_partial > 0 or n_full > 0:
        print(f"  Saturation detected: {n_partial} partial, {n_full} fully saturated pixels")

    b1 = 1.0 / (1.0 - b)
    b2 = 2.0 / (1.0 - b * b)

    Ld = (b1 * (Lmax - Lmin) + 0.5).astype(np.int32)
    Lg = (b2 * (Lmin - b * Lmax) + 0.5).astype(np.int32)

    # For partially saturated pixels, Nayar's formula underestimates Ld
    # because Lmax is clipped. Compensate by boosting Ld estimate.
    partial_mask = sat_flag == 1
    if np.any(partial_mask):
        # Use Lmin-based estimate: if only some are saturated, Lmin is still reliable
        # Ld_corrected ~ (Lmax_true - Lmin) / (1-b), approximate Lmax_true
        Ld[partial_mask] = np.clip(
            (b1 * (255 + (255 - Lmin[partial_mask]) * 0.5 - Lmin[partial_mask]) + 0.5).astype(np.int32),
            Ld[partial_mask], 255  # at least as much as original estimate
        )

    # Initialize direct light image (3 channels: Ld, Lg, sat_flag)
    direct_light = np.zeros((height, width, 3), dtype=np.uint8)

    # Where Lg > 0, use computed values; otherwise use Lmax for Ld and 0 for Lg
    mask = Lg > 0

    direct_light[:, :, 0] = np.where(mask, np.clip(Ld, 0, 255), Lmax).astype(np.uint8)
    direct_light[:, :, 1] = np.where(mask, np.clip(Lg, 0, 255), 0).astype(np.uint8)
    direct_light[:, :, 2] = sat_flag

    print(" --- estimate_direct_light END ---")

    return direct_light


def convert_pattern(pattern_image: np.ndarray, projector_size: Tuple[int, int],
                    offset: List[int], binary: bool) -> np.ndarray:
    """
    Convert pattern between Gray code and binary (vectorized)
    """
    if pattern_image is None or pattern_image.size == 0:
        return pattern_image

    if pattern_image.dtype != np.float32 or len(pattern_image.shape) != 3 or pattern_image.shape[2] != 2:
        return pattern_image

    if binary:
        print("Converting binary code to gray")
    else:
        print("Converting gray code to binary")

    result = pattern_image.copy()

    for ch in range(2):
        valid = ~np.isnan(result[:, :, ch])
        p = result[:, :, ch].astype(np.int32)
        frac = result[:, :, ch] - p

        if binary:
            # Binary to Gray
            converted = binary_to_gray(p + offset[ch]) + frac
        else:
            # Gray to Binary
            code = gray_to_binary(p, 32) - offset[ch]
            code = np.clip(code, 0, projector_size[ch] - 1)
            converted = code + frac

        result[:, :, ch] = np.where(valid, converted, result[:, :, ch])

    return result


def decode_pattern(image_files: List[str], projector_size: Tuple[int, int],
                   robust: bool = True, gray_pattern: bool = True,
                   direct_light: np.ndarray = None, m: int = 5) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Decode Gray code pattern from captured images (vectorized version)
    """
    print(" --- decode_pattern START ---")

    total_images = len(image_files)
    total_patterns = total_images // 2 - 1
    total_bits = total_patterns // 2

    print(f"  Total images: {total_images}")
    print(f"  Total patterns: {total_patterns}")
    print(f"  Total bits: {total_bits}")
    print(f"  Expected: 2 + 4 * {total_bits} = {2 + 4 * total_bits}")

    if 2 + 4 * total_bits != total_images:
        print("[decode_pattern] ERROR: cannot detect pattern and bit count from image set.")
        return None, None

    bit_count = [0, total_bits, total_bits]
    set_size = [1, total_bits, total_bits]
    COUNT = 2 * (set_size[0] + set_size[1] + set_size[2])
    pattern_offset = [
        ((1 << total_bits) - projector_size[0]) // 2,
        ((1 << total_bits) - projector_size[1]) // 2
    ]

    if len(image_files) < COUNT:
        print(f"Image list size does not match set size, please supply exactly {COUNT} image names.")
        return None, None

    print(f"Decode: {'Gray ' if gray_pattern else 'Binary '}{'Robust ' if robust else ''}")

    # Load first image to get size
    first_img = get_gray_image(image_files[2])  # Skip white/black
    if first_img is None:
        print(f"Failed to load {image_files[2]}")
        return None, None

    height, width = first_img.shape

    # Initialize arrays
    pattern_image = np.zeros((height, width, 2), dtype=np.float32)
    min_max_image = np.zeros((height, width, 2), dtype=np.uint8)
    min_max_image[:, :, 0] = 255  # Initialize min to max value
    min_max_image[:, :, 1] = 0    # Initialize max to min value

    # Track uncertain pixels
    uncertain = np.zeros((height, width, 2), dtype=bool)

    # Get direct light if using robust decoding
    sat_flag = None
    if robust and direct_light is not None:
        Ld = direct_light[:, :, 0].astype(np.int32)
        Lg = direct_light[:, :, 1].astype(np.int32)
        if direct_light.shape[2] >= 3:
            sat_flag = direct_light[:, :, 2]

    current_set = 0
    current = 0

    for t in range(0, COUNT, 2):
        if current == set_size[current_set]:
            current_set += 1
            current = 0

        if current_set == 0:
            current += 1
            continue

        bit = bit_count[current_set] - current - 1
        channel = current_set - 1

        # Load image pair
        gray_image1 = get_gray_image(image_files[t])
        gray_image2 = get_gray_image(image_files[t + 1])

        if gray_image1 is None or gray_image2 is None:
            print(f"Failed to load image pair {t}")
            return None, None

        # Check size
        if gray_image1.shape != (height, width) or gray_image2.shape != (height, width):
            print(f" --> Image pair {t} has different size (skipped!)")
            current += 1
            continue

        value1 = gray_image1.astype(np.int32)
        value2 = gray_image2.astype(np.int32)

        # Update min/max (vectorized)
        min_val = np.minimum(value1, value2).astype(np.uint8)
        max_val = np.maximum(value1, value2).astype(np.uint8)
        min_max_image[:, :, 0] = np.minimum(min_max_image[:, :, 0], min_val)
        min_max_image[:, :, 1] = np.maximum(min_max_image[:, :, 1], max_val)

        if not robust:
            # Simple pattern bit assignment (vectorized)
            mask = value1 > value2
            pattern_image[:, :, channel] += np.where(mask, 1 << bit, 0).astype(np.float32)
        else:
            # Robust pattern bit assignment (vectorized)
            if direct_light is not None:
                bits = get_robust_bit_vectorized(value1, value2, Ld, Lg, m)

                # ---- Rescue saturated pixels ----
                if sat_flag is not None:
                    partial = (sat_flag == 1) & (bits == BIT_UNCERTAIN)
                    diff_abs = np.abs(value1.astype(np.float32) - value2.astype(np.float32))
                    rescuable = partial & (diff_abs > m * 2)
                    bits[rescuable & (value1 > value2)] = 1
                    bits[rescuable & (value1 <= value2)] = 0

                # ---- Rescue dark/low-reflectance pixels ----
                # For very dark surfaces, Ld < m so robust classification gives UNCERTAIN.
                # But if pattern vs inverse still shows a detectable difference, we can
                # recover the bit using simple comparison with a stricter threshold.
                dark_uncertain = (bits == BIT_UNCERTAIN) & (Ld < m) & (Ld > 0)
                if np.any(dark_uncertain):
                    diff_dark = np.abs(value1.astype(np.float32) - value2.astype(np.float32))
                    # For dark pixels, even a small difference (> 1) can be meaningful
                    # if it's consistent relative to the signal level
                    # Require diff > max(1, Ld * 0.3) to be somewhat confident
                    dark_thresh = np.maximum(1, Ld * 0.3).astype(np.float32)
                    dark_rescuable = dark_uncertain & (diff_dark > dark_thresh)
                    bits[dark_rescuable & (value1 > value2)] = 1
                    bits[dark_rescuable & (value1 <= value2)] = 0

                # Mark uncertain pixels
                new_uncertain = (bits == BIT_UNCERTAIN)
                uncertain[:, :, channel] |= new_uncertain

                # Add bits to pattern (only for non-uncertain)
                valid_bits = ~uncertain[:, :, channel]
                pattern_image[:, :, channel] += np.where(valid_bits & (bits == 1), 1 << bit, 0).astype(np.float32)

        current += 1

    # Set uncertain pixels to NaN
    pattern_image[uncertain] = PIXEL_UNCERTAIN

    # Convert Gray code to binary if needed
    if gray_pattern:
        pattern_image = convert_pattern(pattern_image, projector_size, pattern_offset, binary=False)

    print(" --- decode_pattern END ---")

    return pattern_image, min_max_image


def colorize_pattern(pattern_image: np.ndarray, channel: int, max_value: float) -> Optional[np.ndarray]:
    """
    Create color visualization of decoded pattern (vectorized)
    """
    if pattern_image is None or pattern_image.size == 0:
        return None

    if pattern_image.dtype != np.float32 or len(pattern_image.shape) != 3 or pattern_image.shape[2] != 2:
        return None

    if channel not in [0, 1]:
        return None

    height, width = pattern_image.shape[:2]
    image = np.full((height, width, 3), 128, dtype=np.uint8)  # Default grey

    values = pattern_image[:, :, channel]
    valid = ~np.isnan(values) & (values <= max_value)

    t = values * 255.0 / max_value
    n = 4.0
    dt = 255.0 / n

    # Black -> Red (t <= dt)
    mask = valid & (t <= dt)
    c = n * t
    image[:, :, 2] = np.where(mask, np.clip(c, 0, 255), image[:, :, 2])  # R
    image[:, :, 1] = np.where(mask, 0, image[:, :, 1])  # G
    image[:, :, 0] = np.where(mask, 0, image[:, :, 0])  # B

    # Red -> Yellow (dt < t <= 2*dt)
    mask = valid & (t > dt) & (t <= 2 * dt)
    c = n * (t - dt)
    image[:, :, 2] = np.where(mask, 255, image[:, :, 2])
    image[:, :, 1] = np.where(mask, np.clip(c, 0, 255), image[:, :, 1])
    image[:, :, 0] = np.where(mask, 0, image[:, :, 0])

    # Yellow -> Green (2*dt < t <= 3*dt)
    mask = valid & (t > 2 * dt) & (t <= 3 * dt)
    c = n * (t - 2 * dt)
    image[:, :, 2] = np.where(mask, np.clip(255 - c, 0, 255), image[:, :, 2])
    image[:, :, 1] = np.where(mask, 255, image[:, :, 1])
    image[:, :, 0] = np.where(mask, 0, image[:, :, 0])

    # Green -> Blue (t > 3*dt)
    mask = valid & (t > 3 * dt)
    c = n * (t - 3 * dt)
    image[:, :, 2] = np.where(mask, 0, image[:, :, 2])
    image[:, :, 1] = np.where(mask, np.clip(255 - c, 0, 255), image[:, :, 1])
    image[:, :, 0] = np.where(mask, np.clip(c, 0, 255), image[:, :, 0])

    return image.astype(np.uint8)