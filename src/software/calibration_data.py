"""
calibration_data.py - Camera-Projector Calibration Data

Ported from Brown University scan3d-capture project
Original authors: Daniel Moreno and Gabriel Taubin

This module handles:
- Loading/saving calibration data in YAML format
- Saving calibration data in MATLAB format
"""

import numpy as np
import cv2
from typing import Optional
import os


class CalibrationData:
    """Camera-Projector stereo calibration data"""

    CALIBRATION_FILE_VERSION = 1

    __slots__ = ['cam_K', 'cam_kc', 'proj_K', 'proj_kc', 'R', 'T',
                 'cam_error', 'proj_error', 'stereo_error', 'filename']

    def __init__(self):
        self.cam_K: Optional[np.ndarray] = None
        self.cam_kc: Optional[np.ndarray] = None
        self.proj_K: Optional[np.ndarray] = None
        self.proj_kc: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None

        self.cam_error: float = 0.0
        self.proj_error: float = 0.0
        self.stereo_error: float = 0.0

        self.filename: str = ""

    def clear(self):
        """Clear all calibration data"""
        self.cam_K = None
        self.cam_kc = None
        self.proj_K = None
        self.proj_kc = None
        self.R = None
        self.T = None
        self.filename = ""
        self.cam_error = 0.0
        self.proj_error = 0.0
        self.stereo_error = 0.0

    def is_valid(self) -> bool:
        """Check if calibration data is valid"""
        return all(x is not None for x in
                   [self.cam_K, self.cam_kc, self.proj_K, self.proj_kc, self.R, self.T])

    def load_calibration(self, filename: str) -> bool:
        """Load calibration from file (auto-detect format)"""
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}")
            return False

        ext = os.path.splitext(filename)[1].lower()

        if ext in (".yml", ".yaml"):
            return self.load_calibration_yml(filename)

        print(f"Error: Unsupported file format: {ext}")
        return False

    def save_calibration(self, filename: str) -> bool:
        """Save calibration to file (auto-detect format)"""
        ext = os.path.splitext(filename)[1].lower()

        if ext in (".yml", ".yaml"):
            return self.save_calibration_yml(filename)
        elif ext == ".m":
            return self.save_calibration_matlab(filename)

        print(f"Error: Unsupported file format: {ext}")
        return False

    def load_calibration_yml(self, filename: str) -> bool:
        """Load calibration from YAML file"""
        try:
            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print(f"Error: Cannot open file: {filename}")
                return False

            # Load matrices
            self.cam_K = fs.getNode("cam_K").mat()
            self.cam_kc = fs.getNode("cam_kc").mat()
            self.proj_K = fs.getNode("proj_K").mat()
            self.proj_kc = fs.getNode("proj_kc").mat()
            self.R = fs.getNode("R").mat()
            self.T = fs.getNode("T").mat()

            # Load errors (with fallback)
            cam_err_node = fs.getNode("cam_error")
            self.cam_error = cam_err_node.real() if not cam_err_node.empty() else 0.0

            proj_err_node = fs.getNode("proj_error")
            self.proj_error = proj_err_node.real() if not proj_err_node.empty() else 0.0

            stereo_err_node = fs.getNode("stereo_error")
            self.stereo_error = stereo_err_node.real() if not stereo_err_node.empty() else 0.0

            fs.release()

            # Validate loaded data
            if not self.is_valid():
                print("Error: Incomplete calibration data in file")
                return False

            self.filename = filename
            return True

        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False

    def save_calibration_yml(self, filename: str) -> bool:
        """Save calibration to YAML file"""
        if not self.is_valid():
            print("Error: Cannot save invalid calibration data")
            return False

        try:
            # Ensure directory exists
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
            if not fs.isOpened():
                print(f"Error: Cannot create file: {filename}")
                return False

            fs.write("cam_K", self.cam_K)
            fs.write("cam_kc", self.cam_kc)
            fs.write("proj_K", self.proj_K)
            fs.write("proj_kc", self.proj_kc)
            fs.write("R", self.R)
            fs.write("T", self.T)
            fs.write("cam_error", self.cam_error)
            fs.write("proj_error", self.proj_error)
            fs.write("stereo_error", self.stereo_error)

            fs.release()

            self.filename = filename
            return True

        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def save_calibration_matlab(self, filename: str) -> bool:
        """Save calibration to MATLAB format"""
        if not self.is_valid():
            print("Error: Cannot save invalid calibration data")
            return False

        try:
            # Ensure directory exists
            dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            # Convert rotation matrix to Rodrigues vector
            rvec, _ = cv2.Rodrigues(self.R)

            # Flatten distortion coefficients
            cam_kc = self.cam_kc.flatten()
            proj_kc = self.proj_kc.flatten()

            # Ensure we have at least 5 distortion coefficients
            if len(cam_kc) < 5:
                cam_kc = np.pad(cam_kc, (0, 5 - len(cam_kc)))
            if len(proj_kc) < 5:
                proj_kc = np.pad(proj_kc, (0, 5 - len(proj_kc)))

            with open(filename, 'w') as fp:
                fp.write("% Projector-Camera Stereo calibration parameters:\n\n")

                # Camera intrinsic parameters
                fp.write("% Intrinsic parameters of camera:\n")
                fp.write(f"fc_left = [ {self.cam_K[0, 0]:.10f} {self.cam_K[1, 1]:.10f} ]; % Focal Length\n")
                fp.write(f"cc_left = [ {self.cam_K[0, 2]:.10f} {self.cam_K[1, 2]:.10f} ]; % Principal point\n")
                fp.write(f"alpha_c_left = [ {self.cam_K[0, 1]:.10f} ]; % Skew\n")
                fp.write(f"kc_left = [ {cam_kc[0]:.10f} {cam_kc[1]:.10f} {cam_kc[2]:.10f} {cam_kc[3]:.10f} {cam_kc[4]:.10f} ]; % Distortion\n\n")

                # Projector intrinsic parameters
                fp.write("% Intrinsic parameters of projector:\n")
                fp.write(f"fc_right = [ {self.proj_K[0, 0]:.10f} {self.proj_K[1, 1]:.10f} ]; % Focal Length\n")
                fp.write(f"cc_right = [ {self.proj_K[0, 2]:.10f} {self.proj_K[1, 2]:.10f} ]; % Principal point\n")
                fp.write(f"alpha_c_right = [ {self.proj_K[0, 1]:.10f} ]; % Skew\n")
                fp.write(f"kc_right = [ {proj_kc[0]:.10f} {proj_kc[1]:.10f} {proj_kc[2]:.10f} {proj_kc[3]:.10f} {proj_kc[4]:.10f} ]; % Distortion\n\n")

                # Extrinsic parameters
                fp.write("% Extrinsic parameters (position of projector wrt camera):\n")
                fp.write(f"om = [ {rvec[0, 0]:.10f} {rvec[1, 0]:.10f} {rvec[2, 0]:.10f} ]; % Rotation vector\n")
                fp.write(f"T = [ {self.T[0, 0]:.10f} {self.T[1, 0]:.10f} {self.T[2, 0]:.10f} ]; % Translation vector\n")

            return True

        except Exception as e:
            print(f"Error saving MATLAB calibration: {e}")
            return False

    def display(self):
        """Print calibration data to console"""
        print("=" * 50)
        print("Camera Calibration:")
        print(f"  Reprojection error: {self.cam_error:.6f}")
        print(f"  K:\n{self.cam_K}")
        print(f"  kc: {self.cam_kc.flatten() if self.cam_kc is not None else None}")
        print()
        print("Projector Calibration:")
        print(f"  Reprojection error: {self.proj_error:.6f}")
        print(f"  K:\n{self.proj_K}")
        print(f"  kc: {self.proj_kc.flatten() if self.proj_kc is not None else None}")
        print()
        print("Stereo Calibration:")
        print(f"  Reprojection error: {self.stereo_error:.6f}")
        print(f"  R:\n{self.R}")
        print(f"  T: {self.T.flatten() if self.T is not None else None}")
        print("=" * 50)

    def __repr__(self) -> str:
        return (f"CalibrationData(valid={self.is_valid()}, "
                f"cam_err={self.cam_error:.4f}, proj_err={self.proj_error:.4f}, "
                f"stereo_err={self.stereo_error:.4f})")