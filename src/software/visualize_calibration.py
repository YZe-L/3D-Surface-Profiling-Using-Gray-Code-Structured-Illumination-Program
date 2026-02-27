"""
visualize_calibration.py  (v2)

Visualizes camera + projector extrinsics and per-pose reprojection errors.
Style matches MATLAB's showExtrinsics / showReprojectionErrors.

Usage:
    1. Add save_pose_data() call to main.py (see bottom of this file)
    2. Re-run calibration to generate calibration_poses.npz
    3. Run:  python visualize_calibration.py --npz calibration_poses.npz

Interactive 3D plots: drag to rotate, scroll to zoom, then close window when done.
Figures are saved on close, so rotate to a nice angle first.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')  # interactive backend; change to 'Qt5Agg' if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os


# ============ Board defaults (match your setup) ============
BOARD_CORNERS = (6, 9)
SQUARE_SIZE = 24.0


# ============ Helpers ============
def rvec_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def board_rect(board_corners, square_size):
    c, r = board_corners
    return np.array([
        [0, 0, 0],
        [(c-1)*square_size, 0, 0],
        [(c-1)*square_size, (r-1)*square_size, 0],
        [0, (r-1)*square_size, 0],
    ], dtype=np.float64)


def xform(pts, R, t):
    return (R @ pts.T + np.asarray(t).reshape(3, 1)).T


# ============ Drawing primitives ============
def draw_camera(ax, pos, R_axes, scale, color, label):
    """Wireframe camera/projector icon with RGB axes."""
    s = scale
    p = np.asarray(pos).flatten()

    back  = np.array([[-1,-.7,0],[1,-.7,0],[1,.7,0],[-1,.7,0]]) * s * 0.4
    front = np.array([[-.6,-.4,1],[.6,-.4,1],[.6,.4,1],[-.6,.4,1]]) * s * 0.4

    bw = (R_axes @ back.T).T + p
    fw = (R_axes @ front.T).T + p

    for i in range(4):
        j = (i + 1) % 4
        ax.plot3D(*zip(bw[i], bw[j]), color=color, lw=1.8)
        ax.plot3D(*zip(fw[i], fw[j]), color=color, lw=1.8)
        ax.plot3D(*zip(bw[i], fw[i]), color=color, lw=1.8)

    al = s * 0.5
    for a, c in enumerate(['r', 'g', 'b']):
        end = p + R_axes[:, a] * al
        ax.plot3D([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color=c, lw=2)

    ax.text(p[0], p[1] - s * 0.55, p[2], label,
            fontsize=11, fontweight='bold', color=color, ha='center')


def draw_board(ax, corners_w, color, alpha=0.25, label=None):
    poly = Poly3DCollection([corners_w.tolist()], alpha=alpha,
                            facecolor=color, edgecolor=color, lw=1.5)
    ax.add_collection3d(poly)
    if label:
        c = corners_w.mean(axis=0)
        ax.text(c[0], c[1], c[2], label, fontsize=13, fontweight='bold',
                color=color, ha='center')


# ============ Reprojection errors (MATLAB style, one figure per device) ============
def compute_errors(world_pts, img_pts, K, kc, rvecs, tvecs):
    errs = []
    for i in range(len(rvecs)):
        proj, _ = cv2.projectPoints(world_pts[i], rvecs[i], tvecs[i], K, kc)
        obs = np.asarray(img_pts[i]).reshape(-1, 2)
        errs.append(np.mean(np.linalg.norm(proj.reshape(-1, 2) - obs, axis=1)))
    return errs


def plot_error_bar(errors, title_str, cmap_name, save_path=None):
    n = len(errors)
    mu = np.mean(errors)
    mx = np.max(errors)
    x = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.get_cmap(cmap_name)
    cols = [cmap(0.4 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

    ax.bar(x, errors, width=0.55, color=cols, edgecolor='none')
    ax.axhline(mu, color='#DAA520', ls='--', lw=2,
               label=f'Overall Mean Error: {mu:.2f} pixels')
    ax.axhline(mx, color='red', lw=1.5)
    ax.text(n + 0.3, mx, f'{mx:.2f}', color='red', fontsize=12,
            fontweight='bold', va='center')

    ax.set_xlabel('Images', fontsize=12)
    ax.set_ylabel('Mean Error in Pixels', fontsize=12)
    ax.set_title(title_str, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_ylim(0, mx * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


# ============ Interactive 3D extrinsics ============
def plot_extrinsics(cam_rvecs, cam_tvecs, proj_rvecs, proj_tvecs,
                    R_st, T_st, board_corners, square_size, save_dir='.'):
    """
    Two separate interactive figures:
      - Camera-centric (camera at origin)
      - Projector-centric (projector at origin)
    Rotate freely, then close to continue.
    """
    bl = board_rect(board_corners, square_size)
    n = len(cam_rvecs)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))

    # ---- Figure 1: Camera-centric ----
    fig1 = plt.figure('Camera-Centric Extrinsics', figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('Camera-Centric Extrinsics', fontsize=14, pad=15)

    draw_camera(ax1, np.zeros(3), np.eye(3), 50, 'blue', 'Camera')

    # Projector in camera frame:
    #   Stereo convention: X_proj = R @ X_cam + T
    #   Projector origin in camera frame: solve 0 = R @ x + T  =>  x = -R^T T
    proj_pos = -R_st.T @ T_st.flatten()
    proj_axes = R_st.T  # projector axes expressed in camera frame
    draw_camera(ax1, proj_pos, proj_axes, 50, 'red', 'Projector')

    for i in range(n):
        Ri = rvec_to_R(cam_rvecs[i])
        ti = cam_tvecs[i].flatten()
        draw_board(ax1, xform(bl, Ri, ti), colors[i], label=str(i + 1))

    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    ax1.view_init(elev=25, azim=-135)

    # Save on close callback
    def on_close_1(event):
        fig1.savefig(os.path.join(save_dir, 'extrinsics_camera_centric.png'),
                     dpi=200, bbox_inches='tight')
        print("Saved camera-centric extrinsics on close.")
    fig1.canvas.mpl_connect('close_event', on_close_1)

    # ---- Figure 2: Projector-centric ----
    fig2 = plt.figure('Projector-Centric Extrinsics', figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.set_title('Projector-Centric Extrinsics', fontsize=14, pad=15)

    draw_camera(ax2, np.zeros(3), np.eye(3), 50, 'red', 'Projector')

    # Camera in projector frame:
    #   X_proj = R @ X_cam + T  =>  camera origin (X_cam = 0) in proj frame = T
    #   Camera axes in projector frame: R @ I = R
    cam_pos = T_st.flatten()
    cam_axes = R_st
    draw_camera(ax2, cam_pos, cam_axes, 50, 'blue', 'Camera')

    for i in range(n):
        Ri = rvec_to_R(proj_rvecs[i])
        ti = proj_tvecs[i].flatten()
        draw_board(ax2, xform(bl, Ri, ti), colors[i], label=str(i + 1))

    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    ax2.view_init(elev=25, azim=-135)

    def on_close_2(event):
        fig2.savefig(os.path.join(save_dir, 'extrinsics_projector_centric.png'),
                     dpi=200, bbox_inches='tight')
        print("Saved projector-centric extrinsics on close.")
    fig2.canvas.mpl_connect('close_event', on_close_2)

    plt.show()  # blocking — rotate freely, close when done


# ============ Save / Load ============
def save_pose_data(filepath, cam_rvecs, cam_tvecs, proj_rvecs, proj_tvecs,
                   cam_K, cam_kc, proj_K, proj_kc, R, T,
                   world_pts=None, cam_pts=None, proj_pts=None):
    """Call this from main.py right after stereoCalibrate."""
    data = {
        'cam_rvecs':  np.array([np.asarray(r).flatten() for r in cam_rvecs]),
        'cam_tvecs':  np.array([np.asarray(t).flatten() for t in cam_tvecs]),
        'proj_rvecs': np.array([np.asarray(r).flatten() for r in proj_rvecs]),
        'proj_tvecs': np.array([np.asarray(t).flatten() for t in proj_tvecs]),
        'cam_K': cam_K, 'cam_kc': cam_kc,
        'proj_K': proj_K, 'proj_kc': proj_kc,
        'R': R, 'T': T,
    }
    for name, pts in [('world_pts', world_pts),
                      ('cam_pts', cam_pts),
                      ('proj_pts', proj_pts)]:
        if pts is not None:
            arr = np.empty(len(pts), dtype=object)
            for i, p in enumerate(pts):
                arr[i] = np.asarray(p)
            data[name] = arr
    np.savez(filepath, **data)
    print(f"Saved pose data: {filepath}")


def load_pose_data(filepath):
    d = np.load(filepath, allow_pickle=True)
    result = {
        'cam_rvecs':  [r.reshape(3, 1) for r in d['cam_rvecs']],
        'cam_tvecs':  [t.reshape(3, 1) for t in d['cam_tvecs']],
        'proj_rvecs': [r.reshape(3, 1) for r in d['proj_rvecs']],
        'proj_tvecs': [t.reshape(3, 1) for t in d['proj_tvecs']],
        'cam_K': d['cam_K'], 'cam_kc': d['cam_kc'],
        'proj_K': d['proj_K'], 'proj_kc': d['proj_kc'],
        'R': d['R'], 'T': d['T'],
    }
    for name in ['world_pts', 'cam_pts', 'proj_pts']:
        if name in d:
            raw = d[name]
            result[name] = [np.asarray(p, dtype=np.float32) for p in raw]
        else:
            result[name] = None
    return result


# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description="Calibration Visualization v2")
    parser.add_argument('--npz', required=True, help='calibration_poses.npz')
    parser.add_argument('--save-dir', default='.', help='Output directory')
    parser.add_argument('--board', default='6,9', help='cols,rows')
    parser.add_argument('--square', type=float, default=24.0, help='mm')
    args = parser.parse_args()

    board = tuple(map(int, args.board.split(',')))
    d = load_pose_data(args.npz)
    os.makedirs(args.save_dir, exist_ok=True)

    # Reprojection error bar charts (separate, MATLAB style)
    if d['world_pts'] is not None and d['cam_pts'] is not None:
        cam_err = compute_errors(d['world_pts'], d['cam_pts'],
                                 d['cam_K'], d['cam_kc'],
                                 d['cam_rvecs'], d['cam_tvecs'])
        proj_err = compute_errors(d['world_pts'], d['proj_pts'],
                                  d['proj_K'], d['proj_kc'],
                                  d['proj_rvecs'], d['proj_tvecs'])

        plot_error_bar(cam_err,
                       'Camera: Mean Reprojection Error per Image', 'Blues',
                       os.path.join(args.save_dir, 'cam_reproj_error.png'))
        plot_error_bar(proj_err,
                       'Projector: Mean Reprojection Error per Image', 'Reds',
                       os.path.join(args.save_dir, 'proj_reproj_error.png'))
    else:
        print("No point data in .npz — skipping reprojection error plots.")

    # Interactive 3D extrinsics (rotate freely, saves on close)
    plot_extrinsics(d['cam_rvecs'], d['cam_tvecs'],
                    d['proj_rvecs'], d['proj_tvecs'],
                    d['R'], d['T'], board, args.square, args.save_dir)


if __name__ == '__main__':
    main()


# ==================================================================
#  HOW TO INTEGRATE WITH main.py
# ==================================================================
#  In main.py, after line 487 (after stereo_error print), add:
#
#      from visualize_calibration import save_pose_data
#      save_pose_data(
#          os.path.join(root_dir, "calibration_poses.npz"),
#          cam_rvecs, cam_tvecs, proj_rvecs, proj_tvecs,
#          cam_K, cam_kc, proj_K, proj_kc, R, T,
#          world_corners_all, camera_corners_all, projector_corners_all
#      )
#
#  Then re-run:  python main.py calibrate ./calibration_data
#  Then run:     python visualize_calibration.py --npz ./calibration_data/calibration_poses.npz
# ==================================================================