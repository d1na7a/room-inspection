"""
Project paths, COLMAP cameras and projecting, all scripts read paths from this file, to run a project
from another folder, set the environment variable ROOM_ROOT. example: set ROOM_ROOT=D:\\room
"""
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("ROOM_ROOT", r"C:\project"))
OUT = ROOT / "output"
SPARSE = ROOT / "ws" / "sparse" / "0"   #COLMAP model expored as text
IMAGES = ROOT / "images"                #frames a_XXXX.jpg = video A (before), b_XXXX.jpg = video B (after)
CLOUDS = ROOT / "clouds"                #dense clouds of each video, a.ply and b.ply


def quat_to_R(w, x, y, z):
    """COLMAP quaternion (world -> camera) to a rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_cameras(sparse=SPARSE):
    """returns (cam, cams): shared intrinsics and a list of (image name, R, t)."""
    for line in open(sparse / "cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        model, W, H = p[1], int(p[2]), int(p[3])
        prm = [float(x) for x in p[4:]]
        break
    cam = dict(model=model, W=W, H=H, f=prm[0], cx=prm[1], cy=prm[2],
               k=prm[3] if len(prm) > 3 else 0.0)
    lines = [l for l in open(sparse / "images.txt") if not l.startswith("#")]
    cams = []
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        cams.append((p[9], quat_to_R(*[float(x) for x in p[1:5]]), np.array(p[5:8], float)))
    return cam, cams


def project(P, R, t, cam):
    """3D points (N,3) -> pixel coordinates (N,2) and depth (N,), with radial distortion"""
    Xc = P @ R.T + t
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    d = 1 + cam["k"] * (x * x + y * y)
    return np.stack([cam["f"] * x * d + cam["cx"], cam["f"] * y * d + cam["cy"]], 1), z


def best_view(P, cams, cam, prefix, min_inside=0.9):
    """Nearest camera of one video (prefix 'a_' or 'b_') seeing the whole blob. Returns (distance, image name, pixel coords) or 'None'"""
    c = P.mean(0)
    best = None
    for name, R, t in cams:
        if not name.startswith(prefix):
            continue
        uv, z = project(P, R, t, cam)
        if (z <= 0).any():
            continue
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < cam["W"]) & (uv[:, 1] >= 0) & (uv[:, 1] < cam["H"])
        if ok.mean() < min_inside:
            continue
        dist = np.linalg.norm(-R.T @ t - c)
        if best is None or dist < best[0]:
            best = (dist, name, uv)
    return best


def box_and_crop(uv, cam, pad_frac=0.25, min_pad=20):
    """Tight box around projected points, plus a padded crop window (both clipped to the image)"""
    W, H = cam["W"], cam["H"]
    x0, y0 = np.clip(uv.min(0), 0, [W - 1, H - 1]).astype(int)
    x1, y1 = np.clip(uv.max(0), 0, [W - 1, H - 1]).astype(int)
    pad = max(min_pad, int(pad_frac * max(x1 - x0, y1 - y0)))
    crop = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    return (int(x0), int(y0), int(x1), int(y1)), crop
