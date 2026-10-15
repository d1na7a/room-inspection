"""
Compares the two dense point clouds (video A = before and video B = after)

Finds ADDED and REMOVED blobs, finding the best photo for each and writes to the output:
results.json - one record per blob
blobs/*.ply - the points of each blob
crops/*.jpg - cropped photo of each blob used by label.py
annotated/*.jpg - full photo with a box around each blob
result.ply - grey = unchanged, green = added and red = removed for all detections
meta.json, a_ds.ply, b_ds.ply - used by report.py

Run:  C:\\v\\Scripts\\python C:\\project\\detect.py (add --view to open 3D viewer)

"""
import sys, json
import numpy as np
import open3d as o3d
import cv2
from common import OUT, CLOUDS, IMAGES, load_cameras, best_view, box_and_crop

for d in ("crops", "annotated", "blobs"):
    (OUT / d).mkdir(parents=True, exist_ok=True)

# Settings (all relative to scene size, because COLMAP's units are arbitrary)
VOXEL_DIV = 250    # voxel = scene diagonal / 250
THR_MULT = 2       # a point is "changed" if it is > 2 voxels from the other cloud
EPS_MULT = 3       # DBSCAN neighbourhood = 3 voxels
MIN_POINTS = 15    # smallest blob kept


def clean(path):
    p = o3d.io.read_point_cloud(str(path))
    p, _ = p.remove_statistical_outlier(20, 2.0)
    return p


A, B = clean(CLOUDS / "a.ply"), clean(CLOUDS / "b.ply")
print("points:", len(A.points), len(B.points))
diag = np.linalg.norm(A.get_axis_aligned_bounding_box().get_extent())
voxel = diag / VOXEL_DIV
A, B = A.voxel_down_sample(voxel), B.voxel_down_sample(voxel)
thr = THR_MULT * voxel

json.dump({"diag": float(diag), "voxel": float(voxel), "thr": float(thr)},
          open(OUT / "meta.json", "w"), indent=2)
o3d.io.write_point_cloud(str(OUT / "a_ds.ply"), A)
o3d.io.write_point_cloud(str(OUT / "b_ds.ply"), B)


def far_idx(src, ref):
    d = np.asarray(src.compute_point_cloud_distance(ref))
    return np.where(d > thr)[0]


def clusters(pc):
    if len(pc.points) == 0:
        return []
    lab = np.array(pc.cluster_dbscan(eps=EPS_MULT * voxel, min_points=MIN_POINTS))
    if lab.size == 0 or lab.max() < 0:
        return []
    return [pc.select_by_index(np.where(lab == l)[0]) for l in range(lab.max() + 1)]


ia, ir = far_idx(B, A), far_idx(A, B)
static = B.select_by_index(ia, invert=True)
static.paint_uniform_color([0.7, 0.7, 0.7])
added = clusters(B.select_by_index(ia))
removed = clusters(A.select_by_index(ir))

cam, cams = load_cameras()
print("camera model:", cam["model"])


def obb_extent(pc):
    """Oriented box size (follows the object, unlike an axis-aligned box). Largest first."""
    try:
        return sorted((float(x) for x in pc.get_oriented_bounding_box().extent), reverse=True)
    except RuntimeError:   # too few or flat points
        return sorted((float(x) for x in pc.get_axis_aligned_bounding_box().get_extent()), reverse=True)


results = []


def process(kind, blobs, prefix, color):
    for i, b in enumerate(blobs):
        b.paint_uniform_color(color)
        P = np.asarray(b.points)
        bb = b.get_axis_aligned_bounding_box()
        rec = {
            "id": f"{kind}_{i}",
            "kind": kind,
            "centre": [round(float(x), 3) for x in bb.get_center()],
            "size_pct_of_scene": [round(float(x), 1) for x in 100 * bb.get_extent() / diag],
            "obb_extent": [round(x, 4) for x in obb_extent(b)],
            "n_points": int(len(P)),
        }
        o3d.io.write_point_cloud(str(OUT / "blobs" / f"{rec['id']}.ply"), b)
        bv = best_view(P, cams, cam, prefix)
        if bv is None:
            rec["photo"] = None
        else:
            _, name, uv = bv
            img = cv2.imread(str(IMAGES / name))
            (x0, y0, x1, y1), (a0, b0, a1, b1) = box_and_crop(uv, cam)
            cv2.imwrite(str(OUT / "crops" / f"{rec['id']}.jpg"), img[b0:b1, a0:a1])
            ann = img.copy()
            box_col = (0, 255, 0) if kind == "ADDED" else (0, 0, 255)  # BGR
            cv2.rectangle(ann, (x0, y0), (x1, y1), box_col, 4)
            cv2.imwrite(str(OUT / "annotated" / f"{rec['id']}.jpg"), ann)
            rec["photo"] = name
            rec["box"] = [x0, y0, x1, y1]
        results.append(rec)


process("ADDED", added, "b_", [0, 1, 0])
process("REMOVED", removed, "a_", [1, 0, 0])

json.dump(results, open(OUT / "results.json", "w"), indent=2)
o3d.io.write_point_cloud(str(OUT / "static.ply"), static)
scene = o3d.geometry.PointCloud(static)
for b in added + removed:
    scene += b
o3d.io.write_point_cloud(str(OUT / "result.ply"), scene)

for r in results:
    print(r["id"], "centre", r["centre"], "size%", r["size_pct_of_scene"],
          "pts", r["n_points"], "photo", r["photo"])
print(f"\nWrote {len(results)} blobs to {OUT}")

if "--view" in sys.argv:
    o3d.visualization.draw_geometries([scene])
