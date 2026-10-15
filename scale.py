"""

Convert COLMAP's arbitrary units into centimetres

Click two ends of a length you measured with a tape, this script will work out centimeter
per cloud unit and saves it to the output or scale.json
Run:  C:\\v\\Scripts\\python C:\\project\\scale.py
In the window: Shift + left-click the two ends, then press Q

"""
import json
import numpy as np
import open3d as o3d
from common import OUT

KNOWN_CM = 145.0   # width of foot of the bed

pcd = o3d.io.read_point_cloud(str(OUT / "result.ply"))
print("Shift+click the two ends of the known length, then press Q")
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Shift+click the 2 ends of the known length, then press Q")
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

idx = vis.get_picked_points()
if len(idx) != 2:
	raise SystemExit(f"Need exactly 2 picked points, got {len(idx)}, please run again.")

P = np.asarray(pcd.points)[idx]
d = float(np.linalg.norm(P[0] - P[1]))
cm_per_unit = KNOWN_CM / d
diag_cm = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent())) * cm_per_unit

json.dump({"known_cm": KNOWN_CM, "cloud_units": d, "cm_per_unit": cm_per_unit},
          open(OUT / "scale.json", "w"), indent=2)
print(f"picked length = {d:.3f} cloud units")
print(f"scale         = {cm_per_unit:.2f} cm per unit")
print(f"scene diagonal ~ {diag_cm:.0f} cm  (check: should be a few metres)")
