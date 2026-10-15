"""Step 5: pictures of the verified changes (reads output/changes.json from report.py).

Writes to output/:
  panels/*.jpg      before/after photos for each change (box = where the change is)
  changes_3d.html   interactive 3D view, opens in any web browser
  changes_3d.png    3D view: grey = unchanged, red = removed / old position,
                    green = added / new position, orange arrow = movement

Run:  C:\\v\\Scripts\\python C:\\project\\viz.py     (add --view to keep the 3D window open;
      in that window press P to save a screenshot of the current view)
"""
import sys, re, json
import numpy as np
import cv2
from common import OUT, IMAGES, load_cameras, project, best_view, box_and_crop

RED, GREEN, ORANGE = [0.85, 0.1, 0.1], [0.1, 0.7, 0.1], [1.0, 0.55, 0.0]
COLOUR = {"old": RED, "removed": RED, "new": GREEN, "added": GREEN}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def roles(events):
    """Which blob plays which part in which change."""
    out = {}
    for e in events:
        if e["type"] == "moved":
            out[e["ids"][0]], out[e["ids"][1]] = ("old", e), ("new", e)
        else:
            out[e["ids"][0]] = (e["type"], e)
    return out


# ---------- before / after photo panels ----------
def view_crop(P, prefix, cams, cam, colour_bgr):
    bv = best_view(P, cams, cam, prefix, min_inside=0.8)
    if bv is None:
        return None
    _, name, uv = bv
    img = cv2.imread(str(IMAGES / name))
    if img is None:
        return None
    (x0, y0, x1, y1), (a0, b0, a1, b1) = box_and_crop(uv, cam, pad_frac=0.8, min_pad=80)
    cv2.rectangle(img, (x0, y0), (x1, y1), colour_bgr, 4)
    return img[b0:b1, a0:a1], name


def tile(view, title, w=480, h=360):
    canvas = np.full((h + 40, w, 3), 255, np.uint8)
    if view is None:
        cv2.putText(canvas, "no clear view", (20, 40 + h // 2), FONT, 0.8, (90, 90, 90), 2)
    else:
        img = view[0]
        s = min(w / img.shape[1], h / img.shape[0])
        im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
        y, x = 40 + (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
    cv2.putText(canvas, title, (10, 28), FONT, 0.75, (30, 30, 30), 2)
    return canvas


def make_panel(e, pts, cams, cam):
    if e["type"] == "moved":
        before = view_crop(pts[e["ids"][0]], "a_", cams, cam, (0, 0, 230))
        after = view_crop(pts[e["ids"][1]], "b_", cams, cam, (0, 190, 0))
    else:
        P = pts[e["ids"][0]]
        col = (0, 190, 0) if e["type"] == "added" else (0, 0, 230)
        before, after = view_crop(P, "a_", cams, cam, col), view_crop(P, "b_", cams, cam, col)
    left, right = tile(before, "BEFORE (video A)"), tile(after, "AFTER (video B)")
    row = np.hstack([left, np.full((left.shape[0], 10, 3), 255, np.uint8), right])
    cap = np.full((80, row.shape[1], 3), 255, np.uint8)
    cv2.putText(cap, e["text"], (10, 32), FONT, 0.9, (0, 0, 0), 2)
    frames = f"frames: {before[1] if before else '-'}  ->  {after[1] if after else '-'}"
    cv2.putText(cap, frames, (10, 66), FONT, 0.6, (90, 90, 90), 1)
    return np.vstack([row, cap])


# ---------- 3D helpers ----------
def rot_z_to(d):
    """Rotation that turns the +z axis into direction d (Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ d)
    if c > 0.9999:
        return np.eye(3)
    if c < -0.9999:
        return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, d)
    s = np.linalg.norm(ax)
    ax /= s
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def pick_view(centres, cams, cam):
    """Camera that sees the most changes, preferring the one furthest back."""
    best = None
    for name, R, t in cams:
        uv, z = project(centres, R, t, cam)
        inside = int(((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam["W"]) &
                      (uv[:, 1] >= 0) & (uv[:, 1] < cam["H"])).sum())
        d = float(np.linalg.norm(centres - (-R.T @ t), axis=1).mean())
        if best is None or (inside, d) > best[0]:
            best = ((inside, d), name, R, t)
    return best


def make_html(static_pts, pts, rl, events, up):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly is not installed, skipping the HTML view. Fix: C:\\v\\Scripts\\pip install plotly")
        return
    S = static_pts
    if len(S) > 80000:
        S = S[np.random.default_rng(0).choice(len(S), 80000, replace=False)]
    data = [go.Scatter3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], mode="markers", name="unchanged",
                         marker=dict(size=1.2, color="rgb(175,175,175)"), hoverinfo="skip")]
    anns = []
    names = {"old": "old position", "new": "new position", "added": "added", "removed": "removed"}
    for bid, (role, e) in rl.items():
        P = pts[bid]
        col = "rgb(215,25,25)" if COLOUR[role] is RED else "rgb(25,165,25)"
        data.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                 name=f"{e['label']} ({names[role]})", hovertext=e["text"],
                                 hoverinfo="text", marker=dict(size=2.5, color=col)))
        if role != "old":   # one floating label per change (drawn on top, never hidden by points)
            c = P.mean(0)
            anns.append(dict(x=float(c[0]), y=float(c[1]), z=float(c[2]), text=e["text"],
                             showarrow=True, arrowhead=2, ax=0, ay=-70 if len(anns) % 2 == 0 else 70,
                             bgcolor="rgba(255,255,255,0.9)", bordercolor="black",
                             font=dict(size=13, color="black")))
    for e in events:
        if e["type"] == "moved":
            c0, c1 = pts[e["ids"][0]].mean(0), pts[e["ids"][1]].mean(0)
            data.append(go.Scatter3d(x=[c0[0], c1[0]], y=[c0[1], c1[1]], z=[c0[2], c1[2]],
                                     mode="lines", name=e["text"], line=dict(color="orange", width=8)))
    hidden = dict(visible=False)
    fig = go.Figure(data)
    fig.update_layout(title="What changed in the room (drag to rotate, scroll to zoom)",
                      legend=dict(itemsizing="constant"),
                      scene=dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                                 annotations=anns,
                                 camera=dict(up=dict(x=float(up[0]), y=float(up[1]), z=float(up[2])))))
    fig.write_html(str(OUT / "changes_3d.html"), include_plotlyjs=True)
    print("wrote", OUT / "changes_3d.html")


def make_3d(static, blobs, rl, events, cam, cams, voxel, show):
    import open3d as o3d
    geoms = [static]
    for bid, (role, e) in rl.items():
        pc = blobs[bid]
        pc.paint_uniform_color(COLOUR[role])
        geoms.append(pc)
        box = pc.get_oriented_bounding_box()
        box.color = COLOUR[role]
        geoms.append(box)
    for e in events:
        if e["type"] != "moved":
            continue
        c0 = np.asarray(blobs[e["ids"][0]].points).mean(0)
        c1 = np.asarray(blobs[e["ids"][1]].points).mean(0)
        v = c1 - c0
        L = float(np.linalg.norm(v))
        r = 1.5 * voxel
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=r, cone_radius=2.5 * r,
                                                       cylinder_height=0.8 * L, cone_height=0.2 * L)
        arrow.rotate(rot_z_to(v / L), center=(0, 0, 0))
        arrow.translate(c0)
        arrow.paint_uniform_color(ORANGE)
        arrow.compute_vertex_normals()
        geoms.append(arrow)

    w, h = 1280, 720
    vis = o3d.visualization.Visualizer()
    vis.create_window("What changed in the room", w, h)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.ones(3)

    # start from the video camera that sees the most changes, pulled back a little
    centres = np.array([np.asarray(blobs[b].points).mean(0) for b in rl])
    (_, d), name, R, t = pick_view(centres, cams, cam)
    C = -R.T @ t - R[2] * 0.4 * d
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    fx = cam["f"] * w / cam["W"]
    params.intrinsic.set_intrinsics(w, h, fx, fx, (w - 1) / 2, (h - 1) / 2)
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, -R @ C
    params.extrinsic = E
    try:
        ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(params)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(OUT / "changes_3d.png"), do_render=True)
    print("wrote", OUT / "changes_3d.png", f"(view based on camera {name})")
    if show:
        vis.run()
    vis.destroy_window()


def main():
    import open3d as o3d
    events = json.load(open(OUT / "changes.json"))
    if not events:
        print("changes.json has no verified changes - run report.py first.")
        return
    meta = json.load(open(OUT / "meta.json"))
    cam, cams = load_cameras()
    rl = roles(events)
    blobs = {b: o3d.io.read_point_cloud(str(OUT / "blobs" / f"{b}.ply")) for b in rl}
    pts = {b: np.asarray(pc.points) for b, pc in blobs.items()}

    (OUT / "panels").mkdir(exist_ok=True)
    for i, e in enumerate(events, 1):
        fn = OUT / "panels" / f"{i}_{e['type']}_{re.sub(r'[^A-Za-z0-9]+', '_', e['label']).strip('_')}.jpg"
        cv2.imwrite(str(fn), make_panel(e, pts, cams, cam))
        print("wrote", fn)

    static = o3d.io.read_point_cloud(str(OUT / "static.ply"))
    up = -np.mean([R[1] for _, R, _ in cams], 0)
    up /= np.linalg.norm(up)
    make_html(np.asarray(static.points), pts, rl, events, up)
    make_3d(static, blobs, rl, events, cam, cams, meta["voxel"], "--view" in sys.argv)


if __name__ == "__main__":
    main()
