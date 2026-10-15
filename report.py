"""

Verifies blobs, pairs moved objects and report write-up

Order: detect -> label -> scale (once) -> report -> viz
Output is written in plain english in report.txt and changes.json for viz.py

For all the blobs the script checks if the other video checks it thoroughly, ray casting from each camera is used of the other video through the blobs
Free - Ray passes through the spot
____ - other video saw the spit without the object
occupied - other video has geometry within the spot
blocked - something is in front of the spot
unknown - nothing reconstructed

Blob is reported if enough cameras say "free"

Run: C:\\v\\Scripts\\python C:\\project\\report.py

"""
import json
import numpy as np
from common import OUT, SPARSE, load_cameras

# Scene things that show up as false alarms (surfaces, not movable objects)
STRUCTURE = {"wall", "floor", "curtain", "headboard", "edge of a bed", "bed", "duvet",
             "mirror", "wardrobe", "drawer", "dresser", "bedside table", "radiator"}
SYNONYMS = [{"vase", "ornament"}, {"cup", "mug"}, {"cushion", "pillow"}]

MIN_VIEWS = 3      # cameras that must have a clear look at the spot
FRAC = 0.6         # share of those that must say "free" (or "occupied")
MAX_CAMS = 40      # cameras tested per blob (evenly spaced, for speed)
K_PTS = 5          # points needed along a ray to count as "something there"
SIZE_RATIO = 3.0   # a moved object may look up to 3x bigger/smaller: boxes are axis-aligned
                   # in COLMAP's tilted frame, so sizes are inflated by different amounts


def robust_size(P):
    """Largest dimension of a blob along its own main axes, ignoring the outer 2% of
    points on each side (so a few stray points cannot stretch the size)."""
    c = P - P.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    lo, hi = np.percentile(c @ Vt.T, [2, 98], axis=0)
    return float((hi - lo).max())


def observe(centre, margin, tube, other, cams, cam):
    """Cast rays from each camera through centre and classifies what the other cloud shows."""
    usable = []
    for name, R, t in cams:
        Xc = R @ centre + t
        if Xc[2] <= 0:
            continue
        x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
        d = 1 + cam["k"] * (x * x + y * y)
        u, v = cam["f"] * x * d + cam["cx"], cam["f"] * y * d + cam["cy"]
        if 0 <= u < cam["W"] and 0 <= v < cam["H"]:
            usable.append((name, R, t))
    if len(usable) > MAX_CAMS:
        idx = np.linspace(0, len(usable) - 1, MAX_CAMS).astype(int)
        usable = [usable[i] for i in idx]

    counts = dict(free=0, occupied=0, blocked=0, unknown=0)
    hits = []   #where blocked the occupied rays hit geometry (used to find what covers a spot)
    for name, R, t in usable:
        C = (-R.T @ t).astype(np.float32)
        ray = (centre - C).astype(np.float32)
        dist = float(np.linalg.norm(ray))
        ray /= dist
        dvec = other - C
        tt = dvec @ ray
        perp2 = np.einsum("ij,ij->i", dvec, dvec) - tt * tt
        m = (perp2 < tube * tube) & (tt > 0)
        tn, pn = tt[m], other[m]
        front = tn < dist - margin
        at = np.abs(tn - dist) <= margin
        behind = tn > dist + margin
        if front.sum() >= K_PTS:
            counts["blocked"] += 1
            hits.append(np.median(pn[front], 0))
        elif at.sum() >= K_PTS:
            counts["occupied"] += 1
            hits.append(np.median(pn[at], 0))
        elif behind.sum() >= K_PTS:
            counts["free"] += 1
        else:
            counts["unknown"] += 1
    counts["views"] = len(usable)
    return counts, hits


def decide(c):
    viewed = c["free"] + c["occupied"]
    if viewed >= MIN_VIEWS and c["free"] / viewed >= FRAC:
        return "confirmed"
    if viewed >= MIN_VIEWS and c["occupied"] / viewed > FRAC:
        return "disputed"
    return "unobserved"


def bbox(r, diag, pad):
    ext = np.array(r["size_pct_of_scene"]) / 100 * diag
    c = np.array(r["centre"])
    return c - ext / 2 - pad, c + ext / 2 + pad


def owner(hits, blobs, diag, pad):
    """If most blocked/occupied rays hit one blob from the other video, return it."""
    if len(hits) < MIN_VIEWS:
        return None
    best, best_n = None, 0
    for b in blobs:
        lo, hi = bbox(b, diag, pad)
        n = sum(1 for h in hits if np.all((h >= lo) & (h <= hi)))
        if n > best_n:
            best, best_n = b, n
    return best if best_n >= FRAC * len(hits) else None


def is_object(r):
    return r["label"] not in STRUCTURE and r.get("label_conf", 0) >= 0.4


def same_label(a, b):
    if a == b:
        return True
    return any(a in g and b in g for g in SYNONYMS)


def pair_moves(changes, scale):
    rem = [r for r in changes if r["kind"] == "REMOVED"]
    add = [a for a in changes if a["kind"] == "ADDED" and a["verdict"] == "CHANGE"]
    cands = []
    for r in rem:
        for a in add:
            if not same_label(r["label"], a["label"]):
                continue
            ratio = r["size_cm"] / max(a["size_cm"], 1e-6)
            if not 1 / SIZE_RATIO <= ratio <= SIZE_RATIO:
                print(f"not paired: {r['id']} ({r['size_cm']} cm) vs {a['id']} "
                      f"({a['size_cm']} cm), size ratio {ratio:.2f}")
                continue
            d = float(np.linalg.norm(np.array(r["centre"]) - np.array(a["centre"]))) * scale
            cands.append((d, r, a))
    cands.sort(key=lambda x: x[0])
    used, pairs = set(), []
    for d, r, a in cands:
        if r["id"] in used or a["id"] in used:
            continue
        used.update([r["id"], a["id"]])
        r["moved_to"], a["moved_from"] = a["id"], r["id"]
        pairs.append((d, r, a))
    return pairs


def main():
    import open3d as o3d
    results = json.load(open(OUT / "results.json"))
    scale = json.load(open(OUT / "scale.json"))["cm_per_unit"]
    meta = json.load(open(OUT / "meta.json"))
    diag, voxel = meta["diag"], meta["voxel"]
    cam, cams = load_cameras(SPARSE)
    clouds = {p: np.asarray(o3d.io.read_point_cloud(str(OUT / f"{p}ds.ply")).points,
                            dtype=np.float32) for p in ("a_", "b_")}

    hits = {}
    for r in results:
        for key in ("replaces", "moved_to", "moved_from", "covered_by"):
            r.pop(key, None)   # clear results from a previous run
        ext = np.array(r["size_pct_of_scene"]) / 100 * diag
        blob_file = OUT / "blobs" / f"{r['id']}.ply"
        if blob_file.exists():
            size_u = robust_size(np.asarray(o3d.io.read_point_cloud(str(blob_file)).points))
        elif r.get("obb_extent"):
            size_u = max(r["obb_extent"])
        else:
            size_u = float(ext.max())
        r["size_cm"] = round(float(size_u * scale), 1)
        r["label"] = r.get("label", "unknown")
        # an ADDED blob lives in video B, so check video A (and vice versa)
        other_p = "a_" if r["kind"] == "ADDED" else "b_"
        margin = max(2 * voxel, 0.35 * float(ext.max()))
        c, h = observe(np.array(r["centre"]), margin, 1.5 * voxel, clouds[other_p],
                       [x for x in cams if x[0].startswith(other_p)], cam)
        r["observe"], r["status"], hits[r["id"]] = c, decide(c), h

    added = [r for r in results if r["kind"] == "ADDED"]
    removed = [r for r in results if r["kind"] == "REMOVED"]
    byid = {r["id"]: r for r in results}
    for r in results:
        opp = removed if r["kind"] == "ADDED" else added
        o = None if r["status"] == "confirmed" else owner(hits[r["id"]], opp, diag, 2 * voxel)
        r["covered_by"] = o["id"] if o else None

    # ADDED first: a new object standing where a removed object used to be is a real change
    for r in added:
        if r["status"] == "confirmed":
            r["verdict"] = "structure (ignored)" if r["label"] in STRUCTURE else "CHANGE"
        elif r["covered_by"] and is_object(r) and is_object(byid[r["covered_by"]]):
            r["verdict"], r["replaces"] = "CHANGE", r["covered_by"]
        else:
            r["verdict"] = r["status"]
    # REMOVED: if a verified new object now hides the spot, we cannot see it -> "covered"
    for r in removed:
        if r["status"] == "confirmed":
            r["verdict"] = "structure (ignored)" if r["label"] in STRUCTURE else "CHANGE"
        elif r["covered_by"] and is_object(r) and byid[r["covered_by"]]["verdict"] == "CHANGE":
            r["verdict"] = "covered"
        else:
            r["verdict"] = r["status"]

    cands = [r for r in results if r["verdict"] in ("CHANGE", "covered")]
    pairs = pair_moves(cands, scale)
    moved_ids = {r["id"] for _, r, a in pairs} | {a["id"] for _, r, a in pairs}

    lines = ["WHAT CHANGED IN THE ROOM",
             "Video A (before) vs video B (after). Distances are approximate: the scale comes "
             "from one tape measurement.", ""]
    events, uncertain, ev = 0, [], []
    for d, r, a in pairs:
        lab = a["label"] if a["label_conf"] >= r["label_conf"] else r["label"]
        extra = ""
        if r["verdict"] == "covered":
            extra = f"; its old spot is now hidden behind the {byid[r['covered_by']]['label']}"
        lines.append(f"- {lab} moved about {d:.0f} cm (before: {r['photo']}, after: {a['photo']}){extra}")
        ev.append(dict(type="moved", label=lab, ids=[r["id"], a["id"]], distance_cm=round(d),
                       text=f"{lab} moved about {d:.0f} cm"))
        events += 1
    for r in cands:
        if r["id"] in moved_ids:
            continue
        if r["verdict"] == "covered":
            uncertain.append(f"- {r['label']} may have been removed, but in video B its spot is "
                             f"hidden behind the {byid[r['covered_by']]['label']} ({r['photo']})")
            continue
        lab = r["label"] if r["label_conf"] >= 0.4 else f"unidentified object ({r['label']}?)"
        verb = "added" if r["kind"] == "ADDED" else "removed"
        where = f", where the {byid[r['replaces']]['label']} used to be" if r.get("replaces") else ""
        lines.append(f"- {lab} {verb}, about {r['size_cm']:.0f} cm across{where} ({r['photo']})")
        ev.append(dict(type=verb, label=lab, ids=[r["id"]], size_cm=round(r["size_cm"]),
                       text=f"{lab} {verb}{where}"))
        events += 1
    if uncertain:
        lines += ["", "Uncertain (not counted):"] + uncertain
    n_by = {v: sum(1 for r in results if r["verdict"] == v)
            for v in ("disputed", "unobserved", "structure (ignored)")}
    lines += ["", f"{len(results)} candidate blobs -> {events} reported changes.",
              f"Rejected: {n_by['disputed']} disputed (other video has geometry there), "
              f"{n_by['unobserved']} unobserved (occluded or not reconstructed), "
              f"{n_by['structure (ignored)']} scene surfaces."]
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    json.dump(ev, open(OUT / "changes.json", "w"), indent=2)
    json.dump(results, open(OUT / "results.json", "w"), indent=2)

    print(f"{'id':<12}{'label':<18}{'conf':<7}{'free':<6}{'occ':<5}{'blk':<5}{'unk':<5}"
          f"{'covered by':<13}verdict")
    for r in results:
        c = r["observe"]
        print(f"{r['id']:<12}{r['label'][:17]:<18}{r.get('label_conf', 0):<7}{c['free']:<6}"
              f"{c['occupied']:<5}{c['blocked']:<5}{c['unknown']:<5}"
              f"{str(r['covered_by'] or '-'):<13}{r['verdict']}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
