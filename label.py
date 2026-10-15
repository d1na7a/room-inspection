"""

Labels every blob crop with CLIP, reads outputs and output/crops/*.jpg, writes the labels back into
output or results.json

Run:  C:\\v\\Scripts\\python C:\\project\\label.py
The first run downloads the CLIP model

---
LABELS = [
	"handbag", "vase", "ornament", "paper towel roll", "bottle", "cup", "mug", "box", "book", "shoes", "clothes", "cushion", "pillow", "towel", "plant", "flowers", "lamp", "picture frame", "cable", "remove control", "phone", "bed", "duvet", "headboard", "bedside table", "drawer", "dresser", "wardrobe", "curtain", "mirror", "radiator", "wall", "floor", "edge of a bed",
]

"""
import json
import torch
import open_clip
from PIL import Image
from common import OUT


# Item names, including external scene items
LABELS = [
	"handbag", "vase", "ornament", "paper towel roll", "bottle", "cup", "mug", "box", "book", "shoes", "clothes", "cushion", "pillow", "towel", "plant", "flowers", "lamp", "picture frame", "cable", "remove control", "phone", "bed", "duvet", "headboard", "bedside table", "drawer", "dresser", "wardrobe", "curtain", "mirror", "radiator", "wall", "floor", "edge of a bed",
]

results = json.load(open(OUT / "results.json"))
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

with torch.no_grad():
    text = tokenizer([f"a photo of a {l}" for l in LABELS])
    tf = model.encode_text(text)
    tf /= tf.norm(dim=-1, keepdim=True)
    for r in results:
        path = OUT / "crops" / f"{r['id']}.jpg"
        if not r.get("photo") or not path.exists():
            r["label"], r["label_conf"] = "unknown (no clear view)", 0.0
            continue
        img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
        f = model.encode_image(img)
        f /= f.norm(dim=-1, keepdim=True)
        probs = (100 * f @ tf.T).softmax(dim=-1)[0]
        top = probs.topk(3)
        r["label"] = LABELS[int(top.indices[0])]
        r["label_conf"] = round(float(top.values[0]), 3)
        r["label_top3"] = [LABELS[int(i)] for i in top.indices]

json.dump(results, open(OUT / "results.json", "w"), indent=2)
for r in results:
    print(f"{r['id']:<12} {r.get('label','?'):<24} conf {r.get('label_conf', 0):<6}",
          r.get("label_top3", ""))
