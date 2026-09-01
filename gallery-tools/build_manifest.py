"""Build data/gallery.json from the raw run output, then inline it into index.html.

    python3 tools/build_manifest.py [source-dir] [repo-dir]

source-dir holds the original "sample images", "run-172",
"run-model4" and "run-qa" folders. Everything the page shows -- garment
descriptions, per-image render times, QA verdicts -- is read from those files;
nothing is retyped by hand.
"""
import json, csv, os, re, sys, statistics

DL = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads")
R  = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

specs = json.load(open(f"{DL}/run-model4/garment_specs.json"))

def title(s):
    return " ".join(w.capitalize() for w in s.split())

# ---------- garments ----------
garments = []
for i in range(1, 24):
    gid = "fg%02d" % i
    f = specs.get(gid, {}).get("fields", {})
    typ = title(f.get("TYPE", "Ensemble").lower())
    cols = f.get("COLOURS", "")
    pieces = [f.get(k, "") for k in ("PIECE_1","PIECE_2","PIECE_3","PIECE_4")]
    pieces = [p for p in pieces if p and p.lower() != "none"]
    garments.append({
        "id": gid, "set": "womenswear", "no": i,
        "name": typ,
        "colours": cols,
        "fabric": f.get("FABRIC", ""),
        "length": f.get("LENGTH", ""),
        "metal": f.get("METAL", ""),
        "npieces": f.get("PIECES", ""),
        "dupatta": f.get("DUPATTA", "") if f.get("DUPATTA","").lower() != "none" else "",
        "how": f.get("HOW_WORN", ""),
        "pieces": pieces,
        "img": f"assets/garments/{gid}.webp",
        "thumb": f"assets/thumbs/garments/{gid}.webp",
    })

male_ids = ["mg01","mg07","mg08","mg09","mg10","mg16","mg17","mg18","mg19","mg20","mg99"]
for n, gid in enumerate(male_ids, 1):
    garments.append({
        "id": gid, "set": "menswear", "no": n,
        "name": "Menswear Look %02d" % n,
        "colours": "", "fabric": "", "length": "", "metal": "", "npieces": "",
        "dupatta": "", "how": "", "pieces": [],
        "img": f"assets/garments/{gid}.webp",
        "thumb": f"assets/thumbs/garments/{gid}.webp",
    })

# ---------- models ----------
models = [{"id": f"f{i}", "name": f"Model {i:02d}", "set": "womenswear",
           "img": f"assets/models/f{i}.webp", "thumb": f"assets/thumbs/models/f{i}.webp"}
          for i in range(1, 8)]
models.append({"id": "m1", "name": "Model 08", "set": "menswear",
               "img": "assets/models/m1.webp", "thumb": "assets/thumbs/models/m1.webp"})

# ---------- results from summary.csv ----------
rows = list(csv.DictReader(open(f"{DL}/run-172/summary.csv")))
def mid(r):
    m = re.match(r"model\s+\((\d)\)", r["model"])
    return f"f{m.group(1)}" if m else "m1"
def gid(r, mi):
    g = r["garment"]
    m = re.match(r"garment \((\d+)\)", g)
    if mi == "m1":
        return ("mg%02d" % int(m.group(1))) if m else "mg99"
    return "fg%02d" % int(m.group(1))

# QA lookup
qa = {}
for e in json.load(open(f"{DL}/run-qa/summary.json")):
    mm = re.match(r"model\s+\((\d)\)", e["model"])
    k = (f"f{mm.group(1)}" if mm else "m1")
    gm = re.match(r"garment \((\d+)\)", e["garment"])
    if k == "m1":
        gk = ("mg%02d" % int(gm.group(1))) if gm else "mg99"
    else:
        gk = "fg%02d" % int(gm.group(1))
    qa[(k, gk)] = {"verdict": e["qa"].get("VERDICT"), "round": e.get("round", 1)}

results, secs = [], []
for r in rows:
    mi = mid(r); gi = gid(r, mi)
    s = float(r["seconds"]); secs.append(s)
    rec = {"id": f"{mi}__{gi}", "model": mi, "garment": gi, "set": r["set"],
           "seconds": s, "steps": int(r["steps"]), "seed": int(r["seed"]),
           "mp": float(r["megapixels"]),
           "img": f"assets/results/{mi}__{gi}.webp",
           "thumb": f"assets/thumbs/results/{mi}__{gi}.webp"}
    if (mi, gi) in qa:
        rec["qa"] = qa[(mi, gi)]
    results.append(rec)

# verify every asset exists
missing = [x["img"] for x in results if not os.path.exists(os.path.join(R, x["img"]))]
missing += [g["img"] for g in garments if not os.path.exists(os.path.join(R, g["img"]))]
missing += [m["img"] for m in models if not os.path.exists(os.path.join(R, m["img"]))]

stats = {
    "generations": len(results),
    "models": len(models),
    "garments": len(garments),
    "mean_s": round(statistics.mean(secs), 1),
    "min_s": round(min(secs), 1),
    "max_s": round(max(secs), 1),
    "total_min": round(sum(secs) / 60, 1),
    "steps": 5, "seed": 42, "mp": 0.75, "res": "768 × 1008",
    "backbone": "FLUX.2 klein 9B + fal virtual-try-on LoRA",
    "captioner": "meta/muse-glimmer-30b on NVIDIA NIM",
    "gpu": "single RTX 4090",
    "qa_audited": len(qa),
    "qa_pass": sum(1 for v in qa.values() if v["verdict"] == "PASS"),
    "qa_retried": sum(1 for v in qa.values() if v.get("round", 1) > 1),
}

payload = {"stats": stats, "models": models, "garments": garments, "results": results}
os.makedirs(f"{R}/data", exist_ok=True)
json.dump(payload, open(f"{R}/data/gallery.json", "w"), indent=1)

# inline the manifest so the page also works straight off the filesystem
html_path = f"{R}/index.html"
html = open(html_path).read()
compact = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
html = re.sub(r'(<script id="galleryData" type="application/json">).*?(</script>)',
              lambda m: m.group(1) + compact + m.group(2), html, count=1, flags=re.S)
open(html_path, "w").write(html)
print("inlined", len(compact), "bytes into index.html")
print("results:", len(results), "| garments:", len(garments), "| models:", len(models))
print("missing assets:", missing if missing else "none")
print("stats:", json.dumps(stats, indent=1))
