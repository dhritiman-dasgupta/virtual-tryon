#!/usr/bin/env python
"""Build the model + garment = output gallery, offline and on S3.

Two files come out of one template:

    <out>/gallery.html        every image embedded as a data URI, so the page
                              keeps working when the presigned links lapse
    s3://<bucket>/<prefix>/   the same page with images served from S3

Artifacts cannot be used for this — their CSP blocks external images, and
embedding 23 full-size PNGs blows the size limit. Hence S3 plus a local copy.

usage:
    build_gallery.py --images ./runs/4090/BEST \
                     --report ./runs/4090/outputs_round4/f6_report.json \
                     --crops  ./runs/4090/cache/crops/round4 \
                     --model  "./samples/inputs/female models/model  (6).jpeg"
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

WEEK = 604800

CSS = """
:root{--bg:#0f1115;--card:#171a20;--line:#252a33;--tx:#e8ecf2;--dim:#9aa3b2;
      --accent:#6ea8fe}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:28px 34px 20px;border-bottom:1px solid var(--line);
  display:flex;align-items:baseline;gap:18px;flex-wrap:wrap}
h1{margin:0;font-size:22px;letter-spacing:-.01em}
.meta{color:var(--dim);font-size:13px}
main{padding:22px 34px 10px}
.row{background:var(--card);border:1px solid var(--line);border-radius:12px;
  margin-bottom:14px;overflow:hidden}
.hdr{display:flex;align-items:center;gap:12px;padding:10px 16px;
  border-bottom:1px solid var(--line)}
.hdr b{font-size:15px}
.desc{color:var(--dim);font-size:13px}
.eq{display:grid;grid-template-columns:1fr 18px 1fr 18px 1fr;align-items:center;
  gap:1px;background:var(--line)}
.eq .op{background:var(--card);color:var(--dim);text-align:center;font-size:15px}
.cell{background:#0b0d10;position:relative}
.cell img{width:100%;aspect-ratio:3/4;object-fit:contain;display:block}
.cap{position:absolute;top:0;left:0;right:0;padding:5px 9px;font-size:10px;
  letter-spacing:.07em;color:#dfe4ec;background:linear-gradient(#000b,#0000)}
a{color:inherit;text-decoration:none}
@media(max-width:820px){.eq{grid-template-columns:1fr}.eq .op{display:none}}
"""


def data_uri(path: pathlib.Path, long_edge: int = 760) -> str:
    """Downscaled JPEG data URI. Keeps a 23-row page around 12 MB, not 100."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((long_edge, long_edge), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def page(order, recs, cells_for, subtitle) -> str:
    p = ["<!doctype html><html><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>Try-On</title>", f"<style>{CSS}</style></head><body>",
         "<header><h1>Virtual try-on</h1>",
         f"<span class='meta'>{subtitle}</span></header><main>"]
    for g in order:
        r = recs[g]
        p.append("<div class='row'>")
        p.append(f"<div class='hdr'><b>{g}</b>"
                 f"<span class='desc'>{r.get('summary','')}</span></div>")
        p.append("<div class='eq'>")
        for i, (src, cap) in enumerate(cells_for(g)):
            p.append(f"<div class='cell'>{src}<div class='cap'>{cap}</div></div>")
            if i < 2:
                p.append(f"<div class='op'>{'+' if i == 0 else '='}</div>")
        p.append("</div></div>")
    p.append("</main></body></html>")
    return "\n".join(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of <garment>.png outputs")
    ap.add_argument("--report", required=True, help="f6_report.json")
    ap.add_argument("--crops", required=True, help="dir of cropped garment jpgs")
    ap.add_argument("--model", required=True, help="the person photo")
    ap.add_argument("--out", default=".")
    ap.add_argument("--bucket", default="<your-s3-bucket>")
    ap.add_argument("--prefix", default="results/")
    ap.add_argument("--profile", default="default")
    ap.add_argument("--no-s3", action="store_true")
    a = ap.parse_args()

    ex = lambda p: pathlib.Path(p).expanduser()
    recs = {r["garment"]: r for r in json.loads(ex(a.report).read_text())}
    order = sorted(recs)
    images, crops, model = ex(a.images), ex(a.crops), ex(a.model)
    out = ex(a.out); out.mkdir(parents=True, exist_ok=True)
    subtitle = f"{len(order)} garments"

    muri = data_uri(model)
    offline = page(order, recs, lambda g: [
        (f"<img src='{muri}'>", "MODEL"),
        (f"<img loading='lazy' src='{data_uri(crops / f'{g}.jpg')}'>", "GARMENT"),
        (f"<img loading='lazy' src='{data_uri(images / f'{g}.png')}'>", "OUTPUT"),
    ], subtitle)
    local = out / "gallery.html"
    local.write_text(offline)
    print(f"offline  {local}  ({local.stat().st_size/1e6:.1f} MB)")

    if a.no_s3:
        return 0

    import boto3
    s3 = boto3.Session(profile_name=a.profile, region_name="us-east-1").client("s3")
    jobs = [(model, f"{a.prefix}model.jpeg", "image/jpeg")]
    for g in order:
        jobs.append((images / f"{g}.png", f"{a.prefix}{g}.png", "image/png"))
        jobs.append((crops / f"{g}.jpg", f"{a.prefix}g_{g}.jpg", "image/jpeg"))
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda x: s3.upload_file(str(x[0]), a.bucket, x[1],
                                              ExtraArgs={"ContentType": x[2]}), jobs))

    def url(k):
        return s3.generate_presigned_url(
            "get_object", Params={"Bucket": a.bucket, "Key": k}, ExpiresIn=WEEK)

    linked = page(order, recs, lambda g: [
        (f"<a href='{url(a.prefix+'model.jpeg')}' target='_blank'>"
         f"<img src='{url(a.prefix+'model.jpeg')}'></a>", "MODEL"),
        (f"<a href='{url(a.prefix+'g_'+g+'.jpg')}' target='_blank'>"
         f"<img loading='lazy' src='{url(a.prefix+'g_'+g+'.jpg')}'></a>", "GARMENT"),
        (f"<a href='{url(a.prefix+g+'.png')}' target='_blank'>"
         f"<img loading='lazy' src='{url(a.prefix+g+'.png')}'></a>", "OUTPUT"),
    ], subtitle)

    for key, body in ((f"{a.prefix}index.html", linked),
                      (f"{a.prefix}gallery.html", offline)):
        s3.put_object(Bucket=a.bucket, Key=key, Body=body.encode(),
                      ContentType="text/html; charset=utf-8")
    print("linked  ", url(f"{a.prefix}index.html"))
    print("embedded", url(f"{a.prefix}gallery.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
