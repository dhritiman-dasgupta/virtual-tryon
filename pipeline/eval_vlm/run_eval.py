#!/usr/bin/env python
"""Bake off small vision models on the guardrail task.

Runs the shipped guardrail prompt over a labelled set and scores each model on
the two things that matter separately:

    catch rate    of the defects present, how many did it flag?
    false alarms  of the correct images, how many did it flag anyway?

They are reported apart because the set is deliberately imbalanced - 126 PASS
labels against 21 FAIL - so a model that answers PASS to everything scores 86%
overall while being completely useless. A wasted reseed costs seconds; a missed
defect ships a wrong garment to a customer, so catch rate is weighted higher in
the summary but never allowed to hide the false-alarm number.

Unparseable output counts as a miss, matching the shipped behaviour: the
guardrail fails closed, and a model whose format cannot be read is not usable
regardless of how good its judgement is.

usage (Colab):
    !python run_eval.py --bundle /content/eval --models qwen3vl2b,qwen3vl4b
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import re
import time

# Multi-image capability is a hard requirement: the prompt shows three images.
# Single-image models (moondream, Florence-2) cannot run this task at all.
CANDIDATES = {
    # Qwen3.5 is a unified family - one model does text and vision, superseding
    # the separate -VL line. Included because the small ones are the most
    # plausible way to beat the current 4B on speed.
    "qwen35_08b":  "Qwen/Qwen3.5-0.8B",              # 0.87B
    "qwen35_2b":   "Qwen/Qwen3.5-2B",                # 2.27B
    "qwen35_4b":   "Qwen/Qwen3.5-4B",                # 4.66B
    # The -VL line, including the current production choice as the baseline.
    "qwen3vl2b":   "Qwen/Qwen3-VL-2B-Instruct",      # 2.13B
    "qwen3vl4b":   "Qwen/Qwen3-VL-4B-Instruct",      # 4.44B  <- production
    "qwen3vl8b":   "Qwen/Qwen3-VL-8B-Instruct",      # 8.77B
    # Previous generation and other families, as controls.
    "qwen25vl3b":  "Qwen/Qwen2.5-VL-3B-Instruct",
    "smolvlm2":    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "internvl3":   "OpenGVLab/InternVL3-2B-hf",
    "gemma3":      "google/gemma-3-4b-it",
    # Mixture-of-experts: 26B total but only ~4B active per token, so it may
    # run at small-model speed. Candidate for the catalogue pass rather than
    # the per-image guardrail.
    # --- catalogue-pass candidates ---------------------------------------
    # Too large to co-reside with the generator on a 24 GB card, so these are
    # for the one-off garment analysis that runs before generation starts.
    # diffusiongemma decodes blocks in parallel rather than one token at a
    # time, which suits a fixed 14-line format; 25.8B total but ~4B active.
    "diffgemma":   "google/diffusiongemma-26B-A4B-it",   # ~13 GB at 4-bit
    "gemma4_moe":  "google/gemma-4-26B-A4B-it",
}

_VERDICT = re.compile(r"\b(PASS(?:ES|ED)?|FAIL(?:S|ED|URE)?)\b", re.I)


def parse(text: str, fields: list[str]) -> dict[str, str]:
    """Pull `FIELD: ... PASS|FAIL ...` out of a reply.

    Mirrors the shipped parser, including its positional fallback: models drop
    the field labels often enough that a strict parser would score format
    compliance rather than judgement.
    """
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        line = line.strip().lstrip("-*# ").strip()
        if not line:
            continue
        for f in fields:
            if re.match(rf"^\**{f}\**\s*[:\-]", line, re.I):
                hit = _VERDICT.search(line)
                if hit:
                    out[f] = "FAIL" if hit.group(1).upper().startswith("FAIL") else "PASS"
                break
    if len(out) < len(fields) // 2:
        # Fallback: take verdicts in order from lines that carry one.
        verdicts = [("FAIL" if m.group(1).upper().startswith("FAIL") else "PASS")
                    for line in (text or "").splitlines()
                    if (m := _VERDICT.search(line))]
        if len(verdicts) >= len(fields):
            out = dict(zip(fields, verdicts[:len(fields)]))
    return out


def load(model_id: str, quantise: str, dtype_name: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    # T4 is Turing: no bf16. fp16 keeps the same models runnable on free Colab.
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    kwargs: dict = {"device_map": "cuda:0"}
    if quantise == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    else:
        kwargs["dtype"] = dtype
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.eval()
    return proc, model


def ask(proc, model, images, prompt: str, max_new_tokens: int, long_edge: int):
    import torch
    from PIL import Image

    pil = []
    for p in images:
        im = Image.open(p).convert("RGB")
        # Uniform downscale across models: the comparison is of judgement, and
        # letting each model see a different number of pixels would confound it.
        im.thumbnail((long_edge, long_edge), Image.LANCZOS)
        pil.append(im)
    content = [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}],
                                    tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=pil, padding=True,
                  return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    dt = time.time() - t0
    new = ids[0][inputs.input_ids.shape[1]:]
    return proc.decode(new, skip_special_tokens=True), dt, len(new)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="/content/eval")
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--quantise", default="4bit", choices=["4bit", "none"])
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=420)
    ap.add_argument("--long-edge", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/content/vlm_eval_results.json")
    a = ap.parse_args()

    import torch

    b = pathlib.Path(a.bundle)
    items = json.loads((b / "labels.json").read_text())
    if a.limit:
        items = items[:a.limit]
    prompt = (b / "ask.txt").read_text()
    fields = json.loads((b / "checks.json").read_text())["fields"]

    results = []
    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        model_id = CANDIDATES.get(key, key)
        print(f"\n=== {key}  {model_id}")
        rec = {"key": key, "model": model_id, "quantise": a.quantise}
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            proc, model = load(model_id, a.quantise, a.dtype)
            rec["load_seconds"] = round(time.time() - t0, 1)
            rec["weights_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        except Exception as e:                      # noqa: BLE001
            print(f"  load failed: {type(e).__name__}: {e}")
            results.append({**rec, "error": f"{type(e).__name__}: {e}"})
            continue

        per_item, lat, toks = [], [], []
        for n, it in enumerate(items, 1):
            imgs = [b / it["person"], b / it["garment"], b / it["output"]]
            try:
                raw, dt, ntok = ask(proc, model, imgs, prompt,
                                    a.max_new_tokens, a.long_edge)
            except Exception as e:                  # noqa: BLE001
                print(f"  [{n:2d}/{len(items)}] {it['id']:28s} ERROR {e}")
                per_item.append({"id": it["id"], "error": str(e), "got": {}})
                continue
            got = parse(raw, fields)
            lat.append(dt); toks.append(ntok)
            per_item.append({"id": it["id"], "seconds": round(dt, 2),
                             "tokens": ntok, "parsed": len(got),
                             "got": got, "raw": raw[:600]})
            print(f"  [{n:2d}/{len(items)}] {it['id']:28s} {dt:5.1f}s  "
                  f"{len(got):2d}/{len(fields)} fields")

        # --- score ----------------------------------------------------------
        caught = missed = false_alarm = correct_pass = unparsed = 0
        per_check: dict[str, dict] = {}
        for it, res in zip(items, per_item):
            got = res.get("got", {})
            for field, truth in it["labels"].items():
                if field not in fields:
                    continue
                said = got.get(field)
                d = per_check.setdefault(field, {"caught": 0, "missed": 0,
                                                 "false_alarm": 0, "ok": 0})
                if said is None:
                    unparsed += 1
                    # Fails closed: unreadable counts as a miss on a real defect
                    # and as a false alarm on a clean one.
                    if truth == "FAIL":
                        missed += 1; d["missed"] += 1
                    else:
                        false_alarm += 1; d["false_alarm"] += 1
                elif truth == "FAIL" and said == "FAIL":
                    caught += 1; d["caught"] += 1
                elif truth == "FAIL" and said == "PASS":
                    missed += 1; d["missed"] += 1
                elif truth == "PASS" and said == "FAIL":
                    false_alarm += 1; d["false_alarm"] += 1
                else:
                    correct_pass += 1; d["ok"] += 1

        n_fail = caught + missed
        n_pass = false_alarm + correct_pass
        rec.update(
            items=len(items),
            catch_rate=round(caught / n_fail, 3) if n_fail else None,
            false_alarm_rate=round(false_alarm / n_pass, 3) if n_pass else None,
            balanced=round(((caught / n_fail if n_fail else 0) +
                            (correct_pass / n_pass if n_pass else 0)) / 2, 3),
            unparsed_fields=unparsed,
            mean_seconds=round(sum(lat) / len(lat), 2) if lat else None,
            tokens_per_second=round(sum(toks) / sum(lat), 1) if lat else None,
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
            per_check=per_check,
            per_item=per_item,
        )
        print(f"  catch {rec['catch_rate']}  false-alarm {rec['false_alarm_rate']}  "
              f"balanced {rec['balanced']}  {rec['mean_seconds']}s/item  "
              f"{rec['peak_vram_gb']} GB")
        results.append(rec)

        del model, proc
        gc.collect(); torch.cuda.empty_cache()

    pathlib.Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\n{'model':14s} {'catch':>6s} {'false':>6s} {'bal':>6s} "
          f"{'s/item':>7s} {'tok/s':>6s} {'VRAM':>6s}")
    for r in results:
        if "error" in r:
            print(f"{r['key']:14s} {'FAILED TO LOAD':>40s}")
            continue
        print(f"{r['key']:14s} {r['catch_rate']:>6} {r['false_alarm_rate']:>6} "
              f"{r['balanced']:>6} {r['mean_seconds']:>7} "
              f"{r['tokens_per_second']:>6} {r['peak_vram_gb']:>6}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
