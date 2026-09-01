#!/usr/bin/env python
"""Run the full catalogue and archive the complete state of that run.

Every invocation writes tests/runs/<timestamp>/ containing everything needed to
understand the result later, or to tell two runs apart:

    state.json     config, environment, git commit, per-garment results
    summary.md     the same, readable
    images/        every generated image
    prompts/       the exact prompt each garment was given
    report.json    the raw report from the box

The point of archiving the environment and the commit alongside the images is
that this project has repeatedly produced numbers that could not later be
explained - a resolution that was not what it appeared, a guardrail that never
ran, timings from a configuration nobody recorded. A run that cannot be
reconstructed is not evidence.

usage:
    HOST=1.2.3.4 PORT=<PORT> python3 tests/run_suite.py
    HOST=... PORT=... python3 tests/run_suite.py --guardrail on --label with-qa
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
ROOT = os.environ.get("TRYON_ROOT",
        str(pathlib.Path(__file__).resolve().parent.parent / "runs"))


def ssh(host: str, port: str, key: str, cmd: str, timeout: int = 3600) -> str:
    full = ["ssh", "-o", "ConnectTimeout=15", "-p", port, "-i", key,
            f"root@{host}", cmd]
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr[-400:]}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", ""))
    ap.add_argument("--port", default=os.environ.get("PORT", ""))
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/id_ed25519"))
    ap.add_argument("--model", default="f6")
    ap.add_argument("--guardrail", choices=["on", "off", "default"], default="off")
    ap.add_argument("--megapixels", default="0.75")
    ap.add_argument("--steps", default="5")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--label", default="", help="short note kept with the run")
    a = ap.parse_args()

    if not a.host or not a.port:
        print("set HOST and PORT (the box changes address on every restart)")
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{a.model}-{a.guardrail}" + (f"-{a.label}" if a.label else "")
    out = HERE / "runs" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    print(f"run -> {out}")

    # --- environment, captured before anything else can change it -----------
    env_cmd = (
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        "--format=csv,noheader | head -1; echo '---'; "
        f"{ROOT}/venv/bin/python -c \"import torch;print(torch.__version__,"
        "torch.version.cuda)\"; echo '---'; "
        "curl -s -m 5 http://127.0.0.1:8000/v1/phase")
    raw_env = ssh(a.host, a.port, a.key, env_cmd, timeout=120)
    gpu, torch_v, phase = (raw_env.split("---") + ["", "", ""])[:3]

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())

    # --- the run -------------------------------------------------------------
    tag = f"suite_{stamp}"
    cmd = (f"cd {ROOT} && ./venv/bin/python pipeline/run_catalogue.py "
           f"--model {a.model} --guardrail {a.guardrail} "
           f"--megapixels {a.megapixels} --steps {a.steps} --seed {a.seed} "
           f"--tag {shlex.quote(tag)}")
    t0 = time.time()
    log = ssh(a.host, a.port, a.key, cmd)
    wall = time.time() - t0
    (out / "run.log").write_text(log)
    print(log.strip().splitlines()[-1] if log.strip() else "(no output)")

    # --- pull everything back ------------------------------------------------
    for remote, local in ((f"{ROOT}/outputs_{tag}/{a.model}", out / "images"),
                          (f"{ROOT}/outputs_{tag}/prompts", out / "prompts")):
        subprocess.run(
            f"ssh -p {a.port} -i {a.key} root@{a.host} "
            f"'cd {remote} && tar czf - .' | tar xzf - -C {local}",
            shell=True, capture_output=True)
    report_raw = ssh(a.host, a.port, a.key,
                     f"cat {ROOT}/outputs_{tag}/{a.model}_report.json")
    (out / "report.json").write_text(report_raw)
    records = json.loads(report_raw)

    # --- state ---------------------------------------------------------------
    done = [r for r in records if r.get("ok")]
    times = [r["seconds"] for r in done if "seconds" in r]
    checked = [r for r in done if r.get("guardrail")]
    passed = [r for r in checked if r.get("guardrail_ok")]
    state = {
        "run": name, "utc": stamp, "label": a.label,
        "commit": commit, "working_tree_dirty": dirty,
        "config": {"model": a.model, "guardrail": a.guardrail,
                   "megapixels": a.megapixels, "steps": a.steps, "seed": a.seed},
        "environment": {"gpu": gpu.strip(), "torch": torch_v.strip(),
                        "phase_at_start": phase.strip()},
        "results": {
            "garments": len(records), "succeeded": len(done),
            "failed": [r["garment"] for r in records if not r.get("ok")],
            "wall_seconds": round(wall, 1),
            "mean_seconds": round(sum(times) / len(times), 2) if times else None,
            "fastest": round(min(times), 2) if times else None,
            "slowest": round(max(times), 2) if times else None,
            "inspected": len(checked), "passed": len(passed),
            # Never conflate "not inspected" with "passed".
            "unverified": len(done) - len(checked),
            "retried": len([r for r in done if (r.get("attempts") or 1) > 1]),
        },
        "per_garment": records,
    }
    (out / "state.json").write_text(json.dumps(state, indent=2))

    r = state["results"]
    md = [f"# Run {name}", "",
          f"- commit `{commit}`" + ("  **(uncommitted changes present)**" if dirty else ""),
          f"- GPU: {gpu.strip()}", f"- torch: {torch_v.strip()}",
          f"- config: {a.model}, guardrail **{a.guardrail}**, {a.megapixels} MP, "
          f"{a.steps} steps, seed {a.seed}", ""]
    md += ["| metric | value |", "|---|---|",
           f"| garments | {r['garments']} |",
           f"| succeeded | {r['succeeded']} |",
           f"| mean | {r['mean_seconds']} s |",
           f"| fastest / slowest | {r['fastest']} s / {r['slowest']} s |",
           f"| wall clock | {r['wall_seconds'] / 60:.1f} min |",
           f"| inspected by guardrail | {r['inspected']} |",
           f"| passed | {r['passed']} |",
           f"| unverified (not inspected) | {r['unverified']} |",
           f"| needed a retry | {r['retried']} |", ""]
    if r["failed"]:
        md += [f"**Failed:** {', '.join(r['failed'])}", ""]
    md += ["| garment | seconds | pieces | guardrail |", "|---|---|---|---|"]
    for x in records:
        verdict = ("off" if not x.get("guardrail")
                   else "not inspected" if x.get("guardrail_ok") is None
                   else "pass" if x["guardrail_ok"] else "FLAG")
        md.append(f"| {x['garment']} | {x.get('seconds', '—')} | "
                  f"{x.get('pieces', '—')} | {verdict} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")

    print(f"\n{r['succeeded']}/{r['garments']} in {r['wall_seconds'] / 60:.1f} min "
          f"— mean {r['mean_seconds']}s")
    print(f"state -> {out}/state.json")
    print(f"summary -> {out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
