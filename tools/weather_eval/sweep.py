"""Severity sweep: generate configs, evaluate each, and collect one results.csv.

For every (weather, severity, modality, seed) it runs make_weather_config.py
then run_eval.py. Runs whose metrics.json already exists are skipped, so an
interrupted sweep can simply be restarted. Severity 0 is the clean baseline: it
is run once per seed (as weather "clean") and shared by all weather curves.

Run from the repository root, e.g.:
  python tools/weather_eval/sweep.py --checkpoint pretrained/bevfusion-det.pth

  # Per-sensor ablation for fog only, three severities:
  python tools/weather_eval/sweep.py --checkpoint pretrained/bevfusion-det.pth \
    --weathers fog --severities 0.3 0.6 1.0 --modalities both camera lidar

  # Rebuild results.csv from existing runs without evaluating anything:
  python tools/weather_eval/sweep.py --collect-only
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CLASSES = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
CSV_FIELDS = (
    ["weather", "severity", "modality", "seed", "mAP", "mAP_present", "NDS",
     "mATE", "mASE", "mAOE", "mAVE", "mAAE"]
    + [f"AP_{c}" for c in CLASSES]
    + ["num_samples", "dataset_version", "inference_seconds", "run_dir"]
)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--checkpoint", help="BEVFusion checkpoint (.pth)")
  p.add_argument(
      "--root", default="work_dirs/weather_sweep", help="Output root directory"
  )
  p.add_argument("--weathers", nargs="+", default=["fog", "rain", "snow"],
                 choices=["fog", "rain", "snow"])
  p.add_argument("--severities", nargs="+", type=float,
                 default=[0.2, 0.4, 0.6, 0.8, 1.0],
                 help="Non-zero severities; the clean baseline is added automatically")
  p.add_argument("--modalities", nargs="+", default=["both"],
                 choices=["both", "camera", "lidar"])
  p.add_argument("--seeds", nargs="+", type=int, default=[42])
  p.add_argument("--no-clean", action="store_true",
                 help="Do not run the severity-0 clean baseline")
  p.add_argument("--base-config", default=None,
                 help="Override make_weather_config.py's default base config")
  p.add_argument("--ann-file", default="data/nuscenes/nuscenes_infos_val_radar.pkl")
  p.add_argument("--workers", type=int, default=4)
  p.add_argument("--dry-run", action="store_true", help="Print commands only")
  p.add_argument("--collect-only", action="store_true",
                 help="Only rebuild results.csv from existing metrics.json files")
  p.add_argument("--keep-going", action="store_true",
                 help="Continue with the next run if one fails")
  return p.parse_args()


def run_name(weather, severity, modality, seed):
  return f"{weather}_{round(severity * 100):03d}_{modality}_s{seed}"


def plan_runs(args):
  runs = []
  for seed in args.seeds:
    if not args.no_clean:
      runs.append(("clean", 0.0, "both", seed))
    for weather in args.weathers:
      for modality in args.modalities:
        for sev in args.severities:
          if sev <= 0:
            continue
          runs.append((weather, sev, modality, seed))
  return runs


def call(cmd, dry_run):
  print("$ " + " ".join(cmd), flush=True)
  if not dry_run:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def execute(args, runs):
  failed = []
  for i, (weather, sev, modality, seed) in enumerate(runs, 1):
    run_dir = Path(args.root) / run_name(weather, sev, modality, seed)
    print(f"\n[{i}/{len(runs)}] {run_dir}", flush=True)
    if (REPO_ROOT / run_dir / "metrics.json").exists():
      print("  done already, skipping")
      continue
    gen = [
        sys.executable, str(HERE / "make_weather_config.py"),
        "--output-dir", str(run_dir),
        "--ann-file", args.ann_file,
        "--weather", weather,
        "--severity", str(sev),
        "--modality", modality,
        "--seed", str(seed),
        "--mode", "full",
        "--skip-bev-map",
    ]
    if args.base_config:
      gen += ["--base-config", args.base_config]
    ev = [
        sys.executable, str(HERE / "run_eval.py"),
        str(run_dir / "config.py"), args.checkpoint,
        "--workers", str(args.workers),
        "--skip-existing",
    ]
    try:
      call(gen, args.dry_run)
      call(ev, args.dry_run)
    except subprocess.CalledProcessError as exc:
      failed.append(str(run_dir))
      print(f"  FAILED: {exc}", flush=True)
      if not args.keep_going:
        raise
  return failed


def _class_ap(rec, run_dir):
  """Per-class AP from metrics.json, or from the devkit summary for older runs."""
  if rec.get("class_ap"):
    return rec["class_ap"]
  summary = run_dir / "nusc" / "metrics_summary.json"
  if summary.exists():
    return json.loads(summary.read_text()).get("mean_dist_aps", {})
  return {}


def collect(root):
  root_path = REPO_ROOT / root
  records = [
      (path.parent, json.loads(path.read_text()))
      for path in sorted(root_path.glob("*/metrics.json"))
  ]
  # All runs share one val split, so a run recorded before mAP_present existed
  # can borrow the present-class list from any newer run.
  shared_present = next(
      (rec["present_classes"] for _, rec in records if rec.get("present_classes")),
      None,
  )
  rows = []
  for run_dir, rec in records:
    m = rec.get("metrics", {})
    class_ap = _class_ap(rec, run_dir)
    present = rec.get("present_classes") or shared_present
    map_present = rec.get("mAP_present")
    if map_present is None and present and class_ap:
      map_present = sum(class_ap[c] for c in present) / len(present)
    row = dict(
        weather=rec.get("weather"),
        severity=rec.get("severity"),
        modality=rec.get("modality", "both"),
        seed=rec.get("degradation_seed"),
        mAP=rec.get("mAP"),
        mAP_present=map_present,
        NDS=rec.get("NDS"),
        mATE=m.get("object/mATE"),
        mASE=m.get("object/mASE"),
        mAOE=m.get("object/mAOE"),
        mAVE=m.get("object/mAVE"),
        mAAE=m.get("object/mAAE"),
        num_samples=rec.get("num_samples"),
        dataset_version=rec.get("dataset_version"),
        inference_seconds=rec.get("inference_seconds"),
        run_dir=str(run_dir.relative_to(REPO_ROOT)),
    )
    for c in CLASSES:
      row[f"AP_{c}"] = class_ap.get(c)
    rows.append(row)
  rows.sort(key=lambda r: (r["weather"] != "clean", r["weather"], r["modality"],
                           r["severity"], r["seed"]))
  out = root_path / "results.csv"
  root_path.mkdir(parents=True, exist_ok=True)
  with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nWrote {len(rows)} rows to {out}")
  if shared_present:
    print(f"mAP_present averages over: {', '.join(shared_present)}")
  for r in rows:
    mp = "  n/a " if r["mAP_present"] is None else f"{r['mAP_present']:.4f}"
    print(f"  {r['weather']:>5} s={r['severity']:.2f} {r['modality']:>6} "
          f"seed={r['seed']}  mAP={r['mAP']:.4f}  mAP_present={mp}  "
          f"NDS={r['NDS']:.4f}")


def main():
  args = parse_args()
  if not args.collect_only:
    if not args.checkpoint:
      sys.exit("--checkpoint is required unless --collect-only is given")
    runs = plan_runs(args)
    print(f"Planned {len(runs)} runs under {args.root}")
    failed = execute(args, runs)
    if failed:
      print(f"\n{len(failed)} run(s) failed: " + ", ".join(failed))
  if not args.dry_run:
    collect(args.root)


if __name__ == "__main__":
  main()
