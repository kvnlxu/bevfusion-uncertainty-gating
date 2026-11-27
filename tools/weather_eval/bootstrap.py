"""Paired bootstrap significance test for mAP differences between two runs.

Compares two evaluated runs (each a directory with nusc/results_nusc.json, as
written by run_eval.py) on the same nuScenes split. Each bootstrap replicate
resamples the evaluation samples, applies the SAME resample to both runs
(paired), and recomputes nuScenes AP for each, so the reported difference
accounts for sampling noise from having only 81 samples.

Matching follows the nuScenes devkit exactly (center distance, greedy by
confidence, thresholds 0.5/1/2/4 m, min_recall/min_precision clipping). Because
a prediction can only match ground truth in its own sample, each prediction's
TP/FP outcome is computed once; a replicate only re-weights samples, which
makes thousands of replicates cheap. The un-resampled AP is checked against the
devkit's metrics_summary.json before any bootstrapping.

Consecutive samples in a scene are strongly correlated (2 Hz keyframes), so by
default samples are resampled in contiguous blocks within each scene (circular
block bootstrap). Use --block 1 for an i.i.d. bootstrap (narrower, optimistic).

Run from the repository root, e.g.:
  python tools/weather_eval/bootstrap.py \
    --pair work_dirs/weather_sweep/fog_200_both_s42 work_dirs/weather_sweep/lidaronly_fog_200_lidar_s42

  # The default set of comparisons behind the current findings, plus pooled
  # tests (e.g. the mean fused-vs-LiDAR-only gap over fog 1.5-3.0):
  python tools/weather_eval/bootstrap.py --preset fog-crossover

  # Pool your own pairs into one test:
  python tools/weather_eval/bootstrap.py --pool --pair RUN_A1 RUN_B1 --pair RUN_A2 RUN_B2
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = "work_dirs/weather_sweep"

PRESETS = {
    # (A, B): is A better than B?
    "fog-crossover": [
        (f"{SWEEP}/clean_000_both_s42", f"{SWEEP}/lidaronly_clean_000_lidar_s42"),
        (f"{SWEEP}/fog_100_both_s42", f"{SWEEP}/lidaronly_fog_100_lidar_s42"),
        (f"{SWEEP}/fog_150_both_s42", f"{SWEEP}/lidaronly_fog_150_lidar_s42"),
        (f"{SWEEP}/fog_200_both_s42", f"{SWEEP}/lidaronly_fog_200_lidar_s42"),
        (f"{SWEEP}/fog_250_both_s42", f"{SWEEP}/lidaronly_fog_250_lidar_s42"),
        (f"{SWEEP}/fog_300_both_s42", f"{SWEEP}/lidaronly_fog_300_lidar_s42"),
        # Ruined camera + clean LiDAR vs. the LiDAR-only model on clean data.
        (f"{SWEEP}/fog_200_camera_s42", f"{SWEEP}/lidaronly_clean_000_lidar_s42"),
        (f"{SWEEP}/fog_300_camera_s42", f"{SWEEP}/lidaronly_clean_000_lidar_s42"),
    ],
    # Camera switched off at the fusion point (sweep.py --camera-scale 0) vs.
    # the LiDAR-only model and vs. the unmodified fused model.
    "camera-off": [
        (f"{SWEEP}/{w}_s42_cam000", f"{SWEEP}/lidaronly_{w.replace('both', 'lidar')}_s42")
        for w in ("clean_000_both", "fog_100_both", "fog_150_both",
                  "fog_200_both", "fog_300_both")
    ] + [
        (f"{SWEEP}/{w}_s42_cam000", f"{SWEEP}/{w}_s42")
        for w in ("clean_000_both", "fog_100_both", "fog_150_both",
                  "fog_200_both", "fog_300_both")
    ],
}

# Pooled tests: the mean difference over several pairs, computed on the same
# bootstrap replicates, so it is one test instead of several. More powerful
# than the individual tests when the effect points the same way throughout.
_HEAVY_FOG = ("fog_150", "fog_200", "fog_250", "fog_300")
POOLS = {
    "fog-crossover": [
        ("heavy fog 1.5-3.0: fused vs LiDAR-only",
         [(f"{SWEEP}/{f}_both_s42", f"{SWEEP}/lidaronly_{f}_lidar_s42")
          for f in _HEAVY_FOG]),
        ("camera-only fog 2.0-3.0 vs LiDAR-only clean",
         [(f"{SWEEP}/{f}_camera_s42", f"{SWEEP}/lidaronly_clean_000_lidar_s42")
          for f in ("fog_200", "fog_300")]),
    ],
    "camera-off": [
        ("fog 1.0-3.0: camera off vs LiDAR-only",
         [(f"{SWEEP}/{f}_both_s42_cam000", f"{SWEEP}/lidaronly_{f}_lidar_s42")
          for f in ("fog_100", "fog_150", "fog_200", "fog_300")]),
        ("fog 1.0-3.0: camera off vs normal fused",
         [(f"{SWEEP}/{f}_both_s42_cam000", f"{SWEEP}/{f}_both_s42")
          for f in ("fog_100", "fog_150", "fog_200", "fog_300")]),
    ],
}


def gate_preset(prefix):
  """Pairs and pools for a trained gate whose runs are named <prefix>...

  Compares the gated model against the original fused model and the LiDAR-only
  model, normally and with the camera switched off (sweep.py --camera-scale 0).
  """
  heavy = ["fog_150", "fog_200", "fog_250", "fog_300"]
  off = ["fog_100", "fog_150", "fog_200", "fog_300"]
  g = lambda w, cam="": f"{SWEEP}/{prefix}{w}_both_s42{cam}"
  fused = lambda w, cam="": f"{SWEEP}/{w}_both_s42{cam}"
  lidar = lambda w: f"{SWEEP}/lidaronly_{w}_lidar_s42"
  pairs = (
      [(g(w), fused(w)) for w in ["clean_000", "fog_100", *heavy]]
      + [(g(w), lidar(w)) for w in ["clean_000", "fog_100", *heavy]]
      + [(g(w, "_cam000"), lidar(w)) for w in ["clean_000", *off]]
      + [(g(w, "_cam000"), fused(w, "_cam000")) for w in ["clean_000", *off]]
  )
  pools = [
      ("gated vs fused, heavy fog 1.5-3.0", [(g(w), fused(w)) for w in heavy]),
      ("gated vs LiDAR-only, heavy fog 1.5-3.0", [(g(w), lidar(w)) for w in heavy]),
      ("gated camera-off vs LiDAR-only, fog 1.0-3.0 (success criterion a)",
       [(g(w, "_cam000"), lidar(w)) for w in off]),
      ("gated camera-off vs fused camera-off, fog 1.0-3.0",
       [(g(w, "_cam000"), fused(w, "_cam000")) for w in off]),
  ]
  return pairs, pools


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--pair", nargs=2, action="append", metavar=("RUN_A", "RUN_B"),
                 help="Two run directories to compare (repeatable)")
  p.add_argument("--preset", choices=sorted(PRESETS) + ["gate"],
                 help="Named set of pairs. 'gate' compares a trained gate "
                      "(runs named --gate-prefix...) with both baselines")
  p.add_argument("--gate-prefix", default="gated_modality_",
                 help="Run-name prefix of the gated model for --preset gate")
  p.add_argument("--pool", action="store_true",
                 help="Also test the mean difference over all --pair pairs")
  p.add_argument("--reps", type=int, default=2000, help="Bootstrap replicates")
  p.add_argument("--block", type=int, default=8,
                 help="Block length in consecutive samples within a scene "
                      "(default 8 = 4 s at 2 Hz; 1 = i.i.d.)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--dataroot", default="data/nuscenes")
  p.add_argument("--classes", nargs="+", default=None,
                 help="Classes to average (default: classes with GT in the split)")
  p.add_argument("--out", default=f"{SWEEP}/bootstrap.csv",
                 help="CSV to write (one row per pair)")
  return p.parse_args()


def _devkit():
  from nuscenes import NuScenes
  from nuscenes.eval.detection.config import config_factory
  from nuscenes.eval.detection.data_classes import DetectionBox
  try:
    from nuscenes.eval.common.loaders import (
        add_center_dist, filter_eval_boxes, load_gt, load_prediction)
  except ImportError:  # older devkit layout
    from nuscenes.eval.detection.loaders import (
        add_center_dist, filter_eval_boxes, load_gt, load_prediction)
  return (NuScenes, config_factory, DetectionBox, add_center_dist,
          filter_eval_boxes, load_gt, load_prediction)


class Split:
  """Ground truth and sample layout shared by every run on one split."""

  def __init__(self, nusc, cfg, eval_set, devkit):
    _, _, DetectionBox, add_center_dist, filter_eval_boxes, load_gt, _ = devkit
    gt = load_gt(nusc, eval_set, DetectionBox, verbose=False)
    gt = add_center_dist(nusc, gt)
    gt = filter_eval_boxes(nusc, gt, cfg.class_range, verbose=False)
    self.gt = gt
    self.classes = list(cfg.class_names)
    self.tokens = list(gt.sample_tokens)
    self.index = {t: i for i, t in enumerate(self.tokens)}
    # GT count per (sample, class) after the devkit's filtering.
    self.ngt = np.zeros((len(self.tokens), len(self.classes)))
    for t in self.tokens:
      for box in gt[t]:
        self.ngt[self.index[t], self.classes.index(box.detection_name)] += 1
    # Scenes as ordered lists of sample indices, for block resampling.
    scenes = {}
    for t in self.tokens:
      s = nusc.get("sample", t)
      scenes.setdefault(s["scene_token"], []).append((s["timestamp"], self.index[t]))
    self.scenes = [np.array([i for _, i in sorted(v)]) for v in scenes.values()]


class Run:
  """Per-prediction TP/FP outcomes for one run, precomputed once."""

  def __init__(self, run_dir, split, nusc, cfg, devkit):
    _, _, DetectionBox, add_center_dist, filter_eval_boxes, _, load_prediction = devkit
    self.run_dir = Path(run_dir)
    result_path = self.run_dir / "nusc" / "results_nusc.json"
    if not result_path.exists():
      raise FileNotFoundError(f"{result_path} not found; run run_eval.py first")
    pred, _ = load_prediction(str(result_path), cfg.max_boxes_per_sample,
                              DetectionBox, verbose=False)
    if set(pred.sample_tokens) != set(split.tokens):
      raise ValueError(f"{run_dir}: predictions do not cover the eval split")
    pred = add_center_dist(nusc, pred)
    pred = filter_eval_boxes(nusc, pred, cfg.class_range, verbose=False)
    self.dist_ths = list(cfg.dist_ths)
    self.min_recall = cfg.min_recall
    self.min_precision = cfg.min_precision
    # outcomes[cls][dist_th] = (sample_idx, tp) in descending-confidence order
    self.outcomes = {}
    for cls in split.classes:
      boxes = [b for b in pred.all if b.detection_name == cls]
      confs = [b.detection_score for b in boxes]
      # Same ordering as the devkit's accumulate(): sort (score, index) pairs
      # ascending, then reverse. Unlike np.argsort, ties are broken by index,
      # and tied scores do occur, so this matters for exact agreement.
      order = [i for (_, i) in sorted((v, i) for (i, v) in enumerate(confs))][::-1]
      # Ground-truth centers of this class per sample (devkit: center distance
      # on x/y; ties resolved to the first GT in list order, as np.argmin does).
      gt_xy = {}
      for t in split.tokens:
        xy = [g.translation[:2] for g in split.gt[t] if g.detection_name == cls]
        gt_xy[t] = np.array(xy, dtype=np.float64).reshape(-1, 2)
      sample_idx = np.array([split.index[boxes[i].sample_token] for i in order],
                            dtype=np.int64)
      per_th = {}
      for th in self.dist_ths:
        free = {t: np.ones(len(xy), dtype=bool) for t, xy in gt_xy.items()}
        tp = np.zeros(len(order))
        for k, i in enumerate(order):
          box = boxes[i]
          xy, avail = gt_xy[box.sample_token], free[box.sample_token]
          if not avail.any():
            continue
          d = np.linalg.norm(xy - np.asarray(box.translation[:2]), axis=1)
          d[~avail] = np.inf
          j = int(np.argmin(d))
          if d[j] < th:
            avail[j] = False
            tp[k] = 1.0
        per_th[th] = (sample_idx, tp)
      self.outcomes[cls] = per_th

  def ap(self, cls, th, weights, npos):
    """nuScenes AP for one class/threshold with per-sample weights."""
    if npos <= 0:
      return 0.0
    sample_idx, tp = self.outcomes[cls][th]
    w = weights[sample_idx]
    keep = w > 0
    if not keep.any():
      return 0.0
    w, tp = w[keep], tp[keep]
    tp_c = np.cumsum(tp * w)
    if tp_c[-1] <= 0:  # devkit: no true positives -> no_predictions -> AP 0
      return 0.0
    fp_c = np.cumsum((1.0 - tp) * w)
    prec = tp_c / (tp_c + fp_c)
    rec = tp_c / npos
    prec = np.interp(np.linspace(0, 1, 101), rec, prec, right=0)
    prec = prec[round(100 * self.min_recall) + 1:]
    prec = prec - self.min_precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - self.min_precision)

  def class_ap(self, cls, weights, npos):
    return float(np.mean([self.ap(cls, th, weights, npos) for th in self.dist_ths]))

  def mean_ap(self, classes, split, weights):
    npos = weights @ split.ngt
    return float(np.mean([
        self.class_ap(c, weights, npos[split.classes.index(c)]) for c in classes
    ]))

  def check_against_devkit(self, split):
    summary = json.loads((self.run_dir / "nusc" / "metrics_summary.json").read_text())
    ones = np.ones(len(split.tokens))
    npos = ones @ split.ngt
    worst = max(
        abs(self.class_ap(c, ones, npos[i]) - summary["mean_dist_aps"][c])
        for i, c in enumerate(split.classes)
    )
    # Expected to agree to floating-point precision. Small differences can come
    # from devkit-version details; 1e-3 AP (0.1 points) is far below the effects
    # being tested, so warn up to that and refuse beyond it.
    if worst > 1e-3:
      raise RuntimeError(
          f"{self.run_dir}: re-computed AP differs from the devkit by {worst:.2e}; "
          "the bootstrap would not be trustworthy"
      )
    if worst > 1e-6:
      print(f"  note: {self.run_dir.name}: re-computed per-class AP differs from "
            f"the devkit by up to {worst:.1e} (negligible)")
    return worst


def block_weights(split, block, rng):
  """Sample counts for one circular-block bootstrap replicate, per scene.

  Blocks wrap around the end of the scene so every sample has the same chance
  of being drawn (a plain moving-block scheme underweights scene ends).
  """
  w = np.zeros(len(split.tokens))
  for scene in split.scenes:
    n = len(scene)
    b = max(1, min(block, n))
    starts = rng.integers(0, n, size=-(-n // b))
    picked = np.concatenate([scene[(s + np.arange(b)) % n] for s in starts])[:n]
    np.add.at(w, picked, 1.0)
  return w


def main():
  args = parse_args()
  user_pairs = [tuple(p) for p in (args.pair or [])]
  pairs = list(user_pairs)
  pools = []
  if args.pool and len(user_pairs) > 1:
    pools.append(("all --pair pairs", user_pairs))
  if args.preset == "gate":
    gate_pairs, gate_pools = gate_preset(args.gate_prefix)
    pairs += gate_pairs
    pools += gate_pools
  elif args.preset:
    pairs += PRESETS[args.preset]
    pools += POOLS.get(args.preset, [])
  if not pairs:
    sys.exit("Give --pair RUN_A RUN_B and/or --preset")
  for label, members in pools:
    extra = [m for m in members if m not in pairs]
    if extra:
      sys.exit(f"Pool '{label}' uses pairs that are not being compared: {extra}")
  abs_pair = lambda pr: (str(REPO_ROOT / pr[0]), str(REPO_ROOT / pr[1]))
  keys = list(dict.fromkeys(pairs))  # de-duplicate, keep order
  pairs = [abs_pair(pr) for pr in keys]
  missing = sorted({d for p in pairs for d in p
                    if not (Path(d) / "nusc" / "results_nusc.json").exists()})
  if missing:
    sys.exit("Missing results for:\n  " + "\n  ".join(missing))

  devkit = _devkit()
  NuScenes, config_factory = devkit[0], devkit[1]
  version = json.loads(
      (Path(pairs[0][0]) / "metrics.json").read_text()
  ).get("dataset_version") or "v1.0-mini"
  eval_set = {"v1.0-mini": "mini_val", "v1.0-trainval": "val"}[version]
  cfg = config_factory("detection_cvpr_2019")
  nusc = NuScenes(version=version, dataroot=args.dataroot, verbose=False)
  split = Split(nusc, cfg, eval_set, devkit)

  classes = args.classes or [
      c for i, c in enumerate(split.classes) if split.ngt[:, i].sum() > 0
  ]
  print(f"{version}/{eval_set}: {len(split.tokens)} samples in "
        f"{len(split.scenes)} scenes; averaging AP over {len(classes)} classes: "
        f"{', '.join(classes)}")
  print(f"{args.reps} paired replicates, block length {args.block}\n")

  runs = {}
  for d in sorted({d for p in pairs for d in p}):
    runs[d] = Run(d, split, nusc, cfg, devkit)
    runs[d].check_against_devkit(split)

  rng = np.random.default_rng(args.seed)
  replicates = [block_weights(split, args.block, rng) for _ in range(args.reps)]
  ones = np.ones(len(split.tokens))

  def summarize(base, diffs):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided p-value: twice the share of replicates whose difference has the
    # opposite sign to (or equals zero relative to) the observed one.
    p = min(1.0, 2 * float(np.mean(diffs <= 0) if base > 0 else np.mean(diffs >= 0)))
    sig = "significant" if lo > 0 or hi < 0 else "not significant"
    return lo, hi, p, sig

  rows = []
  results = {}
  for key, (a, b) in zip(keys, pairs):
    ra, rb = runs[a], runs[b]
    map_a, map_b = ra.mean_ap(classes, split, ones), rb.mean_ap(classes, split, ones)
    base = map_a - map_b
    diffs = np.array([
        ra.mean_ap(classes, split, w) - rb.mean_ap(classes, split, w)
        for w in replicates
    ])
    results[key] = (map_a, map_b, base, diffs)
    lo, hi, p, sig = summarize(base, diffs)
    rows.append(dict(
        run_a=str(Path(a).relative_to(REPO_ROOT)),
        run_b=str(Path(b).relative_to(REPO_ROOT)),
        mAP_a=map_a, mAP_b=map_b,
        diff=base, ci_low=lo, ci_high=hi, p_value=p,
        reps=args.reps, block=args.block, n_classes=len(classes),
    ))
    print(f"{Path(a).name}  vs  {Path(b).name}")
    print(f"  mAP {map_a:.4f} vs {map_b:.4f}  "
          f"diff {100 * base:+.2f} pts  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
          f"p={p:.3f}  ({sig} at 5%)")

  if pools:
    print("\nPooled tests (mean difference over the listed pairs, same replicates):")
  for label, members in pools:
    m = [results[k] for k in members]
    map_a = float(np.mean([r[0] for r in m]))
    map_b = float(np.mean([r[1] for r in m]))
    base = float(np.mean([r[2] for r in m]))
    diffs = np.mean([r[3] for r in m], axis=0)
    lo, hi, p, sig = summarize(base, diffs)
    rows.append(dict(
        run_a=f"POOLED: {label}",
        run_b=" | ".join(f"{Path(a).name} vs {Path(b).name}" for a, b in members),
        mAP_a=map_a, mAP_b=map_b,
        diff=base, ci_low=lo, ci_high=hi, p_value=p,
        reps=args.reps, block=args.block, n_classes=len(classes),
    ))
    print(f"{label}  ({len(members)} pairs)")
    print(f"  mean mAP {map_a:.4f} vs {map_b:.4f}  "
          f"diff {100 * base:+.2f} pts  95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]  "
          f"p={p:.3f}  ({sig} at 5%)")

  out = REPO_ROOT / args.out
  out.parent.mkdir(parents=True, exist_ok=True)
  with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nWrote {out}")


if __name__ == "__main__":
  main()
