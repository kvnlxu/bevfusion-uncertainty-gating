"""Plot robustness curves and the per-sensor ablation from sweep.py's results.csv.

Writes PNG (for slides/notes) and PDF (for papers) to <root>/plots/:
  robustness_curves   mAP vs. severity, one panel per weather: fused model with
                      both / camera / LiDAR degraded, and the LiDAR-only model
  sensor_ablation     mAP points lost at one severity, by which sensor is degraded

Run from the repository root:
  python tools/weather_eval/plot_results.py
  python tools/weather_eval/plot_results.py --metric mAP --ablation-severity 0.6
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# Reference categorical palette, slots 1-4 in fixed order (validated: adjacent
# CVD dE >= 11.8; slots 2-4 are below 3:1 contrast on white, so every series is
# also direct-labeled and carries its own marker/line style).
SERIES = [
    # key, label, color, marker, linestyle
    (("fusion", "both"), "Fusion, both degraded", "#257de0", "o", "-"),
    (("fusion", "camera"), "Fusion, camera degraded", "#f47230", "s", "-"),
    (("fusion", "lidar"), "Fusion, LiDAR degraded", "#00b894", "^", "-"),
    (("lidar", "lidar"), "LiDAR-only model", "#f2a600", "D", "--"),
]
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"
WEATHERS = ["fog", "rain", "snow"]
METRIC_LABEL = {
    "mAP_present": "mAP (classes present in split)",
    "mAP": "mAP (10-class nuScenes)",
    "NDS": "NDS",
}


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--root", default="work_dirs/weather_sweep")
  p.add_argument("--metric", default="mAP_present", choices=sorted(METRIC_LABEL))
  p.add_argument("--ablation-severity", type=float, default=1.0)
  p.add_argument("--seed", type=int, default=42, help="Degradation seed to plot")
  return p.parse_args()


def load(root, seed):
  rows = []
  with open(REPO_ROOT / root / "results.csv", newline="") as f:
    for r in csv.DictReader(f):
      if r["seed"] and int(float(r["seed"])) != seed:
        continue
      if float(r.get("camera_scale") or 1.0) != 1.0:  # camera-off runs: not plotted
        continue
      for k in ("severity", "mAP", "mAP_present", "NDS"):
        r[k] = float(r[k]) if r.get(k) not in (None, "") else None
      r["model"] = r.get("model") or "fusion"
      rows.append(r)
  return rows


def clean_value(rows, model, metric):
  for r in rows:
    if r["model"] == model and r["weather"] == "clean":
      return r[metric]
  return None


def curve(rows, model, modality, weather, metric):
  pts = {r["severity"]: r[metric] for r in rows
         if r["model"] == model and r["modality"] == modality
         and r["weather"] == weather and r[metric] is not None}
  base = clean_value(rows, model, metric)
  if pts and base is not None:
    pts.setdefault(0.0, base)
  return sorted(pts.items())


def style_axes(ax):
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
  ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
  ax.grid(axis="y", color=GRID, linewidth=0.8)
  ax.set_axisbelow(True)


def plot_curves(rows, metric, out_dir):
  fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
  handles = {}
  for ax, weather in zip(axes, WEATHERS):
    style_axes(ax)
    ends = []
    for (model, modality), label, color, marker, ls in SERIES:
      pts = curve(rows, model, modality, weather, metric)
      if len(pts) < 2:
        continue
      xs, ys = zip(*pts)
      (h,) = ax.plot(xs, ys, color=color, linewidth=2, linestyle=ls,
                     marker=marker, markersize=6,
                     markeredgecolor="white", markeredgewidth=1.5, label=label)
      handles[label] = h
      ends.append([xs[-1], ys[-1], label.split(",")[-1].strip()
                   if model == "fusion" else "LiDAR-only", color])
    # Direct labels at line ends, nudged apart so they do not collide.
    ends.sort(key=lambda e: e[1])
    for i in range(1, len(ends)):
      if ends[i][1] - ends[i - 1][1] < 0.022:
        ends[i][1] = ends[i - 1][1] + 0.022
    x_max = max((e[0] for e in ends), default=0)
    for x, y, text, color in ends:
      if x < x_max:  # series stops mid-panel: label above its last point
        ax.annotate(text, (x, y), xytext=(4, 9), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8.5, color=TEXT_2)
      else:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TEXT_2)
    ax.set_title(weather.capitalize(), color=TEXT, fontsize=12, loc="left")
    ax.set_xlabel("Severity", color=TEXT_2, fontsize=10)
    ax.margins(x=0.18)
  axes[0].set_ylabel(METRIC_LABEL[metric], color=TEXT_2, fontsize=10)
  fig.legend([handles[k] for k in [s[1] for s in SERIES] if k in handles],
             [k for k in [s[1] for s in SERIES] if k in handles],
             loc="upper center", ncol=4, frameon=False, fontsize=9,
             labelcolor=TEXT, bbox_to_anchor=(0.5, 1.02))
  fig.suptitle("BEVFusion robustness to synthetic weather (nuScenes mini-val, 81 samples)",
               color=TEXT, fontsize=13, x=0.01, ha="left", y=1.09)
  fig.tight_layout()
  save(fig, out_dir / "robustness_curves")


def plot_ablation(rows, metric, severity, out_dir):
  base = clean_value(rows, "fusion", metric)
  bars = SERIES[:3]  # fused model: both / camera / LiDAR degraded
  fig, ax = plt.subplots(figsize=(7.5, 4.2))
  style_axes(ax)
  width = 0.26
  gap = 0.02  # surface gap between adjacent bars
  for k, ((model, modality), label, color, _, _) in enumerate(bars):
    for i, weather in enumerate(WEATHERS):
      val = next((r[metric] for r in rows if r["model"] == model
                  and r["modality"] == modality and r["weather"] == weather
                  and r["severity"] is not None
                  and abs(r["severity"] - severity) < 1e-9), None)
      if val is None or base is None:
        continue
      drop = 100 * (base - val)
      x = i + (k - 1) * width
      ax.bar(x, drop, width=width - gap, color=color,
             label=label if i == 0 else None)
      ax.annotate(f"{drop:.1f}", (x, drop), xytext=(0, 3),
                  textcoords="offset points", ha="center", fontsize=8.5,
                  color=TEXT_2)
  ax.set_xticks(range(len(WEATHERS)))
  ax.set_xticklabels([w.capitalize() for w in WEATHERS], color=TEXT, fontsize=10)
  ax.set_ylabel(f"{METRIC_LABEL[metric]}\npoints lost vs. clean", color=TEXT_2,
                fontsize=10)
  ax.set_title(f"Which sensor's degradation costs accuracy (severity {severity:g})",
               color=TEXT, fontsize=12, loc="left")
  ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
  fig.tight_layout()
  save(fig, out_dir / f"sensor_ablation_s{round(severity * 100):03d}")


def save(fig, stem):
  for ext in ("png", "pdf"):
    fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
  plt.close(fig)
  print(f"Wrote {stem}.png / .pdf")


def main():
  args = parse_args()
  rows = load(args.root, args.seed)
  out_dir = REPO_ROOT / args.root / "plots"
  out_dir.mkdir(parents=True, exist_ok=True)
  plot_curves(rows, args.metric, out_dir)
  plot_ablation(rows, args.metric, args.ablation_severity, out_dir)


if __name__ == "__main__":
  main()
