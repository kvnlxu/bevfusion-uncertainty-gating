"""Write a training config for the Phase 4 weather gate (used by train_gate.py).

Starts from the same resolved camera+lidar BEVFusion config as the evaluation
configs and changes it for a short fine-tune of the fusion stage:

  * model.fuser -> GatedConvFuser (gate type, hidden width, camera dropout)
  * data.train  -> nuScenes mini train infos (323 samples), plain dataset
                   (no CBGS class re-sampling), ObjectPaste removed (no GT
                   database exists for mini), BEV-map steps removed, and a
                   WeatherDegradation step in random mode after the loaders
  * schedule    -> AdamW, cosine decay with warmup, a few epochs, fp16 kept
  * trainable   -> `trainable_modules` (default: fuser + decoder); everything
                   else is frozen by train_gate.py
  * load_from   -> the released checkpoint

Run from the repository root:
  python tools/weather_eval/make_gate_config.py --output-dir work_dirs/gate/modality
"""
import argparse
from pathlib import Path
import sys

from mmcv import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_weather_config import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    MAX_SEVERITY,
    _check_model,
    _find_load_insert_idx,
    _load_merged_yaml,
    _op_names,
    _op_type,
    _resolve_all,
    _strip_bev_map,
)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--output-dir", required=True)
  p.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
  p.add_argument("--checkpoint", default="pretrained/bevfusion-det.pth",
                 help="Released checkpoint to start from")
  p.add_argument("--train-ann-file",
                 default="data/nuscenes/nuscenes_infos_train_radar.pkl")
  p.add_argument("--val-ann-file",
                 default="data/nuscenes/nuscenes_infos_val_radar.pkl")
  p.add_argument("--gate", choices=["modality", "channel", "spatial"],
                 default="modality")
  p.add_argument("--gate-hidden", type=int, default=64)
  p.add_argument("--camera-drop-prob", type=float, default=0.2,
                 help="Per-sample probability of zeroing camera features in "
                      "training (teaches LiDAR fallback)")
  p.add_argument("--trainable", nargs="+", default=["fuser", "decoder"],
                 help="Top-level model modules to train; the rest are frozen")
  p.add_argument("--clean-prob", type=float, default=0.25,
                 help="Share of training samples left without weather")
  p.add_argument("--weathers", nargs="+", default=["fog", "rain", "snow"])
  p.add_argument("--severity-range", nargs=2, type=float,
                 default=[0.0, MAX_SEVERITY])
  p.add_argument("--modality-probs", nargs=3, type=float, default=[0.5, 0.25, 0.25],
                 metavar=("BOTH", "CAMERA", "LIDAR"),
                 help="Probabilities of degrading both / camera only / LiDAR only")
  p.add_argument("--epochs", type=int, default=6)
  p.add_argument("--lr", type=float, default=1e-4)
  p.add_argument("--weight-decay", type=float, default=0.01)
  p.add_argument("--warmup-iters", type=int, default=100)
  p.add_argument("--samples-per-gpu", type=int, default=2)
  p.add_argument("--workers", type=int, default=4)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--keep-gridmask", action="store_true",
                 help="Keep GridMask camera augmentation (off by default: it "
                      "would compete with the weather degradation)")
  return p.parse_args()


def main():
  args = parse_args()
  dest = Path(args.output_dir)
  dest.mkdir(parents=True, exist_ok=True)

  cfg = Config(_resolve_all(_load_merged_yaml(Path(args.base_config))))
  if _check_model(cfg) != "fusion":
    sys.exit("The gate needs the camera+lidar fusion base config")

  # Model: gated fuser.
  fuser = cfg["model"]["fuser"]
  cfg["model"]["fuser"] = dict(
      type="GatedConvFuser",
      in_channels=list(fuser["in_channels"]),
      out_channels=int(fuser["out_channels"]),
      gate_type=args.gate,
      hidden_channels=args.gate_hidden,
      camera_drop_prob=args.camera_drop_prob,
  )

  imports = dict(cfg.get("custom_imports") or {})
  names = imports.get("imports", [])
  names = [names] if isinstance(names, str) else list(names)
  imports["imports"] = list(dict.fromkeys(
      [*names, "tools.weather_eval.weather_degradation"]))
  imports["allow_failed_imports"] = False
  cfg.custom_imports = imports

  # Training data.
  train = cfg["data"]["train"]
  if train.get("type") == "CBGSDataset":
    train = dict(train["dataset"])
  train["ann_file"] = args.train_ann_file
  removed = []
  pipe = []
  for op in _strip_bev_map(list(train["pipeline"]), True):
    t = _op_type(op)
    if t == "ObjectPaste" or (t == "GridMask" and not args.keep_gridmask):
      removed.append(t)
      continue
    pipe.append(op)
  both, cam, lid = args.modality_probs
  deg = dict(
      type="WeatherDegradation",
      random_cfg=dict(
          clean_prob=args.clean_prob,
          weathers=list(args.weathers),
          severity_range=list(args.severity_range),
          modality_probs=dict(both=both, camera=cam, lidar=lid),
      ),
  )
  pipe.insert(_find_load_insert_idx(pipe), deg)
  train["pipeline"] = pipe
  cfg["data"]["train"] = train
  cfg["data"]["samples_per_gpu"] = args.samples_per_gpu
  cfg["data"]["workers_per_gpu"] = args.workers
  for split in ("val", "test"):
    if cfg["data"].get(split):
      cfg["data"][split]["ann_file"] = args.val_ann_file
      cfg["data"][split]["pipeline"] = _strip_bev_map(
          list(cfg["data"][split]["pipeline"]), True)

  # Schedule: short fine-tune from the released checkpoint.
  cfg.trainable_modules = list(args.trainable)
  cfg.load_from = args.checkpoint
  cfg.resume_from = None
  cfg.max_epochs = args.epochs
  cfg.runner = dict(type="CustomEpochBasedRunner", max_epochs=args.epochs)
  cfg.optimizer = dict(type="AdamW", lr=args.lr, weight_decay=args.weight_decay)
  cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
  cfg.lr_config = dict(
      policy="CosineAnnealing", warmup="linear",
      warmup_iters=args.warmup_iters, warmup_ratio=0.1, min_lr_ratio=1e-2,
  )
  cfg.momentum_config = None
  cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=args.epochs)
  cfg.log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
  cfg.seed = args.seed
  cfg.deterministic = False
  cfg.work_dir = str(dest)

  out = dest / "config.py"
  cfg.dump(str(out))
  print(f"Wrote {out}")
  print(f"Fuser: {cfg['model']['fuser']}")
  print(f"Trainable modules: {cfg.trainable_modules}; removed from train pipeline: "
        f"{removed or 'nothing'}")
  print(f"Train pipeline: {_op_names(pipe)}")


if __name__ == "__main__":
  main()
