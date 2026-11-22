import argparse
from pathlib import Path
import sys

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.transforms_3d import PointsRangeFilter
import torch


class _DummyRadarPoints:
  """Stand-in for radar when `results['radar']` is a raw dict or None."""

  def __init__(self, n=0):
    self.tensor = torch.zeros((n, 0), dtype=torch.float32)

  def in_range_bev(self, _pc_range):
    return torch.ones(len(self.tensor), dtype=torch.bool)

  def __getitem__(self, _idx):
    return self


def patch_points_range_filter():
  """Keep a point-like object on dict/None `radar` for downstream formatting."""
  orig_call = PointsRangeFilter.__call__

  def safe_call(self, results):
    orig_radar = results.get("radar", None)
    if orig_radar is None or isinstance(orig_radar, dict):
      results["radar"] = _DummyRadarPoints(0)
    return orig_call(self, results)

  PointsRangeFilter.__call__ = safe_call


def build_smoke_dataset(config_path: str, drop_bev_seg: bool = True):
  cfg = Config.fromfile(config_path)
  ds_cfg = (
      cfg.data.test._cfg_dict
      if hasattr(cfg.data.test, "_cfg_dict")
      else dict(cfg.data.test)
  )

  pipeline = list(ds_cfg.get("pipeline", []))
  if drop_bev_seg:
    pipeline = [
        op for op in pipeline if op.get("type") != "LoadBEVSegmentation"
    ]
    for op in pipeline:
      if op.get("type") == "Collect3D" and "keys" in op:
        op["keys"] = [k for k in op["keys"] if k != "gt_masks_bev"]

  ds_cfg["pipeline"] = pipeline
  return build_dataset(ds_cfg)


def main():
  parser = argparse.ArgumentParser(
      description="Run single-sample smoke test for WeatherDegradation pipeline."
  )
  parser.add_argument(
      "--config",
      default="work_dirs/weather_smoke/fog_060/config.py",
      help="Path to generated smoke config file.",
  )
  parser.add_argument(
      "--index",
      type=int,
      default=0,
      help="Dataset sample index to run through the pipeline.",
  )
  parser.add_argument(
      "--keep-bev-seg",
      action="store_true",
      help="Keep LoadBEVSegmentation and gt_masks_bev if map expansion JSONs exist.",
  )
  args = parser.parse_args()

  patch_points_range_filter()
  ds = build_smoke_dataset(args.config, drop_bev_seg=not args.keep_bev_seg)
  print(f"Dataset built: {type(ds)} | len: {len(ds)}")

  sample = ds[args.index]
  print(f"Sample {args.index} keys: {list(sample.keys())}")

  debug_dir = Path(args.config).parent / "inputs"
  if debug_dir.exists():
    print(f"Debug artifacts in {debug_dir}:")
    for p in sorted(debug_dir.iterdir()):
      print(f"  - {p.name} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
  sys.exit(main())