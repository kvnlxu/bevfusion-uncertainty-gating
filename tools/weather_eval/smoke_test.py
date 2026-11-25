import argparse
from pathlib import Path
import sys

# Running `python tools/weather_eval/smoke_test.py` puts tools/weather_eval on
# sys.path, not the repo root, so the config's custom_imports
# ("tools.weather_eval.weather_degradation") would not be importable.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from mmcv import Config  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402


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


def check_model(config_path: str, checkpoint: str):
  """Build the model from the config and compare it with the checkpoint.

  Catches configs whose architecture does not match the checkpoint (wrong
  base config, missing fuser or camera backbone) without needing a GPU.
  """
  import torch
  from mmdet3d.models import build_model

  cfg = Config.fromfile(config_path)
  cfg.model.pretrained = None
  cfg.model.train_cfg = None
  model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
  print(f"Model built: {type(model).__name__} | fuser: {type(model.fuser).__name__}")

  # mmcv's own sparse conv layers share names with mmdet3d.ops.spconv in the
  # CONV_LAYERS registry but cannot consume mmdet3d's SparseConvTensor.
  foreign = sorted({
      f"{type(m).__module__}.{type(m).__name__}"
      for m in model.modules()
      if type(m).__module__.startswith("mmcv.ops")
  })
  if foreign:
    raise SystemExit(
        "Model contains mmcv.ops layers that mmdet3d's sparse encoder cannot "
        f"run: {foreign}. Check CONV_LAYERS registration in "
        "mmdet3d/ops/spconv/conv.py."
    )

  ckpt = torch.load(checkpoint, map_location="cpu")
  ckpt_sd = ckpt.get("state_dict", ckpt)
  model_sd = model.state_dict()
  missing = sorted(set(model_sd) - set(ckpt_sd))
  unexpected = sorted(set(ckpt_sd) - set(model_sd))
  mismatched = sorted(
      k for k in set(model_sd) & set(ckpt_sd)
      if tuple(model_sd[k].shape) != tuple(ckpt_sd[k].shape)
  )
  for name, keys in (("missing", missing), ("unexpected", unexpected),
                     ("shape-mismatched", mismatched)):
    print(f"Checkpoint {name} keys: {len(keys)}")
    for k in keys[:10]:
      print(f"  - {k}")
  if missing or unexpected or mismatched:
    raise SystemExit("Config does not match checkpoint; see keys above.")
  print("Checkpoint matches model.")


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
  parser.add_argument(
      "--checkpoint",
      default=None,
      help="Also build the model and check it against this checkpoint.",
  )
  args = parser.parse_args()

  if args.checkpoint:
    check_model(args.config, args.checkpoint)

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