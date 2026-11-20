import argparse
from pathlib import Path
from mmengine.config import Config


def parse_args():
  parser = argparse.ArgumentParser(
      description="Generate weather-degraded BEVFusion config."
  )
  parser.add_argument(
      "--base-config", required=True, help="Path to base MMDet3D/BEVFusion config"
  )
  parser.add_argument(
      "--output-dir", required=True, help="Directory for generated config & artifacts"
  )
  parser.add_argument(
      "--weather",
      required=True,
      choices=["clean", "fog", "rain", "snow"],
      help="Weather condition",
  )
  parser.add_argument(
      "--severity", required=True, type=float, help="Weather severity in [0, 1]"
  )
  parser.add_argument(
      "--seed", type=int, default=42, help="Random seed for degradation"
  )
  parser.add_argument(
      "--mode",
      choices=["smoke", "full"],
      default="full",
      help="Smoke (1 sample + debug artifacts) or full mini_val",
  )
  parser.add_argument(
      "--import-module",
      default="tools.weather_eval.weather_degradation",
      help="Python module path registering WeatherDegradation",
  )
  parser.add_argument(
      "--data-root",
      default=None,
      help="Optional override for dataset data_root (e.g., data/nuscenes/)",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  if not (0.0 <= args.severity <= 1.0):
    raise ValueError(f"Severity must be in [0, 1], got {args.severity}")

  dest = Path(args.output_dir)
  dest.mkdir(parents=True, exist_ok=True)

  cfg = Config.fromfile(args.base_config)

  imports = dict(cfg.get("custom_imports") or {})
  names = imports.get("imports", [])
  if isinstance(names, str):
    names = [names]
  imports["imports"] = list(dict.fromkeys([*names, args.import_module]))
  imports["allow_failed_imports"] = False
  cfg.custom_imports = imports

  for split in ("test", "val"):
    if f"{split}_dataloader" not in cfg:
      continue
    dl = cfg[f"{split}_dataloader"]
    ds = dl.dataset
    if args.data_root:
      ds.data_root = args.data_root

    pipeline = list(ds.pipeline)
    types = [x["type"] for x in pipeline]
    assert (
        "BEVLoadMultiViewImageFromFiles" in types
        and "LoadPointsFromFile" in types
    ), f"Unexpected pipeline ops: {types}"

    loads = [
        i
        for i, t in enumerate(types)
        if t in (
            "BEVLoadMultiViewImageFromFiles",
            "LoadPointsFromFile",
            "LoadPointsFromMultiSweeps",
        )
    ]
    insert_idx = max(loads) + 1
    assert not any(
        "Aug" in t or "RangeFilter" in t or "Pack" in t
        for t in types[:insert_idx]
    ), f"Augmentation/filter found before load end: {types[:insert_idx]}"

    pipeline.insert(
        insert_idx,
        dict(
            type="WeatherDegradation",
            weather=args.weather,
            severity=float(args.severity),
            seed=int(args.seed),
            debug_dir=str(dest / "inputs") if args.mode == "smoke" else None,
        ),
    )
    ds.pipeline = pipeline

    if args.mode == "smoke":
      ds.indices = [0]
    else:
      ds.pop("indices", None)

    dl.batch_size = 1
    dl.num_workers = 0
    dl.persistent_workers = False

    if f"{split}_evaluator" in cfg:
      metric = cfg[f"{split}_evaluator"]
      metric.format_only = args.mode == "smoke"
      metric.jsonfile_prefix = str(dest / "predictions")

  cfg.work_dir = str(dest / "work")
  out_cfg_path = dest / "config.py"
  cfg.dump(str(out_cfg_path))
  print(f"Wrote {out_cfg_path}")
  print("Pipeline:", [x["type"] for x in cfg.test_dataloader.dataset.pipeline])


if __name__ == "__main__":
  main()