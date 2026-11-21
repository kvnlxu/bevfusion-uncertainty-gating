import argparse
import ast
from collections.abc import Mapping
import re
from pathlib import Path
from mmcv import Config


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
      help="Smoke (1 sample + debug artifacts) or full val/test",
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
  parser.add_argument(
      "--ann-file",
      default=None,
      help="Optional override for test/val ann_file (e.g., data/nuscenes/nuscenes_infos_val_radar.pkl)",
  )
  parser.add_argument(
      "--skip-bev-map",
      action="store_true",
      help="Remove LoadBEVSegmentation and gt_masks_bev when map expansion JSONs are absent",
  )
  return parser.parse_args()


def _deep_merge(base, override):
  if isinstance(base, Mapping) and isinstance(override, Mapping):
    merged = {k: _deep_merge(v, v) for k, v in base.items()}
    for k, v in override.items():
      if (
          k in merged
          and isinstance(merged[k], Mapping)
          and isinstance(v, Mapping)
      ):
        merged[k] = _deep_merge(merged[k], v)
      else:
        merged[k] = _deep_merge({}, v) if isinstance(v, Mapping) else v
    return merged
  if isinstance(override, list):
    return [_deep_merge({}, x) if isinstance(x, Mapping) else x for x in override]
  return override


def _lookup_path(root, expr_head):
  m = re.match(r"^([A-Za-z0-9_.]+)(\[[^\]]+\])?$", expr_head.strip())
  if not m:
    raise KeyError(f"Unsupported interpolation target: {expr_head}")
  parts = m.group(1).split(".")
  cur = root
  for p in parts:
    if isinstance(cur, Mapping) and p in cur:
      cur = cur[p]
    else:
      raise KeyError(f"Key '{p}' not found while resolving '{expr_head}'")
  if m.group(2):
    raw_idx = m.group(2)[1:-1].strip()
    if ":" in raw_idx:
      slc_args = ",".join(
          s.strip() if s.strip() else "None" for s in raw_idx.split(":")
      )
      idx_val = ast.literal_eval(f"slice({slc_args})")
    else:
      idx_val = ast.literal_eval(raw_idx)
    cur = cur[idx_val]
  return cur


def _eval_expr(expr, root):
  expr = expr.strip()
  if "+" in expr and ('"' in expr or "'" in expr):
    parts = [p.strip() for p in expr.split("+")]
    out = []
    for p in parts:
      if (p.startswith('"') and p.endswith('"')) or (
          p.startswith("'") and p.endswith("'")
      ):
        out.append(ast.literal_eval(p))
      else:
        out.append(str(_lookup_path(root, p)))
    return "".join(out)
  return _lookup_path(root, expr)


def _resolve_once(node, root):
  if isinstance(node, Mapping):
    return {k: _resolve_once(v, root) for k, v in node.items()}
  if isinstance(node, list):
    return [_resolve_once(v, root) for v in node]
  if isinstance(node, str):
    full_match = re.fullmatch(r"\$\{([^}]+)\}", node.strip())
    if full_match:
      return _eval_expr(full_match.group(1), root)
    if "${" in node:
      return re.sub(
          r"\$\{([^}]+)\}", lambda m: str(_eval_expr(m.group(1), root)), node
      )
  return node


def _resolve_all(cfg_dict):
  cur = cfg_dict
  for _ in range(15):
    nxt = _resolve_once(cur, cur)
    if nxt == cur:
      return nxt
    cur = nxt
  return cur


def _op_type(item):
  if isinstance(item, Mapping):
    return item.get("type", "")
  return ""


def _op_names(pipeline):
  return [_op_type(x) if _op_type(x) else str(x) for x in pipeline]


def _strip_bev_map(pipeline, skip_bev_map):
  if not skip_bev_map:
    return pipeline
  cleaned = []
  for op in pipeline:
    if isinstance(op, Mapping) and op.get("type") == "LoadBEVSegmentation":
      continue
    if (
        isinstance(op, Mapping)
        and op.get("type") == "Collect3D"
        and isinstance(op.get("keys"), list)
    ):
      op = dict(op)
      op["keys"] = [k for k in op["keys"] if k != "gt_masks_bev"]
    cleaned.append(op)
  return cleaned


def _find_load_insert_idx(pipeline):
  types = [_op_type(x) for x in pipeline]
  load_types = {
      "LoadMultiViewImageFromFiles",
      "BEVLoadMultiViewImageFromFiles",
      "LoadPointsFromFile",
      "LoadPointsFromMultiSweeps",
  }
  loads = [i for i, t in enumerate(types) if t in load_types]
  if not loads:
    raise ValueError(
        f"Could not find image/point loader in pipeline ops: {_op_names(pipeline)}"
    )
  insert_idx = max(loads) + 1
  forbidden_prefix = (
      "Aug",
      "RangeFilter",
      "Pack",
      "DefaultFormatBundle",
      "Collect",
  )
  assert not any(
      any(tok in t for tok in forbidden_prefix) for t in types[:insert_idx]
  ), (
      f"Augmentation/filter/formatter found before load end: {_op_names(pipeline[:insert_idx])}"
  )
  return insert_idx


def main():
  args = parse_args()
  if not (0.0 <= args.severity <= 1.0):
    raise ValueError(f"Severity must be in [0, 1], got {args.severity}")

  dest = Path(args.output_dir)
  dest.mkdir(parents=True, exist_ok=True)

  base_path = Path(args.base_config)
  default_yaml = Path("configs/default.yaml")
  if (
      base_path.suffix == ".yaml"
      and default_yaml.exists()
      and base_path.resolve() != default_yaml.resolve()
  ):
    raw_cfg = _deep_merge(
        Config.fromfile(str(default_yaml))._cfg_dict,
        Config.fromfile(str(base_path))._cfg_dict,
    )
  else:
    raw_cfg = Config.fromfile(str(base_path))._cfg_dict

  if args.data_root:
    raw_cfg["dataset_root"] = args.data_root
  resolved_dict = _resolve_all(raw_cfg)
  cfg = Config(resolved_dict)

  imports = dict(cfg.get("custom_imports") or {})
  names = imports.get("imports", [])
  if isinstance(names, str):
    names = [names]
  imports["imports"] = list(dict.fromkeys([*names, args.import_module]))
  imports["allow_failed_imports"] = False
  cfg.custom_imports = imports

  deg_step = dict(
      type="WeatherDegradation",
      weather=args.weather,
      severity=float(args.severity),
      seed=int(args.seed),
      debug_dir=str(dest / "inputs") if args.mode == "smoke" else None,
  )

  updated_pipelines = []

  if "data" in cfg:
    if args.data_root:
      cfg.data["dataset_root"] = args.data_root
    cfg.data["workers_per_gpu"] = 0
    cfg.data["samples_per_gpu"] = 1

    for split in ("test", "val"):
      if split not in cfg.data or cfg.data[split] is None:
        continue
      ds = cfg.data[split]
      if args.data_root:
        ds["dataset_root"] = args.data_root
        if "data_root" in ds:
          ds["data_root"] = args.data_root
      if args.ann_file:
        ds["ann_file"] = args.ann_file
      if "pipeline" in ds and isinstance(ds["pipeline"], list):
        pipe = _strip_bev_map(list(ds["pipeline"]), args.skip_bev_map)
        idx = _find_load_insert_idx(pipe)
        pipe.insert(idx, dict(deg_step))
        ds["pipeline"] = pipe
        updated_pipelines.append((f"data.{split}.pipeline", _op_names(pipe)))

  if "test_pipeline" in cfg and isinstance(cfg.test_pipeline, list):
    pipe = _strip_bev_map(list(cfg.test_pipeline), args.skip_bev_map)
    idx = _find_load_insert_idx(pipe)
    pipe.insert(idx, dict(deg_step))
    cfg.test_pipeline = pipe
    updated_pipelines.append(("test_pipeline", _op_names(pipe)))

  if not updated_pipelines:
    raise RuntimeError(
        "No test/val pipeline found in base config to inject WeatherDegradation."
    )

  cfg.work_dir = str(dest / "work")
  out_cfg_path = dest / "config.py"
  cfg.dump(str(out_cfg_path))
  print(f"Wrote {out_cfg_path}")
  for name, ops in updated_pipelines:
    print(f"Pipeline ({name}): {ops}")


if __name__ == "__main__":
  main()