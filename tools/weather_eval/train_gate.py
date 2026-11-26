"""Train the Phase 4 weather gate on one GPU (config from make_gate_config.py).

tools/train.py needs torchpack's YAML configs and a distributed launch; this
script loads the generated mmcv config.py and trains on a single GPU, which is
all a fine-tune on the 323 nuScenes mini train samples needs.

What it does:
  1. Builds the model with GatedConvFuser and loads the released checkpoint.
     Only the new gate parameters may be missing; anything else is an error.
  2. Freezes every parameter outside cfg.trainable_modules (default: fuser and
     decoder) and keeps the frozen parts in eval mode, so their BatchNorm
     statistics and dropout stay exactly as released.
  3. Trains with mmcv's runner (fp16, AdamW, cosine schedule) and logs the mean
     camera and LiDAR gate weights.

Checkpoints go to the config's directory (epoch_N.pth, latest.pth). Evaluate
with run_eval.py / sweep.py --gate <type> --checkpoint <dir>/latest.pth.

Run from the repository root:
  python tools/weather_eval/train_gate.py work_dirs/gate/modality/config.py
  python tools/weather_eval/train_gate.py work_dirs/gate/modality/config.py --dry-run
"""
import argparse
from pathlib import Path
import random
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import (  # noqa: E402
    Fp16OptimizerHook,
    HOOKS,
    Hook,
    OptimizerHook,
    build_optimizer,
    build_runner,
    load_checkpoint,
)
from mmdet.datasets import build_dataloader  # noqa: E402

from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.utils import get_root_logger  # noqa: E402
import mmdet3d.runner  # noqa: E402,F401  (registers CustomEpochBasedRunner)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("config", help="config.py from make_gate_config.py")
  p.add_argument("--dry-run", action="store_true",
                 help="Build everything and run 2 training iterations, then stop")
  return p.parse_args()


def is_trainable(name, prefixes):
  return any(name == p or name.startswith(p + ".") for p in prefixes)


@HOOKS.register_module(force=True)
class FreezeHook(Hook):
  """Keep frozen top-level modules in eval mode during training.

  The runner calls model.train() at the start of every epoch, which would put
  frozen BatchNorm layers back into training mode and change their statistics.
  """

  def __init__(self, trainable):
    self.trainable = list(trainable)

  def _apply(self, runner):
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    for name, module in model.named_children():
      if not is_trainable(name, self.trainable):
        module.eval()

  def before_train_epoch(self, runner):
    self._apply(runner)

  def before_train_iter(self, runner):
    self._apply(runner)


@HOOKS.register_module(force=True)
class GateWeightLoggerHook(Hook):
  """Add the fuser's mean camera / LiDAR gate weights to the training log."""

  def after_train_iter(self, runner):
    model = runner.model.module if hasattr(runner.model, "module") else runner.model
    weights = getattr(getattr(model, "fuser", None), "last_weights", None)
    if weights is not None:
      runner.log_buffer.update({"w_cam": weights[0], "w_lidar": weights[1]})


@HOOKS.register_module(force=True)
class StopAfterItersHook(Hook):
  """For --dry-run: stop training after a few iterations."""

  def __init__(self, iters):
    self.iters = iters

  def after_train_iter(self, runner):
    if runner.iter + 1 >= self.iters:
      runner._max_iters = runner.iter + 1
      runner._max_epochs = runner.epoch
      raise SystemExit(0)


def main():
  args = parse_args()
  cfg = Config.fromfile(args.config)  # also imports custom modules
  work_dir = Path(cfg.work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)
  timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
  logger = get_root_logger(log_file=str(work_dir / f"{timestamp}.log"))
  logger.info(f"Config: {args.config}")

  seed = cfg.get("seed", 0)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = True

  trainable = list(cfg.get("trainable_modules", ["fuser", "decoder"]))

  # Model + released weights. init_weights() first (it loads ImageNet Swin
  # weights), then the checkpoint overwrites everything it contains.
  model = build_model(cfg.model)
  model.init_weights()
  ckpt = torch.load(cfg.load_from, map_location="cpu")
  state = ckpt.get("state_dict", ckpt)
  model_keys = set(model.state_dict())
  missing = sorted(model_keys - set(state))
  unexpected = sorted(set(state) - model_keys)
  bad_missing = [k for k in missing if not k.startswith("fuser.gate.")]
  if bad_missing or unexpected:
    raise SystemExit(
        f"Checkpoint does not match the gated model: missing {bad_missing[:10]} "
        f"unexpected {unexpected[:10]}"
    )
  load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False,
                  logger=logger)
  logger.info(f"Loaded {cfg.load_from}; new (randomly/zero-initialised) "
              f"parameters: {len(missing)} gate tensors")

  # Freeze everything outside the trainable modules.
  n_train = n_frozen = 0
  for name, param in model.named_parameters():
    param.requires_grad = is_trainable(name, trainable)
    if param.requires_grad:
      n_train += param.numel()
    else:
      n_frozen += param.numel()
  children = [n for n, _ in model.named_children()]
  unknown = [p for p in trainable if p.split(".")[0] not in children]
  if unknown:
    raise SystemExit(f"trainable_modules {unknown} not in model ({children})")
  logger.info(f"Trainable modules {trainable}: {n_train / 1e6:.2f}M parameters "
              f"trained, {n_frozen / 1e6:.2f}M frozen")

  dataset = build_dataset(cfg.data.train)
  logger.info(f"Training samples: {len(dataset)}")
  loader = build_dataloader(
      dataset,
      cfg.data.samples_per_gpu,
      cfg.data.workers_per_gpu,
      num_gpus=1,
      dist=False,
      shuffle=True,
      seed=seed,
  )

  model = MMDataParallel(model.cuda(), device_ids=[0])
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = build_optimizer(model, cfg.optimizer)
  # Keep only trainable parameters in the optimizer (smaller AdamW state).
  optimizer.param_groups = [
      dict(g, params=[p for p in g["params"] if p.requires_grad])
      for g in optimizer.param_groups
  ]
  assert sum(len(g["params"]) for g in optimizer.param_groups) == len(params)

  runner = build_runner(
      cfg.runner,
      default_args=dict(model=model, optimizer=optimizer, work_dir=str(work_dir),
                        logger=logger, meta=dict(config=args.config)),
  )
  if hasattr(runner, "set_dataset"):
    runner.set_dataset([dataset])
  runner.timestamp = timestamp

  fp16 = cfg.get("fp16", None)
  if fp16 is not None:
    optimizer_hook = Fp16OptimizerHook(**cfg.optimizer_config, **fp16,
                                       distributed=False)
  else:
    optimizer_hook = OptimizerHook(**cfg.optimizer_config)
  runner.register_training_hooks(
      cfg.lr_config, optimizer_hook, cfg.checkpoint_config, cfg.log_config,
      cfg.get("momentum_config", None),
  )
  runner.register_hook(FreezeHook(trainable), priority="LOW")
  runner.register_hook(GateWeightLoggerHook(), priority="LOW")
  if args.dry_run:
    runner.register_hook(StopAfterItersHook(2), priority="LOWEST")

  try:
    runner.run([loader], [("train", 1)])
  except SystemExit:
    if not args.dry_run:
      raise
    logger.info("Dry run finished: 2 training iterations ran without errors.")


if __name__ == "__main__":
  main()
