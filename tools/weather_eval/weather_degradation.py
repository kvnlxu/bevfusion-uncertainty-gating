from mmdet.datasets.builder import PIPELINES
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch


def rng_for(seed, sample, stream):
  key = f"{seed}|{sample}|{stream}".encode()
  return np.random.default_rng(
      int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
  )


def _to_numpy_image(img):
  if isinstance(img, Image.Image):
    return np.asarray(img)
  if isinstance(img, torch.Tensor):
    arr = img.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
      arr = np.transpose(arr, (1, 2, 0))
    return arr
  return np.asarray(img)


def _restore_image_type(arr, orig_img):
  uint8_arr = np.clip(arr, 0, 255).astype(np.uint8)
  if isinstance(orig_img, Image.Image):
    return Image.fromarray(uint8_arr)
  if isinstance(orig_img, torch.Tensor):
    t = torch.from_numpy(uint8_arr)
    if orig_img.ndim == 3 and orig_img.shape[0] in (1, 3):
      t = t.permute(2, 0, 1).contiguous()
    return t.to(dtype=orig_img.dtype, device=orig_img.device)
  return arr


def approximate_depth(xyz, projection, h, w):
  # Project LiDAR into original camera pixel coordinates before ImageAug3D.
  p = np.asarray(projection, dtype=np.float64)
  if p.shape not in ((4, 4), (3, 4)):
    raise ValueError(f"Unexpected lidar2img shape {p.shape}")
  q = np.c_[xyz, np.ones(len(xyz))] @ p[:3, :].T
  good = np.isfinite(q).all(1) & (q[:, 2] > 0.5)
  q = q[good]
  u, v = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
  valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
  u, v, z = u[valid], v[valid], q[valid, 2]

  # Closest return per 8-pixel cell, followed by nearest-depth filling.
  gh, gw = (h + 7) // 8, (w + 7) // 8
  grid = np.full((gh, gw), np.inf, dtype=np.float32)
  np.minimum.at(
      grid, (v.astype(int) // 8, u.astype(int) // 8), z.astype(np.float32)
  )
  missing = ~np.isfinite(grid)
  if missing.all():
    raise ValueError(
        "No projected LiDAR points in camera; inspect calibration and sample."
    )
  nearest = distance_transform_edt(
      missing, return_distances=False, return_indices=True
  )
  grid = grid[tuple(nearest)]
  return cv2.resize(
      np.clip(grid, 1, 100), (w, h), interpolation=cv2.INTER_LINEAR
  )


def _as_4x4(mat):
  arr = np.asarray(mat, dtype=np.float64)
  if arr.shape == (4, 4):
    return arr
  if arr.shape == (3, 4):
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = arr
    return out
  raise ValueError(f"Expected (4, 4) or (3, 4) transform, got {arr.shape}")


def _k_3x3(k):
  arr = np.asarray(k, dtype=np.float64)
  if arr.shape == (3, 3):
    return arr
  if arr.shape == (4, 4):
    return arr[:3, :3]
  raise ValueError(f"Expected (3, 3) or (4, 4) intrinsic matrix, got {arr.shape}")


def _get_lidar2img_projections(results, num_views):
  if "lidar2img" in results and results["lidar2img"] is not None:
    projs = list(results["lidar2img"])
    assert len(projs) == num_views, (
        f"Expected {num_views} lidar2img matrices, got {len(projs)}"
    )
    return projs

  intrinsics = results.get("cam_intrinsic") or results.get("camera_intrinsics")
  if intrinsics is not None and len(intrinsics) == num_views:
    if (
        "lidar2cam" in results
        and results["lidar2cam"] is not None
        and len(results["lidar2cam"]) == num_views
    ):
      projs = []
      for k, l2c in zip(intrinsics, results["lidar2cam"]):
        ext34 = _as_4x4(l2c)[:3, :4]
        projs.append(_k_3x3(k) @ ext34)
      return projs

    cam2ego = results.get("cam2ego") or results.get("camera2ego")
    lidar2ego = results.get("lidar2ego")
    if cam2ego is not None and lidar2ego is not None and len(cam2ego) == num_views:
      l2e = _as_4x4(lidar2ego)
      projs = []
      for k, c2e in zip(intrinsics, cam2ego):
        l2c = np.linalg.inv(_as_4x4(c2e)) @ l2e
        projs.append(_k_3x3(k) @ l2c[:3, :4])
      return projs

  raise KeyError(
      "Missing 'lidar2img' and could not derive projections from available"
      f" keys: {list(results.keys())}"
  )


def camera_weather(image, xyz, projection, weather, s, rng):
  img_arr = _to_numpy_image(image)
  out_dtype = (
      img_arr.dtype if np.issubdtype(img_arr.dtype, np.floating) else np.uint8
  )
  if s == 0 or weather == "clean":
    return img_arr.copy()
  h, w = img_arr.shape[:2]
  x = img_arr.astype(np.float32)
  if weather == "fog":
    d = approximate_depth(xyz, projection, h, w)
    t = np.exp(-0.065 * s * d)[..., None]
    x = x * t + 220 * (1 - t)
  elif weather == "rain":
    veil = min(0.28 * s, 1.0)  # capped so severities > 1 stay a valid blend
    x = x * (1 - veil) + 95 * veil
    mask = np.zeros((h, w), np.uint8)
    n = int(0.0015 * h * w * s)
    for u, v in zip(rng.integers(0, w, n), rng.integers(0, h, n)):
      length = int(rng.integers(12, 32))
      cv2.line(
          mask, (int(u), int(v)), (int(u + length // 4), int(v + length)), 255, 1
      )
    a = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255 * 0.65
    x = x * (1 - a) + 205 * a
  elif weather == "snow":
    veil = min(0.35 * s, 1.0)  # capped so severities > 1 stay a valid blend
    x = x * (1 - veil) + 235 * veil
    mask = np.zeros((h, w), np.uint8)
    n = int(0.00035 * h * w * s)
    for u, v in zip(rng.integers(0, w, n), rng.integers(0, h, n)):
      cv2.circle(mask, (int(u), int(v)), int(rng.integers(1, 6)), 255, -1)
    a = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255 * 0.85
    x = x * (1 - a) + 255 * a
  return np.clip(x, 0, 255).astype(out_dtype)


def lidar_keep(xyz, weather, s, rng):
  if s == 0 or weather == "clean":
    return np.ones(len(xyz), dtype=bool)
  r = np.linalg.norm(xyz, axis=1)
  beta = {"fog": 0.045, "rain": 0.018, "snow": 0.030}[weather]
  p = np.exp(-beta * s * r)
  return rng.random(len(xyz)) < p


MODALITIES = ("both", "camera", "lidar")
# Severity 1.0 was the original maximum; values up to 3.0 push into heavier
# weather on the same scale to locate the robustness cliff.
MAX_SEVERITY = 3.0


@PIPELINES.register_module()
class WeatherDegradation(object):
  """Synthetic weather for nuScenes camera images and LiDAR points.

  Args:
    weather: "clean", "fog", "rain" or "snow".
    severity: Strength in [0, MAX_SEVERITY]; 0 is a no-op (clean baseline).
      Coefficients are unchanged from the original [0, 1] scale, so values
      above 1 extend the same curves into heavier weather.
    seed: Base seed; per-sample RNG streams are derived from it.
    modality: Which sensors to degrade: "both" (default), "camera" only or
      "lidar" only. Single-sensor runs isolate which modality drives the
      robustness cliff.
    debug_dir: If set, write degraded images, point clouds and stats.json.
    random_cfg: Training mode. If set, every call draws its own condition
      and ignores weather/severity/modality/seed. Keys (all optional):
        clean_prob (0.25): probability of leaving the sample clean
        weathers (["fog", "rain", "snow"]): weathers to draw from uniformly
        severity_range ([0.0, MAX_SEVERITY]): uniform severity range
        modality_probs ({"both": 0.5, "camera": 0.25, "lidar": 0.25})
      Draws use NumPy's global RNG, which mmdet's dataloader seeds per worker
      and epoch, so conditions differ across epochs but runs are reproducible.
  """

  def __init__(
      self, weather="clean", severity=0.0, seed=42, modality="both",
      debug_dir=None, random_cfg=None
  ):
    if weather not in ("clean", "fog", "rain", "snow"):
      raise ValueError(f"Invalid weather {weather!r}")
    if not 0 <= severity <= MAX_SEVERITY:
      raise ValueError(f"Severity must be in [0, {MAX_SEVERITY}], got {severity}")
    if modality not in MODALITIES:
      raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
    self.weather = weather
    self.s = float(severity)
    self.seed = int(seed)
    self.modality = modality
    self.debug_dir = Path(debug_dir) if debug_dir else None
    self.random_cfg = None
    if random_cfg is not None:
      cfg = dict(random_cfg)
      self.random_cfg = dict(
          clean_prob=float(cfg.get("clean_prob", 0.25)),
          weathers=list(cfg.get("weathers", ["fog", "rain", "snow"])),
          severity_range=tuple(cfg.get("severity_range", (0.0, MAX_SEVERITY))),
          modality_probs=dict(
              cfg.get("modality_probs", {"both": 0.5, "camera": 0.25, "lidar": 0.25})
          ),
      )
      rc = self.random_cfg
      lo, hi = rc["severity_range"]
      if not 0 <= lo <= hi <= MAX_SEVERITY:
        raise ValueError(f"severity_range must lie in [0, {MAX_SEVERITY}]")
      if not set(rc["weathers"]) <= {"fog", "rain", "snow"}:
        raise ValueError(f"Invalid weathers {rc['weathers']}")
      if not set(rc["modality_probs"]) <= set(MODALITIES):
        raise ValueError(f"Invalid modality_probs keys {list(rc['modality_probs'])}")

  def _draw_condition(self):
    """Random (weather, severity, modality) for one training sample."""
    rc = self.random_cfg
    if np.random.rand() < rc["clean_prob"]:
      return "clean", 0.0, "both"
    weather = rc["weathers"][np.random.randint(len(rc["weathers"]))]
    lo, hi = rc["severity_range"]
    severity = float(np.random.uniform(lo, hi))
    names = list(rc["modality_probs"])
    probs = np.array([rc["modality_probs"][n] for n in names], dtype=np.float64)
    modality = names[np.random.choice(len(names), p=probs / probs.sum())]
    return weather, severity, modality

  def transform(self, results):
    raw_images = results["img"]
    images = [_to_numpy_image(im) for im in raw_images]
    points = results["points"]
    assert len(images) == 6, "Expected six camera views"
    projections = _get_lidar2img_projections(results, len(images))

    xyz = points.tensor[:, :3].detach().cpu().numpy()
    sample = str(
        results.get("token") or results.get("sample_idx") or results["img_path"]
    )
    if self.random_cfg is not None:
      weather, s, modality = self._draw_condition()
      seed = int(np.random.randint(2**31 - 1))
    else:
      weather, s, modality, seed = self.weather, self.s, self.modality, self.seed
    # RNG streams are keyed by sample and sensor only, so the camera-only and
    # lidar-only runs apply exactly the same corruption as the "both" run.
    if modality in ("both", "camera"):
      changed = [
          camera_weather(
              im,
              xyz,
              p,
              weather,
              s,
              rng_for(seed, sample, f"{weather}-cam-{i}"),
          )
          for i, (im, p) in enumerate(zip(images, projections))
      ]
    else:
      changed = [im.copy() for im in images]
    if modality in ("both", "lidar"):
      keep = lidar_keep(
          xyz,
          weather,
          s,
          rng_for(seed, sample, f"{weather}-lidar"),
      )
    else:
      keep = np.ones(len(xyz), dtype=bool)

    if s == 0 or weather == "clean":
      assert (
          all(np.array_equal(a, b) for a, b in zip(images, changed))
          and keep.all()
      )

    # Record the applied condition; with "weather_cond" in Collect3D's
    # meta_lis_keys it reaches the model, where GatedConvFuser derives a camera
    # reliability target from it (supervised gate, Phase 4).
    results["weather_cond"] = dict(
        weather=weather, severity=float(s), modality=modality
    )

    # BasePoints slicing preserves all channels, including sweep time and intensity.
    results["points"] = points[
        torch.as_tensor(keep, device=points.tensor.device)
    ]
    results["img"] = [
        _restore_image_type(c, orig) for c, orig in zip(changed, raw_images)
    ]

    if self.debug_dir:
      self.debug_dir.mkdir(parents=True, exist_ok=True)
      for i, im in enumerate(changed):
        cv2.imwrite(
            str(self.debug_dir / f"camera_{i}.jpg"),
            np.clip(im, 0, 255).astype(np.uint8),
        )
      np.save(self.debug_dir / "lidar_clean_xyz.npy", xyz)
      np.save(self.debug_dir / "lidar_weather_xyz.npy", xyz[keep])
      stats = dict(
          sample=sample,
          weather=weather,
          severity=s,
          seed=seed,
          modality=modality,
          clean_points=len(xyz),
          retained_points=int(keep.sum()),
          retained_fraction=float(keep.mean()) if len(keep) else 0.0,
          camera_mean_abs_change=[
              float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
              for a, b in zip(images, changed)
          ],
      )
      (self.debug_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return results

  def __call__(self, results):
    return self.transform(results)