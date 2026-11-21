from mmdet.datasets.builder import PIPELINES
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch


def rng_for(seed, sample, stream):
  key = f"{seed}|{sample}|{stream}".encode()
  return np.random.default_rng(
      int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
  )


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
    assert len(projs) == num_views, f"Expected {num_views} lidar2img matrices, got {len(projs)}"
    return projs

  intrinsics = results.get("cam_intrinsic") or results.get("camera_intrinsics")
  if intrinsics is not None and len(intrinsics) == num_views:
    if "lidar2cam" in results and results["lidar2cam"] is not None and len(results["lidar2cam"]) == num_views:
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
      f"Missing 'lidar2img' and could not derive projections from available keys: {list(results.keys())}"
  )


def camera_weather(image, xyz, projection, weather, s, rng):
  if s == 0 or weather == "clean":
    return image.copy()
  h, w = image.shape[:2]
  x = image.astype(np.float32)
  if weather == "fog":
    d = approximate_depth(xyz, projection, h, w)
    t = np.exp(-0.065 * s * d)[..., None]
    x = x * t + 220 * (1 - t)
  elif weather == "rain":
    x = x * (1 - 0.28 * s) + 95 * (0.28 * s)
    mask = np.zeros((h, w), np.uint8)
    n = int(0.0015 * h * w * s)
    for u, v in zip(rng.integers(0, w, n), rng.integers(0, h, n)):
      length = int(rng.integers(12, 32))
      cv2.line(mask, (int(u), int(v)), (int(u + length // 4), int(v + length)), 255, 1)
    a = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255 * 0.65
    x = x * (1 - a) + 205 * a
  elif weather == "snow":
    x = x * (1 - 0.35 * s) + 235 * (0.35 * s)
    mask = np.zeros((h, w), np.uint8)
    n = int(0.00035 * h * w * s)
    for u, v in zip(rng.integers(0, w, n), rng.integers(0, h, n)):
      cv2.circle(mask, (int(u), int(v)), int(rng.integers(1, 6)), 255, -1)
    a = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255 * 0.85
    x = x * (1 - a) + 255 * a
  return np.clip(x, 0, 255).astype(image.dtype)


def lidar_keep(xyz, weather, s, rng):
  if s == 0 or weather == "clean":
    return np.ones(len(xyz), dtype=bool)
  r = np.linalg.norm(xyz, axis=1)
  beta = {"fog": 0.045, "rain": 0.018, "snow": 0.030}[weather]
  p = np.exp(-beta * s * r)
  return rng.random(len(xyz)) < p


@PIPELINES.register_module()
class WeatherDegradation(object):

  def __init__(
      self, weather="clean", severity=0.0, seed=42, debug_dir=None
  ):
    if weather not in ("clean", "fog", "rain", "snow") or not 0 <= severity <= 1:
      raise ValueError("Invalid weather or severity")
    self.weather = weather
    self.s = float(severity)
    self.seed = int(seed)
    self.debug_dir = Path(debug_dir) if debug_dir else None

  def transform(self, results):
    images = results["img"]
    points = results["points"]
    assert len(images) == 6, "Expected six camera views"
    projections = _get_lidar2img_projections(results, len(images))

    xyz = points.tensor[:, :3].detach().cpu().numpy()
    sample = str(
        results.get("token") or results.get("sample_idx") or results["img_path"]
    )
    changed = [
        camera_weather(
            im,
            xyz,
            p,
            self.weather,
            self.s,
            rng_for(self.seed, sample, f"{self.weather}-cam-{i}"),
        )
        for i, (im, p) in enumerate(zip(images, projections))
    ]
    keep = lidar_keep(
        xyz,
        self.weather,
        self.s,
        rng_for(self.seed, sample, f"{self.weather}-lidar"),
    )

    if self.s == 0 or self.weather == "clean":
      assert (
          all(np.array_equal(a, b) for a, b in zip(images, changed))
          and keep.all()
      )

    # BasePoints slicing preserves all channels, including sweep time and intensity.
    results["points"] = points[
        torch.as_tensor(keep, device=points.tensor.device)
    ]
    results["img"] = changed

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
          weather=self.weather,
          severity=self.s,
          seed=self.seed,
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