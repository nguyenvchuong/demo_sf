"""Logged safety / contact metrics for the getup task."""

from __future__ import annotations

from functools import partial

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor.contact_sensor import ContactSensor

from smp.robot.Mini_M1v3.mini_m13_constants import MINI_M1V3_XML

__all__ = [
  "peak_ground_contact_force",
  "peak_link_contact_force",
  "ground_contacting_bodies",
  "head_height",
  "pelvis_downward_speed",
  "mini_robot_body_names",
  "make_per_link_peak_force_metrics",
]


def _ground_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> ContactSensor:
  return env.scene.sensors[sensor_name]


def mini_robot_body_names() -> list[str]:
  """Body names on Mini_M1v3 (excluding the world body)."""
  spec = mujoco.MjSpec.from_file(str(MINI_M1V3_XML))
  return [body.name for body in spec.bodies[1:]]


def peak_ground_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Peak ground contact force magnitude across all robot bodies this step (N)."""
  data = _ground_sensor(env, sensor_name).data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device)
  force_norm = torch.norm(data.force, dim=-1)
  return torch.max(force_norm, dim=-1)[0]


def peak_link_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  link_name: str,
) -> torch.Tensor:
  """Peak ground contact force on one robot link this step (N)."""
  sensor = _ground_sensor(env, sensor_name)
  data = sensor.data
  if data.force is None:
    return torch.zeros(env.num_envs, device=env.device)
  try:
    link_idx = sensor.primary_names.index(link_name)
  except ValueError:
    return torch.zeros(env.num_envs, device=env.device)
  return torch.norm(data.force[:, link_idx], dim=-1)


def ground_contacting_bodies(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Number of robot bodies in contact with terrain this step."""
  data = _ground_sensor(env, sensor_name).data
  if data.found is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (data.found > 0).sum(dim=-1).float()


def head_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Head site height above world floor (m)."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  return robot.data.site_pos_w[:, head_idx, 2]


def pelvis_downward_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Positive when pelvis is moving downward (m/s)."""
  robot = env.scene["robot"]
  vz = robot.data.root_link_lin_vel_w[:, 2]
  return torch.clamp(-vz, min=0.0)


def _metric_key_for_link(link_name: str) -> str:
  short = link_name.removesuffix("_link") if link_name.endswith("_link") else link_name
  return f"force_{short}_N"


def make_per_link_peak_force_metrics(sensor_name: str) -> dict[str, MetricsTermCfg]:
  """One scalar metric per link for the Viser Metrics tab."""
  metrics: dict[str, MetricsTermCfg] = {}
  for link_name in mini_robot_body_names():
    metrics[_metric_key_for_link(link_name)] = MetricsTermCfg(
      func=partial(peak_link_contact_force, link_name=link_name),
      params={"sensor_name": sensor_name},
    )
  return metrics
