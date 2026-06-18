"""Reward components for the run-then-recover task.

Phase structure (Mini ~0.68 m standing head height):
  RUNNING  (head >= 0.62 m): ``running_velocity`` tracks 1 m/s forward.
  TOPPLING (tilt > 0.6):     ``proactive_roll`` converts topple into a roll.
  ROLLING  (head < 0.42 m):  ``roll_momentum`` sustains the roll on the ground.
  IMPACT   (head < 0.30 m):  ``soft_landing`` absorbs the fall.
  RISING   (head < 0.50 m):  ``upward_velocity`` drives the robot back up.

Terms active outside their window return 1.0 — no cross-phase interference.
``upright_progress`` and ``track_head_height`` are always-on; they carry the
lying→standing gradient from any pose, even when the SMP gate is low.

Combined and SMP-gated via ``smp.rl.rewards.task_smp_product``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from smp.rl.tasks.steering.mdp.commands import SteeringCommand

__all__ = [
  "running_velocity",
  "upright_progress",
  "track_head_height",
  "upward_velocity",
  "soft_landing",
  "roll_momentum",
  "proactive_roll",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def running_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  head_run_threshold: float = 0.62,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward forward velocity tracking when upright (head >= threshold), else 1.0.

  ``exp(-vel_err_scale · ‖tar_speed·tar_dir − root_vel_xy‖²)`` when standing;
  zeroed if the robot projects negatively onto the target dir (no backward reward).
  Returns 1.0 when fallen so this term does not compete with recovery rewards.
  """
  asset = env.scene[asset_cfg.name]
  cmd: "SteeringCommand" = env.command_manager.get_term(command_name)  # type: ignore[assignment]

  head_idx = asset.find_sites(["head"], preserve_order=True)[0][0]
  head_z = asset.data.site_pos_w[:, head_idx, 2]

  root_vel_xy = asset.data.root_link_lin_vel_w[:, :2]
  tar_vel = cmd.tar_speed.unsqueeze(-1) * cmd.tar_dir_w
  vel_err = ((tar_vel - root_vel_xy) ** 2).sum(dim=-1)

  proj_speed = (cmd.tar_dir_w * root_vel_xy).sum(dim=-1)
  shaped = torch.exp(-vel_err_scale * vel_err)
  shaped = torch.where(proj_speed < 0, torch.zeros_like(shaped), shaped)

  # Only active when upright; outside the recovery window → 1.0 (no penalty).
  return torch.where(head_z >= head_run_threshold, shaped, torch.ones_like(shaped))


def upright_progress(
  env: "ManagerBasedRlEnv",
  scale: float = 1.0,
) -> torch.Tensor:
  """Always-on monotonic getup potential from torso orientation.

  ``((1 − g_z) / 2) ^ scale`` where ``g_z`` = body-frame z of gravity:
  ``−1`` upright, ``0`` horizontal, ``+1`` inverted → maps to [0, 1].
  Carries the lying→standing gradient from any pose and is never gated.
  """
  robot = env.scene["robot"]
  g_z = robot.data.projected_gravity_b[:, 2]
  uprightness = torch.clamp((1.0 - g_z) * 0.5, 0.0, 1.0)
  return uprightness.pow(scale)


def track_head_height(
  env: "ManagerBasedRlEnv",
  target_height: float = 0.65,
  scale: float = 1.0,
) -> torch.Tensor:
  """Always-on: reward the ``head`` site approaching ``target_height``.

  ``exp(-scale · max(target_height − head_z, 0)²)`` — no penalty for overshoot.
  Needs the ``head`` site from ``getup_env_cfg.get_mini_spec_with_head``.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
  env: "ManagerBasedRlEnv",
  target_velocity: float = 0.40,
  head_height_threshold: float = 0.50,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity while below ``head_height_threshold`` (else 1).

  ``exp(-scale · max(target_velocity − head_vz, 0)²)``. Drives whole-body rising
  motion using the head site velocity rather than just the pelvis.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(
    head_z < head_height_threshold,
    shaped,
    torch.ones_like(shaped),
  )


def soft_landing(
  env: "ManagerBasedRlEnv",
  scale: float = 4.0,
  head_floor_threshold: float = 0.30,
) -> torch.Tensor:
  """Reward low downward pelvis speed during the impact phase (head < threshold).

  ``exp(-scale · max(-v_z, 0)²)`` → 1.0 when stationary or rising, decays sharply
  for high fall speed.  Returns 1.0 above threshold so it doesn't interfere with
  the standup or running phases.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  base_vz = robot.data.root_link_lin_vel_w[:, 2]
  falling = torch.clamp(-base_vz, min=0.0)
  shaped = torch.exp(-scale * falling * falling)
  return torch.where(head_z < head_floor_threshold, shaped, torch.ones_like(shaped))


def proactive_roll(
  env: "ManagerBasedRlEnv",
  tilt_threshold: float = 0.6,
  target_ang_vel: float = 2.0,
  scale: float = 0.5,
) -> torch.Tensor:
  """Reward converting a committed topple into a roll, triggered by TILT.

  Tilt = ``‖projected_gravity_b[:, :2]‖``: 0 upright, 1 horizontal. Once tilt
  exceeds ``tilt_threshold`` the CoM has left the support polygon and the fall
  cannot be arrested by balancing. At that point rewards horizontal angular
  velocity ``|ω_xy|`` reaching ``target_ang_vel`` — going WITH the rotation into
  a roll rather than resisting it rigidly. Returns 1.0 when tilt is safe.
  """
  robot = env.scene["robot"]
  tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
  ang_xy = robot.data.root_link_ang_vel_w[:, :2]
  ang_mag = torch.norm(ang_xy, dim=-1)
  shortfall = torch.clamp(ang_mag - target_ang_vel, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(tilt > tilt_threshold, shaped, torch.ones_like(shaped))


def roll_momentum(
  env: "ManagerBasedRlEnv",
  target_ang_vel: float = 1.5,
  scale: float = 1.0,
  head_roll_threshold: float = 0.42,
) -> torch.Tensor:
  """Reward active rolling (sagittal + lateral angular velocity) while on ground.

  Rewards horizontal angular velocity magnitude (pitch + roll, not yaw) reaching
  ``target_ang_vel`` rad/s. ``exp(-scale · max(target − |ω_xy|, 0)²)`` → 1.0
  when rolling well.  Returns 1.0 above threshold to not penalise the upright robot.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  ang_xy = robot.data.root_link_ang_vel_w[:, :2]
  ang_mag = torch.norm(ang_xy, dim=-1)
  shortfall = torch.clamp(ang_mag - target_ang_vel, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
