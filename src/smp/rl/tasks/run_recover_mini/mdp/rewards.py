"""Reward components for the run-then-recover task.

Phase structure (Mini ~0.68 m standing head height):
  RUNNING  (head >= 0.62 m): ``running_velocity``, ``gait_symmetry``, ``running_height``
  TOPPLING (tilt > 0.6):     ``proactive_roll`` converts topple into a roll.
  ROLLING  (head < 0.42 m):  ``roll_momentum`` sustains the roll on the ground.
  IMPACT   (head < 0.30 m):  ``soft_landing`` + ``head_safe_landing`` absorb fall.
  RISING   (head < 0.50 m):  ``upward_velocity`` drives the robot back up.

Running-phase terms return 1.0 when fallen (no cross-phase interference).
Impact-phase ``head_safe_landing`` returns 0.0 outside its window.
``upright_progress`` and ``track_head_height`` are always-on.
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
  "gait_symmetry",
  "running_height",
  "head_safe_landing",
  "upright_progress",
  "track_head_height",
  "upward_velocity",
  "soft_landing",
  "roll_momentum",
  "proactive_roll",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Ordered left / right joint name pairs for symmetry reward.
# Convention: dq_left + dq_right ≈ 0 for all pairs during symmetric running
# (anti-phase pitch, mirrored roll/yaw convention in Mini_M1v1 XML).
_L_JOINTS: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_yaw_joint",
)
_R_JOINTS: tuple[str, ...] = (
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_yaw_joint",
)


# ─── Running-phase rewards ────────────────────────────────────────────────────

def running_velocity(
  env: "ManagerBasedRlEnv",
  command_name: str,
  vel_err_scale: float = 0.5,
  head_run_threshold: float = 0.62,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Ungated forward velocity tracking reward.

  ``exp(-vel_err_scale · ‖tar_speed·tar_dir − root_vel_xy‖²)`` when upright;
  0.0 when fallen — robot loses the running reward while down, creating strong
  incentive to stand up and run again.
  Zeroed when root velocity projects negatively onto the target dir.

  Designed as a STANDALONE RewardTermCfg (not inside task_smp_product) so it
  is never discounted by the SMP gate.
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

  return torch.where(head_z >= head_run_threshold, shaped, torch.zeros_like(shaped))


def gait_symmetry(
  env: "ManagerBasedRlEnv",
  scale: float = 0.1,
  head_run_threshold: float = 0.62,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward left-right anti-phase joint velocity symmetry during running.

  For symmetric bipedal gait each joint pair satisfies ``dq_left + dq_right ≈ 0``
  (anti-phase for pitch, mirrored convention for roll/yaw).
  Reward = ``exp(-scale · mean‖dq_L + dq_R‖²)`` over all pairs.
  Returns 1.0 when fallen so it does not interfere with ukemi.

  Joint indices are cached on ``env`` after the first call.
  """
  asset = env.scene[asset_cfg.name]
  head_idx = asset.find_sites(["head"], preserve_order=True)[0][0]
  head_z = asset.data.site_pos_w[:, head_idx, 2]

  if not hasattr(env, "_symm_l_idx"):
    l_ids, _ = asset.find_joints(list(_L_JOINTS), preserve_order=True)
    r_ids, _ = asset.find_joints(list(_R_JOINTS), preserve_order=True)
    env._symm_l_idx = torch.tensor(l_ids, device=env.device, dtype=torch.long)  # type: ignore[attr-defined]
    env._symm_r_idx = torch.tensor(r_ids, device=env.device, dtype=torch.long)  # type: ignore[attr-defined]

  dq = asset.data.joint_vel
  dq_l = dq[:, env._symm_l_idx]  # type: ignore[attr-defined]
  dq_r = dq[:, env._symm_r_idx]  # type: ignore[attr-defined]

  asym = (dq_l + dq_r).pow(2).mean(dim=-1)
  shaped = torch.exp(-scale * asym)

  return torch.where(head_z >= head_run_threshold, shaped, torch.ones_like(shaped))


def running_height(
  env: "ManagerBasedRlEnv",
  target_height: float = 0.30,
  scale: float = 30.0,
  head_run_threshold: float = 0.62,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward proper pelvis height during running — not squatting on thighs.

  Mini_M1v1 standing pelvis height ≈ 0.34–0.37 m.  Penalises the robot for
  crouching below ``target_height`` while upright.  One-sided: no penalty for
  standing taller than target.  Returns 1.0 when fallen.
  """
  asset = env.scene[asset_cfg.name]
  head_idx = asset.find_sites(["head"], preserve_order=True)[0][0]
  head_z = asset.data.site_pos_w[:, head_idx, 2]

  pelvis_z = asset.data.root_link_pos_w[:, 2]
  shortfall = torch.clamp(pelvis_z - target_height, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)

  return torch.where(head_z >= head_run_threshold, shaped, torch.ones_like(shaped))


# ─── Contact / landing reward ─────────────────────────────────────────────────

def head_safe_landing(
  env: "ManagerBasedRlEnv",
  scale: float = 6.0,
  head_floor_threshold: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalise the HEAD diving fast into the ground during impact phase.

  A robot performing correct ukemi rolls on its shoulder / back — the head is
  tucked and the downward head velocity stays near zero even while the body
  makes contact.  A head-first fall produces large negative ``head_vz``.

  ``exp(-scale · max(-head_vz, 0)²)`` when head < threshold; 0.0 outside the
  window.  Returning 0 outside means no free reward when not falling — the
  policy must earn this term by landing safely.

  Complements ``soft_landing`` (which tracks pelvis velocity): together they
  discourage both whole-body hard impacts and head-first falls specifically.
  """
  asset = env.scene[asset_cfg.name]
  head_idx = asset.find_sites(["head"], preserve_order=True)[0][0]
  head_z = asset.data.site_pos_w[:, head_idx, 2]
  head_vz = asset.data.site_lin_vel_w[:, head_idx, 2]

  falling_head = torch.clamp(-head_vz, min=0.0)
  shaped = torch.exp(-scale * falling_head * falling_head)

  return torch.where(head_z < head_floor_threshold, shaped, torch.zeros_like(shaped))


# ─── Always-on recovery rewards ───────────────────────────────────────────────

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
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


# ─── Phase-gated recovery rewards ────────────────────────────────────────────

def upward_velocity(
  env: "ManagerBasedRlEnv",
  target_velocity: float = 0.40,
  head_height_threshold: float = 0.50,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity while below ``head_height_threshold`` (else 1)."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  head_vz = robot.data.site_lin_vel_w[:, head_idx, 2]
  shortfall = torch.clamp(head_vz - target_velocity, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(head_z < head_height_threshold, shaped, torch.ones_like(shaped))


def soft_landing(
  env: "ManagerBasedRlEnv",
  scale: float = 4.0,
  head_floor_threshold: float = 0.30,
) -> torch.Tensor:
  """Reward low downward PELVIS speed during the impact phase (head < threshold).

  ``exp(-scale · max(-v_z, 0)²)`` → 1.0 when stationary or rising.
  Returns 1.0 above threshold so it doesn't interfere with running.
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
  """Reward converting a committed topple into a roll — triggered by TILT.

  Once tilt (= sin of lean angle) exceeds ``tilt_threshold`` the CoM has left
  the support polygon.  Rewards horizontal angular velocity reaching
  ``target_ang_vel`` rad/s — going with the rotation into a roll.
  Returns 1.0 below threshold so balanced running is never disturbed.
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
  ``target_ang_vel`` rad/s.  Returns 1.0 above threshold to not disturb running.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  ang_xy = robot.data.root_link_ang_vel_w[:, :2]
  ang_mag = torch.norm(ang_xy, dim=-1)
  shortfall = torch.clamp(ang_mag - target_ang_vel, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
