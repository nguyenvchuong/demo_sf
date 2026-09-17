"""Reward components for the air-drop ukemi + getup task.

Two phases, gated by the ``env._drop_landed`` contact latch (set by
``smp.rl.events.update_landed_latch``):

  AIRBORNE  (not landed): ``hold_initial_pose`` rewards keeping the joints at the
                          spawn configuration — the robot braces and holds the
                          random pose it was dropped in, instead of flailing.
  GROUNDED  (landed):     the ukemi/getup terms activate — ``soft_landing`` +
                          ``roll_momentum`` + ``rolling_contact_force`` spread and
                          minimise the impact, then ``track_head_height`` +
                          ``upright_progress`` bring it back to a stand.

Each term returns 1.0 outside its own phase so the weighted task sum (weights = 1)
never drops just because a term is inactive — only the active phase shapes reward.
The grounded terms are thin wrappers over the shared ``getup_mini`` rewards with an
extra landed gate, so the impact/roll/recovery logic stays in one place.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from smp.rl.tasks.getup_mini.mdp import rewards as _getup

__all__ = [
  "hold_initial_pose",
  "proactive_roll",
  "soft_landing",
  "roll_momentum",
  "forward_roll",
  "rolling_contact_force",
  "track_head_height",
  "upright_progress",
  "upward_velocity",
  "feet_first_contact",
  "avoid_upper_impact",
  "foot_flat",
  "hands_forward",
  "knee_bend",
  "head_tuck",
]


def _landed_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Per-env "has touched the ground" latch (False everywhere until the latch event
  runs / before the first touchdown)."""
  landed = getattr(env, "_drop_landed", None)
  if landed is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return landed


def _gate_grounded(env: ManagerBasedRlEnv, value: torch.Tensor) -> torch.Tensor:
  """Return ``value`` where the env has landed, else 1.0 (neutral while airborne)."""
  return torch.where(_landed_mask(env), value, torch.ones_like(value))


def _gate_airborne(env: ManagerBasedRlEnv, value: torch.Tensor) -> torch.Tensor:
  """Return ``value`` while the env is AIRBORNE (not yet landed), else 1.0."""
  return torch.where(_landed_mask(env), torch.ones_like(value), value)


def hold_initial_pose(
  env: ManagerBasedRlEnv,
  scale: float = 8.0,
) -> torch.Tensor:
  """Reward holding the spawn joint configuration WHILE AIRBORNE (not yet landed).

  ``exp(-scale · mean((q − q_spawn)²))`` over all joints — 1.0 when the pose is held
  exactly, decaying as the limbs drift from the dropped pose. ``q_spawn`` is the
  configuration captured at reset by ``reset_drop_in_air`` (``env._drop_spawn_joint_pos``).
  Returns 1.0 once the env has landed so it does not fight the ukemi/getup phase.
  """
  robot = env.scene["robot"]
  spawn = getattr(env, "_drop_spawn_joint_pos", None)
  if spawn is None:
    return torch.ones(env.num_envs, device=env.device)
  dev = ((robot.data.joint_pos - spawn) ** 2).mean(dim=-1)
  shaped = torch.exp(-scale * dev)
  # Active only while airborne; neutral (1.0) after touchdown.
  return torch.where(_landed_mask(env), torch.ones_like(shaped), shaped)


def proactive_roll(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.proactive_roll`` gated to the AIRBORNE phase (1.0 after landing).

  Rewards tilting the torso and developing horizontal angular velocity WHILE
  FALLING so the robot enters the roll naturally at touchdown, instead of
  bracing rigidly and absorbing the full impact in a single spike.
  The tilt gate inside ``getup_mini.proactive_roll`` (tilt > tilt_threshold)
  ensures the term only fires when the body has committed to the fall — it
  cannot be farmed by spinning upright.
  """
  return _gate_airborne(env, _getup.proactive_roll(env, **kwargs))


def soft_landing(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.soft_landing`` gated to the GROUNDED phase (1.0 while airborne)."""
  return _gate_grounded(env, _getup.soft_landing(env, **kwargs))


def roll_momentum(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.roll_momentum`` gated to the GROUNDED phase (1.0 while airborne)."""
  return _gate_grounded(env, _getup.roll_momentum(env, **kwargs))


def rolling_contact_force(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.rolling_contact_force`` gated to the GROUNDED phase."""
  return _gate_grounded(env, _getup.rolling_contact_force(env, **kwargs))


def track_head_height(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.track_head_height`` gated to the GROUNDED phase — only stand up
  AFTER landing (airborne the robot holds its pose, it does not reach for height)."""
  return _gate_grounded(env, _getup.track_head_height(env, **kwargs))


def upward_velocity(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.upward_velocity`` gated to the GROUNDED phase.

  Rewards UPWARD head velocity while the head is still low (below the threshold),
  so once on the ground the robot is actively driven to rise QUICKLY rather than
  lie still.  This is the term that breaks the "lie motionless and farm the
  neutral reward" local optimum — it is ~0 for a stationary fallen robot and only
  pays out while the head is accelerating upward toward a stand."""
  return _gate_grounded(env, _getup.upward_velocity(env, **kwargs))


def upright_progress(env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
  """``getup_mini.upright_progress`` gated to the GROUNDED phase."""
  return _gate_grounded(env, _getup.upright_progress(env, **kwargs))


def avoid_upper_impact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_force: float = 80.0,
  scale: float = 1.0,
) -> torch.Tensor:
  """Penalise hard ground contact through the UPPER body (torso / pelvis / arms) —
  i.e. catching the fall on the BACK instead of landing on the feet.

  Reads the peak contact force on the upper-body contact sensor (``sensor_name``,
  a robot-vs-terrain ``ContactSensorCfg`` whose primaries are the torso, pelvis,
  shoulders and arms) and rewards keeping it low:
  ``exp(-scale·(peak_force / max_force)²)``.  A light rolling shoulder/back touch
  during a proper ukemi (low force) is barely penalised; a hard back-slam (high
  force) collapses the reward toward 0.  This is the user's "không đỡ bằng lưng"
  objective — distinct from ``rolling_contact_force`` (which caps the peak over
  ALL bodies including the feet); this one targets the upper body specifically so
  feet contact stays cheap while back contact is expensive.  Gated to the GROUNDED
  phase (1.0 while airborne).
  """
  sensor = env.scene.sensors[sensor_name]
  force_norm = torch.norm(sensor.data.force, dim=-1)  # [B, N]
  peak_force = torch.max(force_norm, dim=-1)[0]  # [B]
  excess = peak_force / max_force
  shaped = torch.exp(-scale * excess * excess)
  return _gate_grounded(env, shaped)


def feet_first_contact(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  upper_sensor_name: str,
  sharpness: float = 1.0,
) -> torch.Tensor:
  """Reward the FEET being the dominant ground contact over the upper body — i.e.
  landing and staying on the feet, not flopping onto the back.

  Compares the peak feet contact force to the peak upper-body contact force and
  rewards the feet share:
  ``(feet / (feet + upper))^sharpness`` → 1.0 when only the feet touch, → 0.0 when
  only the back/torso touches.  This directly encodes "tiếp đất 2 chân trước": at
  touchdown the feet should carry the contact.  ``sharpness`` > 1 makes the reward
  fall off faster as soon as the upper body starts taking load.  Gated to the
  GROUNDED phase, and returns 1.0 when neither set is in contact (mid-roll flight
  phase) so it never penalises the airborne arc of a roll.
  """
  feet_sensor = env.scene.sensors[feet_sensor_name]
  upper_sensor = env.scene.sensors[upper_sensor_name]
  feet_peak = torch.norm(feet_sensor.data.force, dim=-1).max(dim=-1)[0]  # [B]
  upper_peak = torch.norm(upper_sensor.data.force, dim=-1).max(dim=-1)[0]  # [B]
  total = feet_peak + upper_peak
  share = feet_peak / (total + 1e-3)
  shaped = torch.clamp(share, 0.0, 1.0).pow(sharpness)
  # No contact at all → neutral 1.0 (don't penalise the airborne roll arc).
  no_contact = total < 1.0
  shaped = torch.where(no_contact, torch.ones_like(shaped), shaped)
  return _gate_grounded(env, shaped)


def knee_bend(
  env: ManagerBasedRlEnv,
  target_flex: float = 1.2,
  scale: float = 2.0,
  head_roll_threshold: float = 0.62,
) -> torch.Tensor:
  """Reward BENDING THE KNEES while the body is low (rolling) — "knee phải cong".

  After the feet land, the knees must flex to absorb the impact and lower the body
  into the roll (a stiff, straight-legged catch is exactly what we want to avoid).
  Rewards the mean knee-flexion magnitude approaching ``target_flex`` rad:
  ``exp(-scale·max(target_flex − |knee|, 0)²)`` → 1.0 once the knees are bent at
  least ``target_flex`` (deeper is fine), decaying toward straight legs. Active only
  while the head is below ``head_roll_threshold`` (the rolling/absorb phase) and
  GROUNDED, so it does not fight the final straight-legged stand."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  knee_idx = robot.find_joints([".*_knee_joint"], preserve_order=True)[0]
  knee = robot.data.joint_pos[:, knee_idx]  # [B, 2]
  flex = knee.abs().mean(dim=-1)  # [B] mean flexion magnitude
  shortfall = torch.clamp(target_flex - flex, min=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  shaped = torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
  return _gate_grounded(env, shaped)


def head_tuck(
  env: ManagerBasedRlEnv,
  target_drop: float = 0.30,
) -> torch.Tensor:
  """Reward the HEAD DIVING below the pelvis — "đầu chúi xuống quay 1 vòng".

  A proper forward roll tucks the head down and forward so the body rolls over the
  shoulders/back; a stiff topple keeps the head up. Rewards how far the head drops
  BELOW the pelvis: ``clamp((pelvis_z − head_z) / target_drop, 0, 1)`` → 0 when the
  head is up (standing / head above pelvis — naturally self-gating), → 1 once the
  head is ``target_drop`` m below the pelvis (head tucked under = mid-roll
  inversion). Gated to GROUNDED so the airborne arc is never rewarded."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  pelvis_z = robot.data.root_link_pos_w[:, 2]
  drop = pelvis_z - head_z  # > 0 when the head is below the pelvis (tucked/diving)
  shaped = torch.clamp(drop / target_drop, 0.0, 1.0)
  return _gate_grounded(env, shaped)


def forward_roll(
  env: ManagerBasedRlEnv,
  target_ang_vel: float = 1.5,
  scale: float = 1.0,
  head_roll_threshold: float = 1.2,
) -> torch.Tensor:
  """Reward rolling FORWARD specifically — "cơ thể cố roll về đằng trước".

  Uses the BODY-frame pitch rate (``root_link_ang_vel_b[:, 1]``): with the robot
  facing +x (upright spawn + forward-only throw), a positive value is the body
  pitching nose-down/forward — a forward somersault. Rewards it reaching
  ``target_ang_vel``: ``exp(-scale·max(target − pitch_fwd, 0)²)`` → high for a
  forward roll, ~0 for a backward roll or a sideways topple (which have low/negative
  forward pitch). Replaces the direction-agnostic ``roll_momentum`` (which rewarded
  any |ω_xy|). Active while head < ``head_roll_threshold`` so it covers the whole
  roll and the rise, and GROUNDED (1.0 airborne)."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  pitch_fwd = robot.data.root_link_ang_vel_b[:, 1]  # >0 = forward pitch (nose down)
  shortfall = torch.clamp(target_ang_vel - pitch_fwd, min=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  shaped = torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
  return _gate_grounded(env, shaped)


def foot_flat(
  env: ManagerBasedRlEnv,
  feet_sensor_name: str,
  scale: float = 1.0,
  contact_threshold: float = 5.0,
) -> torch.Tensor:
  """Reward landing on the FLAT SOLE — "mặt bàn chân chạm đất".

  When the feet are in contact, reward each foot link's up-axis pointing up
  (``quat_apply(foot_quat, +z)·world_z`` ≈ 1 = sole flat on the ground; an edge/toe
  landing tilts it away from 1). Only evaluated WHILE THE FEET TOUCH (peak feet
  force > ``contact_threshold``) so it shapes the touchdown and the stand without
  penalising the airborne roll arc where the feet are off the ground (returns 1.0
  then). GROUNDED-gated."""
  robot = env.scene["robot"]
  foot_idx = robot.find_bodies(
    ["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
  )[0]
  foot_quat = robot.data.body_link_quat_w[:, foot_idx]  # [B, 2, 4]
  b, nf = foot_quat.shape[0], foot_quat.shape[1]
  e_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(b * nf, 3)
  up = quat_apply(foot_quat.reshape(-1, 4), e_z).reshape(b, nf, 3)
  flat = up[..., 2].clamp(0.0, 1.0).mean(dim=-1)  # [B] 1 = both soles level
  shaped = flat.pow(scale) if scale != 1.0 else flat
  feet_sensor = env.scene.sensors[feet_sensor_name]
  feet_force = torch.norm(feet_sensor.data.force, dim=-1).max(dim=-1)[0]
  in_contact = feet_force > contact_threshold
  shaped = torch.where(in_contact, shaped, torch.ones_like(shaped))
  return _gate_grounded(env, shaped)


def hands_forward(
  env: ManagerBasedRlEnv,
  target_reach: float = 0.20,
  head_roll_threshold: float = 1.2,
) -> torch.Tensor:
  """Reward both WRISTS reaching FORWARD to brace the roll — "2 điểm cuối của tay
  cùng hướng về phía trước để đỡ cho motion".

  Projects each wrist's position (relative to the pelvis) onto the body forward
  axis and rewards the mean forward extension:
  ``clamp(mean(forward_reach) / target_reach, 0, 1)`` → 1 once both hands are
  ``target_reach`` m in front of the pelvis (arms thrown forward to cushion), → 0
  when the arms are back/down. Active while head < ``head_roll_threshold`` (during
  the dive/roll) and GROUNDED."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  quat = robot.data.root_link_quat_w  # [B, 4]
  b = quat.shape[0]
  e_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(b, 3)
  fwd = quat_apply(quat, e_x)  # body forward axis in world [B, 3]
  wrist_idx = robot.find_bodies(
    ["left_wrist_yaw_link", "right_wrist_yaw_link"], preserve_order=True
  )[0]
  pelvis = robot.data.root_link_pos_w  # [B, 3]
  wrists = robot.data.body_link_pos_w[:, wrist_idx]  # [B, 2, 3]
  rel = wrists - pelvis[:, None, :]
  reach = (rel * fwd[:, None, :]).sum(dim=-1)  # [B, 2] forward extension per wrist
  shaped = torch.clamp(reach.mean(dim=-1) / target_reach, 0.0, 1.0)
  shaped = torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
  return _gate_grounded(env, shaped)
