"""Reward components for the ukemi + getup task.

Phase structure (Mini ~1.13 m standing head height):
  Phase 1 — impact    (head < ~0.50 m): ``soft_landing``   rewards low fall speed
  Phase 1–2 — rolling (head < ~0.70 m): ``roll_momentum``  rewards active rolling
  Phase 2–3 — standup (head rising):    ``upward_velocity`` + ``track_head_height``

All phase terms return 1.0 outside their active window so they do not compete.
Combined and SMP-gated via ``smp.rl.rewards.task_smp_product``.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "upright_progress",
  "track_head_height",
  "upward_velocity",
  "soft_landing",
  "roll_momentum",
  "proactive_roll",
  "descend_off_platform",
  "rolling_contact_force",
]


def descend_off_platform(
  env: ManagerBasedRlEnv,
  target_height: float = 0.80,
  scale: float = 4.0,
) -> torch.Tensor:
  """Reward leaving a raised platform and getting the base down to FLOOR level.

  ``base_z`` (pelvis world z, env-origin relative) ABOVE ``target_height`` (the
  ~0.80 m floor-standing pelvis height) is penalised — so standing on a 0.60 m
  platform (base ≈ 1.37 m) is NO LONGER free reward, which is what drives the
  jump-off. At or below floor-standing it returns 1.0, so the descent, the low
  roll, and the final floor stand are all un-penalised:
  ``exp(-scale·max(base_z − target_height, 0)²)``.

  The JUMP-OFF driver for the jump-platform task. Without it the always-on
  ``track_head_height`` / ``upright_progress`` terms saturate to 1.0 on the
  platform (head far above the 0.65 m target = overshoot), giving the robot no
  reason to leave it. (Inert for the plain getup task, where base never exceeds
  ~0.80 m, so it returns 1.0 throughout there.)
  """
  robot = env.scene["robot"]
  base_z = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  over = torch.clamp(base_z - target_height, min=0.0)
  return torch.exp(-scale * over * over)


def upright_progress(
  env: ManagerBasedRlEnv,
  scale: float = 1.0,
) -> torch.Tensor:
  """Ungated monotonic getup potential from torso orientation.

  ``projected_gravity_b[:, 2]`` is the body-frame z of the gravity vector:
  ``-1`` when perfectly upright, ``0`` when horizontal (lying on side / falling
  over), ``+1`` when fully inverted.  Maps it to ``[0, 1]`` via
  ``((1 − g_z)/2)^scale`` so the reward rises smoothly and monotonically as the
  torso rotates from inverted → flat → upright **from any pose**.

  This is the term that carries the lying→standing gradient.  Unlike the
  phase-gated / SMP-gated terms it is ALWAYS active and never multiplied to
  zero off-manifold, so the policy always has a signal pulling it upright while
  it is on the ground.  ``scale`` > 1 sharpens the reward near upright.
  """
  robot = env.scene["robot"]
  g_z = robot.data.projected_gravity_b[:, 2]
  uprightness = torch.clamp((1.0 - g_z) * 0.5, 0.0, 1.0)
  return uprightness.pow(scale)


def track_head_height(
  env: ManagerBasedRlEnv,
  target_height: float = 1.2,
  scale: float = 6.0,
) -> torch.Tensor:
  """Reward the ``head`` site reaching ``target_height``:
  ``exp(-scale·max(target_height − head_z, 0)²)`` (no penalty for overshoot).
  Needs the ``head`` site from ``getup_env_cfg.get_mini_spec_with_head``."""
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  z = robot.data.site_pos_w[:, head_idx, 2]
  shortfall = torch.clamp(z - target_height, max=0.0)
  return torch.exp(-scale * shortfall * shortfall)


def upward_velocity(
  env: ManagerBasedRlEnv,
  target_velocity: float = 0.40,
  head_height_threshold: float = 0.50,
  scale: float = 100.0,
) -> torch.Tensor:
  """Reward upward HEAD velocity while below ``head_height_threshold`` (else 1).
  ``exp(-scale·max(target_velocity − head_vz, 0)²)``.  Uses the head site's world
  velocity so it drives the whole-body rising motion, not just the pelvis."""
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
  env: ManagerBasedRlEnv,
  scale: float = 4.0,
  head_floor_threshold: float = 0.30,
) -> torch.Tensor:
  """Reward low downward pelvis speed during the impact phase (head < threshold).

  Encourages the robot to absorb the fall by rolling rather than resisting
  rigidly.  ``exp(-scale·max(-v_z, 0)²)`` gives 1.0 when stationary or rising
  and decays sharply for high falling speed.  Returns 1.0 above threshold so
  it does not interfere with the standup phase.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  base_vz = robot.data.root_link_lin_vel_w[:, 2]
  # positive = falling down
  falling = torch.clamp(-base_vz, min=0.0)
  shaped = torch.exp(-scale * falling * falling)
  return torch.where(head_z < head_floor_threshold, shaped, torch.ones_like(shaped))


def proactive_roll(
  env: ManagerBasedRlEnv,
  tilt_threshold: float = 0.6,
  target_ang_vel: float = 2.0,
  scale: float = 0.5,
) -> torch.Tensor:
  """Reward converting a COMMITTED topple into a roll — triggered by TILT, not height.

  Tilt = ``‖projected_gravity_b[:, :2]‖`` = sin(lean angle): 0 when upright, 1
  when the torso is horizontal.  Once tilt exceeds ``tilt_threshold`` the CoM has
  left the foot support polygon and the fall can no longer be arrested by
  balancing — resisting it rigidly produces a hard flat impact.  At that point
  this term rewards horizontal angular-velocity magnitude ``|ω_xy|`` reaching
  ``target_ang_vel``, i.e. *going with* the rotation and tucking into a roll
  rather than fighting it.

  Unlike ``roll_momentum`` (gated on head height, so it only fires once already
  low) this is gated on tilt, so it fires while the robot is still up high — the
  PROACTIVE trigger.  A rigid robot that resists the fall has low ``|ω_xy|`` and
  is penalised; a robot that rolls has high ``|ω_xy|`` and scores ~1.  Returns
  1.0 below the threshold so balanced standing is never disturbed.  The tilt
  gate means the term cannot be farmed by spinning while still upright.
  """
  robot = env.scene["robot"]
  tilt = torch.norm(robot.data.projected_gravity_b[:, :2], dim=-1)
  ang_xy = robot.data.root_link_ang_vel_w[:, :2]
  ang_mag = torch.norm(ang_xy, dim=-1)
  shortfall = torch.clamp(ang_mag - target_ang_vel, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(tilt > tilt_threshold, shaped, torch.ones_like(shaped))


def roll_momentum(
  env: ManagerBasedRlEnv,
  target_ang_vel: float = 1.5,
  scale: float = 1.0,
  head_roll_threshold: float = 0.42,
) -> torch.Tensor:
  """Reward active rolling (sagittal + lateral angular velocity) while on the ground.

  Rolling transfers fall energy through time rather than absorbing it as a
  sudden impact spike.  Rewards the horizontal angular velocity magnitude
  (pitch + roll components, not yaw) reaching ``target_ang_vel`` rad/s.
  ``exp(-scale·max(target_ang_vel − |ω_xy|, 0)²)`` → 1.0 when rolling well.
  Returns 1.0 above threshold so it does not penalise the upright robot.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]
  # world-frame x (roll) and y (pitch) — exclude yaw which is not ukemi motion
  ang_xy = robot.data.root_link_ang_vel_w[:, :2]
  ang_mag = torch.norm(ang_xy, dim=-1)
  shortfall = torch.clamp(ang_mag - target_ang_vel, max=0.0)
  shaped = torch.exp(-scale * shortfall * shortfall)
  return torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))


def rolling_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_force: float = 150.0,
  scale: float = 1.0,
  head_roll_threshold: float = 0.42,
) -> torch.Tensor:
  """Reward LOW peak ground-contact force while actively rolling (head < threshold).

  A good roll spreads impact across the body and over time rather than
  slamming a single link into the floor — this penalises exactly that spike.
  Takes the per-body contact-force magnitude from ``sensor_name`` (a
  robot-vs-terrain ``ContactSensorCfg``), reduces to the single worst contact
  this step, and rewards staying under ``max_force`` (newtons):
  ``exp(-scale·max((peak_force − max_force) / max_force, 0)²)``.  Returns 1.0
  above the head-height threshold so it does not interfere with the standup
  phase (and is inert outside the rolling window, e.g. for ``soft_landing``'s
  impact phase, which gates separately on fall *speed* rather than force).
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]

  contact_sensor = env.scene.sensors[sensor_name]
  force_norm = torch.norm(contact_sensor.data.force, dim=-1)  # [B, N]
  peak_force = torch.max(force_norm, dim=-1)[0]  # [B]

  excess = torch.clamp((peak_force - max_force) / max_force, min=0.0)
  shaped = torch.exp(-scale * excess * excess)
  return torch.where(head_z < head_roll_threshold, shaped, torch.ones_like(shaped))
