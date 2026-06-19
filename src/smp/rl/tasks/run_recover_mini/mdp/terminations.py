"""Termination terms for the run-recover task.

No ``base_too_low`` — the robot legitimately lies on the ground during recovery.
``stalled_while_upright`` enforces continuous running: terminate if upright but
not moving forward for too long.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["diverged", "smp_too_low", "stalled_while_upright"]


def diverged(
  env: ManagerBasedRlEnv,
  max_lin_speed: float = 25.0,
  max_ang_speed: float = 40.0,
) -> torch.Tensor:
  """Terminate a physically diverging env before it produces NaN observations.

  Contact-solver blow-ups ramp root speed from ~10 → inf over a few steps.
  Terminating when speed leaves the sane envelope resets the env before NaN
  reaches the actor observation. Independent of SMP score — stable fallen poses
  are NOT killed, only runaway physics.
  """
  robot = env.scene["robot"]
  lin = robot.data.root_link_lin_vel_w
  ang = robot.data.root_link_ang_vel_w
  lin_speed = torch.linalg.norm(lin, dim=-1)
  ang_speed = torch.linalg.norm(ang, dim=-1)
  bad = (lin_speed > max_lin_speed) | (ang_speed > max_ang_speed)
  nonfinite = ~(torch.isfinite(lin).all(dim=-1) & torch.isfinite(ang).all(dim=-1))
  return bad | nonfinite


def smp_too_low(
  env: ManagerBasedRlEnv,
  threshold: float = 0.005,
  ws: float = 6.0,
  grace_steps: int = 50,
) -> torch.Tensor:
  """Terminate when SMP score collapses and stays collapsed (off-manifold exploit).

  Uses the raw MSE stored by ``smp_guidance_reward`` so ``ws`` must match the
  reward's. Grace window prevents termination during the recovery phase itself
  (robot is legitimately off-manifold while rolling up after a push).
  """
  raw_err = getattr(env, "_smp_raw_err", None)
  if raw_err is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  raw_smp = torch.exp(-ws * raw_err)
  past_grace = env.episode_length_buf >= grace_steps
  return (raw_smp < threshold) & past_grace


def stalled_while_upright(
  env: ManagerBasedRlEnv,
  command_name: str = "steering",
  min_forward_speed: float = 0.3,
  head_threshold: float = 0.62,
  hold_steps: int = 30,
  grace_steps: int = 50,
) -> torch.Tensor:
  """Terminate if the robot is upright but not moving forward for ``hold_steps``
  consecutive steps.

  Enforces the "must keep running" contract: a robot that stands still after
  recovery (or refuses to get up and run) is terminated so the episode resets.

  Only active when upright (head ≥ ``head_threshold``) — a fallen robot is NOT
  stalled; it is recovering.  The stall counter resets to 0 whenever the robot
  falls OR resumes forward speed, so each recovery cycle gets a fresh allowance.

  ``grace_steps`` delays the check from episode start so the robot has time to
  accelerate from the initial standing pose before the guard fires.
  """
  robot = env.scene["robot"]
  head_idx = robot.find_sites(["head"], preserve_order=True)[0][0]
  head_z = robot.data.site_pos_w[:, head_idx, 2]

  cmd = env.command_manager.get_term(command_name)
  root_vel_xy = robot.data.root_link_lin_vel_w[:, :2]
  proj_speed = (cmd.tar_dir_w * root_vel_xy).sum(dim=-1)

  # Stall = upright AND not moving forward fast enough.
  stalling = (head_z >= head_threshold) & (proj_speed < min_forward_speed)

  cnt = getattr(env, "_stall_count", None)
  if cnt is None:
    cnt = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  cnt = torch.where(stalling, cnt + 1, torch.zeros_like(cnt))
  env._stall_count = cnt  # type: ignore[attr-defined]

  past_grace = env.episode_length_buf >= grace_steps
  return (cnt >= hold_steps) & past_grace
