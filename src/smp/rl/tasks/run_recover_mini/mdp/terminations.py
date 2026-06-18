"""Termination terms for the run-recover task.

No ``stood_up`` truncation — success is NOT standing still, it's recovering and
continuing to run, so the episode always runs to ``time_out``.
No ``base_too_low`` — the robot legitimately lies on the ground during recovery.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["diverged", "smp_too_low"]


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
