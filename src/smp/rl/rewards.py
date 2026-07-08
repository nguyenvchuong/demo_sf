"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer

if TYPE_CHECKING:
  from collections.abc import Callable

  from mjlab.envs import ManagerBasedRlEnv

  TaskTerm = tuple["Callable[..., torch.Tensor]", float, dict]


def _update_buffer_from_sim(env: ManagerBasedRlEnv) -> None:
  """Push current sim kinematics onto the buffer tail, env-origin-relative
  (matching ``_prime_sim_and_buffer``) so features are placement-invariant."""
  robot = env.scene["robot"]
  ee_indexes = env._smp_ee_indexes  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  origins = env.scene.env_origins
  root_pos = robot.data.root_link_pos_w - origins
  ee_pos = robot.data.body_link_pos_w[:, ee_indexes] - origins[:, None, :]
  # Undo the GSI sim z-offset so the prior sees motion relative to a floor at 0
  # (the robot physically stands on a raised surface). Subtracting from BOTH
  # root and EE leaves the ee−root feature unchanged and only lowers root_pos_z.
  z_off = getattr(env, "_smp_z_offset", 0.0)
  if z_off:
    root_pos = root_pos.clone()
    root_pos[:, 2] -= z_off
    ee_pos = ee_pos.clone()
    ee_pos[..., 2] -= z_off
  buffer.update(
    root_pos,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    ee_pos,
    robot.data.joint_pos,
    robot.data.joint_vel,
  )


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
  normalize: bool = True,
) -> torch.Tensor:
  """SDS-style guidance reward over fixed timesteps ``K``:
  ``exp(-w_s/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)``.  ``normalize`` divides each MSE by a
  ``DiffNormalizer`` running mean (policy-relative) vs. raw (absolute scale);
  always stashes the mean raw MSE on ``env._smp_raw_err``."""
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]
  buffer: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  _update_buffer_from_sim(env)

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  total_raw = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_raw += mse_per_env
      if normalize:
        total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)
      else:
        total_err += mse_per_env

  env._smp_raw_err = total_raw / len(fixed_timesteps)  # type: ignore[attr-defined]
  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)


def task_smp_product(
  env: ManagerBasedRlEnv,
  task_terms: tuple[TaskTerm, ...],
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 6.0,
  smp_floor: float = 0.0,
) -> torch.Tensor:
  """``(Σ wᵢ · taskᵢ(env)) · gate`` where ``gate = smp_floor + (1−smp_floor)·r_smp``.

  ``task_terms`` is a tuple of ``(func, weight, kwargs)``.  Calls
  ``smp_guidance_reward`` once (the sole SMP-buffer update), so it must be the
  task's only SMP reward term.

  ``smp_floor`` (∈ [0, 1]) is the multiplicative floor on the SMP gate:
    * ``0.0`` → pure multiplicative gating (off-manifold states earn nothing).
    * ``>0``  → off-manifold states still earn ``smp_floor · task``, so there is
      always a task-reward gradient (essential for getup/recovery, where the
      starting pose is *necessarily* off the prior's manifold), while on-manifold
      motion still earns the full ``×1`` style bonus.
  """
  task = sum(w * func(env, **kw) for func, w, kw in task_terms)
  r_smp = smp_guidance_reward(env, fixed_timesteps=fixed_timesteps, ws=ws)
  gate = smp_floor + (1.0 - smp_floor) * r_smp
  return task * gate
