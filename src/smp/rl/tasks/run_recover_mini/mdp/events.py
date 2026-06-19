"""Reset events for the run-recover task."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

__all__ = ["reset_stall_counter"]


@torch.no_grad()
def reset_stall_counter(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None
) -> None:
  """Zero the stall counter for the reset envs (no-op until ``stalled_while_upright``
  lazily creates it).  Prevents stale counts from a previous episode carrying over."""
  if not hasattr(env, "_stall_count"):
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env._stall_count[env_ids] = 0  # type: ignore[attr-defined]
