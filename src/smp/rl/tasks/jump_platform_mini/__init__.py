"""SMP jump-platform task — registers ``Smp-Jump-Platform-mini`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_mini_smp_ppo_runner_cfg
from smp.rl.tasks.jump_platform_mini.jump_platform_env_cfg import (
  mini_jump_platform_smp_env_cfg,
)

_jump_platform_rl = unitree_mini_smp_ppo_runner_cfg()
_jump_platform_rl.experiment_name = "smp_jump_platform_mini"
_jump_platform_rl.run_name = "smp_jump_platform_mini"

register_mjlab_task(
  task_id="Smp-Jump-Platform-mini",
  env_cfg=mini_jump_platform_smp_env_cfg(play=False),
  play_env_cfg=mini_jump_platform_smp_env_cfg(play=True),
  rl_cfg=_jump_platform_rl,
)

__all__ = ["mini_jump_platform_smp_env_cfg"]
