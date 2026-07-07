"""SMP getup task — registers ``Smp-Getup-mini-m1v3`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_mini_smp_ppo_runner_cfg
from smp.rl.tasks.getup_mini_m1v3.getup_env_cfg import mini_v3_getup_smp_env_cfg

_getup_rl = unitree_mini_smp_ppo_runner_cfg()
_getup_rl.experiment_name = "smp_getup_mini_m1v3"
_getup_rl.run_name = "smp_getup_mini_m1v3"

register_mjlab_task(
  task_id="Smp-Getup-mini-m1v3",
  env_cfg=mini_v3_getup_smp_env_cfg(play=False),
  play_env_cfg=mini_v3_getup_smp_env_cfg(play=True),
  rl_cfg=_getup_rl,
)

__all__ = ["mini_v3_getup_smp_env_cfg"]
