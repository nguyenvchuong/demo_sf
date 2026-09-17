"""SMP air-drop task — registers ``Smp-Air-Drop-mini`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_mini_smp_ppo_runner_cfg
from smp.rl.tasks.air_drop_mini.air_drop_env_cfg import mini_air_drop_smp_env_cfg

_air_drop_rl = unitree_mini_smp_ppo_runner_cfg()
_air_drop_rl.experiment_name = "smp_air_drop_mini"
_air_drop_rl.run_name = "smp_air_drop_mini"

register_mjlab_task(
  task_id="Smp-Air-Drop-mini",
  env_cfg=mini_air_drop_smp_env_cfg(play=False),
  play_env_cfg=mini_air_drop_smp_env_cfg(play=True),
  rl_cfg=_air_drop_rl,
)

__all__ = ["mini_air_drop_smp_env_cfg"]
