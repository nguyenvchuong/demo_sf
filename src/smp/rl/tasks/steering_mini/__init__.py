"""SMP steering tasks — registers ``Smp-Steering-mini`` and ``Smp-Forward-mini``
on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_mini_smp_ppo_runner_cfg
from smp.rl.tasks.steering_mini.forward_env_cfg import mini_forward_smp_env_cfg
from smp.rl.tasks.steering_mini.steering_env_cfg import mini_steering_smp_env_cfg

_steering_rl = unitree_mini_smp_ppo_runner_cfg()
_steering_rl.experiment_name = "smp_steering_mini"
_steering_rl.run_name = "smp_steering_mini"

register_mjlab_task(
  task_id="Smp-Steering-mini",
  env_cfg=mini_steering_smp_env_cfg(play=False),
  play_env_cfg=mini_steering_smp_env_cfg(play=True),
  rl_cfg=_steering_rl,
)

_forward_rl = unitree_mini_smp_ppo_runner_cfg()
_forward_rl.experiment_name = "smp_forward_mini"
_forward_rl.run_name = "smp_forward_mini"

register_mjlab_task(
  task_id="Smp-Forward-mini",
  env_cfg=mini_forward_smp_env_cfg(play=False),
  play_env_cfg=mini_forward_smp_env_cfg(play=True),
  rl_cfg=_forward_rl,
)

__all__ = [
  "mini_forward_smp_env_cfg",
  "mini_steering_smp_env_cfg",
]
