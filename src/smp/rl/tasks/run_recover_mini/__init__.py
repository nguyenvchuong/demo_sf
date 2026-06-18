"""SMP run-recover task — registers ``Smp-RunRecover-mini`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.rl_cfg import unitree_mini_smp_ppo_runner_cfg
from smp.rl.tasks.run_recover_mini.run_recover_env_cfg import mini_run_recover_smp_env_cfg

_run_recover_rl = unitree_mini_smp_ppo_runner_cfg()
_run_recover_rl.experiment_name = "smp_run_recover_mini"
_run_recover_rl.run_name = "smp_run_recover_mini"

register_mjlab_task(
  task_id="Smp-RunRecover-mini",
  env_cfg=mini_run_recover_smp_env_cfg(play=False),
  play_env_cfg=mini_run_recover_smp_env_cfg(play=True),
  rl_cfg=_run_recover_rl,
)

__all__ = ["mini_run_recover_smp_env_cfg"]
