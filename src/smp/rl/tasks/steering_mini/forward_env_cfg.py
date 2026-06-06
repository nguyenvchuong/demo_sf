"""mini forward task — fixed +x heading, variable target speed.

A specialization of the steering task with the world-frame target and face
directions pinned to ``+x``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.steering import mdp


def mini_forward_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the mini forward env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=0.5,
    tar_speed_max=5.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  # task = velocity tracking, gated by SMP.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.steering_target_velocity,
          1.0,
          {"command_name": "steering", "vel_err_scale": 0.5},
        ),
      ),
    },
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/pretrain/20260606_121844/checkpoint_00300.pt"
  )

  # --- Terminations --------------------------------------------------------
  # Mini_M1v1: pelvis_link starts at z=0.8m; standing pelvis height ~0.37m.
  # 0.3m is too close to standing height — tightens false terminations.
  # Use 0.15m so only a real collapse (knees on ground) triggers this.
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.15,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
