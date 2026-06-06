"""Mini_M1v1 getup task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup_mini import mdp
from smp.robot.Mini_M1v1.mini_m11_constants import get_spec as _get_mini_spec

# Mini_M1v1 geometry (from Mini_M1v1.xml):
#   torso_link at pos="0 0 0" relative to pelvis_link (same height).
#   head_link (commented-out) at pos="0 0 0.31" from torso_link.
#   Standing pelvis height ~0.37 m  →  head centre ~0.37 + 0.31 = 0.68 m.
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.31)

# Reward / termination heights scaled to Mini's ~0.68 m standing head height.
HEAD_TARGET_HEIGHT: float = 0.65        # track_head_height goal (just below full stand)
HEAD_UP_THRESHOLD: float = 0.50         # upward_velocity: drive while head below this
HEAD_STOOD_UP: float = 0.62             # stood_up: success threshold


def get_mini_spec_with_head() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Mini_M1v1 spec with a massless ``head`` site on ``torso_link``."""
  spec = _get_mini_spec()
  torso = spec.body("torso_link")
  if not any(s.name == "head" for s in torso.sites):
    torso.add_site(name="head", pos=HEAD_POS_IN_TORSO)
  return spec


def mini_getup_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Mini_M1v1 getup env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  # Replace the stock spec with one that has a ``head`` site for rewards.
  cfg.scene.entities["robot"].spec_fn = get_mini_spec_with_head

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_getup_f2s2.pt"
  )
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # --- Rewards -------------------------------------------------------------
  # task = 0.7·upward_velocity + 0.3·head_height, gated by SMP.
  # Thresholds scaled to Mini (~0.68 m standing head height vs G1 ~1.7 m).
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "task_terms": (
        (
          mdp.upward_velocity,
          0.7,
          {
            "target_velocity": 0.25,
            "head_height_threshold": HEAD_UP_THRESHOLD,
            "scale": 100.0,
          },
        ),
        (mdp.track_head_height, 0.3, {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0}),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  # Getup starts from fallen pose — remove self_collision to avoid false
  # triggers from the robot lying on the ground.
  cfg.terminations.pop("self_collision", None)

  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.02, "ws": 6.0, "grace_steps": 5},
  )

  # Truncate (time_out=True) once stably upright so value bootstraps correctly.
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": HEAD_STOOD_UP, "max_speed": 0.5, "hold_steps": 25},
  )

  cfg.episode_length_s = 5

  return cfg
