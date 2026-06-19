"""Mini_M1v1 jump-platform task with SMP guidance.

The robot starts standing on a static box platform (see
``Mini_M1v1_platform.xml``) and must step/jump off it down to the floor, then
recover — guided by an SMP prior trained on the
``jump_form_box_to_safety_roll_*`` mocap clips (``roll_platform_pretrained.pt``).
The reward/termination shape is shared with ``getup_mini`` (impact → roll →
stand-up phases): jumping off a platform and recovering from a fall is the
same getup problem with a different starting condition.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup_mini import mdp
from smp.robot.Mini_M1v1.mini_m11_constants import KNEES_BENT_KEYFRAME

_PLATFORM_XML = Path("src/smp/robot/Mini_M1v1/Mini_M1v1_platform.xml")

# Mini_M1v1 geometry (from Mini_M1v1.xml):
#   torso_link at pos="0 0 0" relative to pelvis_link (same height).
#   head_link (commented-out) at pos="0 0 0.31" from torso_link.
#   Standing pelvis height ~0.37 m  →  head centre ~0.37 + 0.31 = 0.68 m.
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.31)

# Reward / termination heights scaled to Mini's ~0.68 m standing head height.
HEAD_TARGET_HEIGHT: float = 0.65        # track_head_height goal (just below full stand)
HEAD_UP_THRESHOLD: float = 0.50         # upward_velocity: drive while head below this
HEAD_STOOD_UP: float = 0.62             # stood_up: success threshold
# Ukemi-specific thresholds.
HEAD_FLOOR_THRESHOLD: float = 0.30      # soft_landing: below = impact/contact phase
HEAD_ROLL_THRESHOLD: float = 0.42       # roll_momentum: below = active rolling phase

# Platform geometry (Mini_M1v1_platform.xml): now the symmetric staircase from
# scene_Mini_M1v1_stair_jump.xml — the TOP tread (where the robot spawns) is at
# z=0.60, 1.5 m deep × 1.2 m wide, with 3 descending 0.15 m steps on each side.
PLATFORM_HEIGHT: float = 0.60
PLATFORM_SPAWN_POS: tuple[float, float, float] = (
  0.0,
  0.0,
  PLATFORM_HEIGHT + KNEES_BENT_KEYFRAME.pos[2],
)


def get_mini_platform_spec() -> mujoco.MjSpec:  # type: ignore[attr-defined]
  """Mini_M1v1 spec with the static jump-off platform and a ``head`` site.

  Loads ``Mini_M1v1_platform.xml`` directly (rather than calling
  ``mini_m11_constants.get_spec()``, which only loads the plain
  ``Mini_M1v1.xml``) so the platform geometry is scoped to this task only —
  other Mini_M1v1 tasks (getup_mini, steering_mini, ...) are unaffected.
  """
  platform_xml = _PLATFORM_XML
  spec = mujoco.MjSpec.from_file(str(platform_xml))
  torso = spec.body("torso_link")
  if not any(s.name == "head" for s in torso.sites):
    torso.add_site(name="head", pos=HEAD_POS_IN_TORSO)
  return spec


def mini_jump_platform_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Mini_M1v1 jump-platform env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  # Replace the stock spec with one that adds the static box platform plus a
  # ``head`` site for rewards.
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.spec_fn = get_mini_platform_spec
  # Spawn standing on top of the platform instead of on the bare floor.
  cfg.scene.entities["robot"] = dataclasses.replace(
    robot_cfg,
    init_state=dataclasses.replace(KNEES_BENT_KEYFRAME, pos=PLATFORM_SPAWN_POS),
  )

  # The collidable staircase generates many more simultaneous contacts than the
  # old single box (the robot can touch several steps + the platform at once,
  # especially mid-roll). The shared cfg's nconmax=35 overflows ("nconmax must
  # be >= 36"); raise the contact buffer to accommodate the stair geometry.
  cfg.sim.nconmax = 256

  # --- Events --------------------------------------------------------------
  # SMP prior trained on the jump_form_box_to_safety_roll_* mocap clips
  # (jumping off a box and recovering with a safety roll), so the sampled
  # GSI poses span standing-on-box → mid-air → impact → roll → stand-up.
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "dataset_mini/chuong_data/roll_platform_pretrained.pt"
  )
  # No z-offset: the task is "jump FROM the 0.60 m platform DOWN to the floor and
  # roll", so the prior's floor (z=0) must map to the real floor (z=0) and its
  # platform (~0.59 m) to the real 0.60 m platform — i.e. the heights already
  # match, no shift. (A 0.60 offset would replay the whole jump+roll ON TOP of the
  # platform, so the robot would never reach the floor.)
  cfg.events["init_smp_state"].params["smp_z_offset"] = 0.0
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # NOTE: the shared push (±0.5 m/s) is kept as-is. GSI already seeds a fraction
  # of episodes directly in fallen/rolling poses (the prior's manifold reaches
  # root_z≈0.12, fully inverted), which is the main roll-to-getup practice — so
  # we do NOT need an aggressive push, and a strong push only risks contact
  # blow-ups. Revisit (modestly) only once training is confirmed stable.

  # --- Rewards -------------------------------------------------------------
  # Jump-off-platform → safety-roll-on-floor → stand (all terms ∈ [0,1], weights
  # sum to 1), SMP-gated (gate = smp_floor + (1−smp_floor)·r_smp). The roll_platform
  # prior carries the jump+roll STYLE; the task terms supply the phase incentives.
  #
  # Behaviour arc (base = pelvis world-z; platform top 0.60 m, floor 0):
  #   1. ON PLATFORM  (base ≈ 1.37): descend_off_platform penalises staying up
  #      → the robot must JUMP OFF the edge toward the floor.
  #   2. FALL         (base 1.37 → 0): the prior + proactive_roll set up the tuck.
  #   3. LAND + ROLL  (head/base low): soft_landing rewards a low-impact landing,
  #      roll_momentum rewards the rolling angular velocity (ukemi).
  #   4. RECOVER      (on floor): track_head_height (→0.65) + upright_progress
  #      bring it to a stable floor stand.
  #
  # Reward ordering by outcome: floor-stand (≈1.0) > stay-on-platform (≈0.87,
  # capped by descend_off_platform) > hard flop — so leaving the platform and
  # rolling to a clean stand is the global optimum.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "smp_floor": 0.0,
      "task_terms": (
        # JUMP-OFF driver: penalise remaining at platform height; 1.0 once the
        # base is at/below floor-standing height.
        (
          mdp.descend_off_platform,
          0.25,
          {"target_height": 0.80, "scale": 4.0},
        ),
        # LANDING: absorb impact with a soft (rolling) landing, not a hard slam.
        (
          mdp.soft_landing,
          0.20,
          {"scale": 4.0, "head_floor_threshold": HEAD_FLOOR_THRESHOLD},
        ),
        # ROLL: reward horizontal angular velocity once low — the safety roll.
        (
          mdp.roll_momentum,
          0.20,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # Initiate the roll mid-fall once the torso tilts past upright.
        (
          mdp.proactive_roll,
          0.10,
          {"tilt_threshold": 0.6, "target_ang_vel": 2.0, "scale": 0.5},
        ),
        # RECOVER: stand tall on the floor after the roll.
        (
          mdp.track_head_height,
          0.15,
          {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0},
        ),
        # RECOVER: torso upright for the final floor stand.
        (
          mdp.upright_progress,
          0.10,
          {"scale": 1.0},
        ),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  # Getup starts from fallen pose — remove self_collision to avoid false
  # triggers from the robot lying on the ground.
  cfg.terminations.pop("self_collision", None)

  # smp_too_low guards against the "violent unphysical getup" exploit, but for a
  # getup task the START pose is necessarily off-manifold, so it must NOT fire
  # during the recovery itself.  With the floored SMP gate the violent shortcut
  # is already much less rewarding, so we relax this hard: a lower threshold + a
  # long grace window (≈1 s at 50 Hz) only terminates on a SUSTAINED, deeply
  # degenerate collapse, leaving the policy time to roll up.
  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.005, "ws": 6.0, "grace_steps": 50},
  )

  # Physics-divergence guard: terminate runaway contact blow-ups (root speed
  # leaving the sane envelope) BEFORE they reach NaN. Independent of the SMP
  # score, so it kills only diverging physics — not stable fallen poses — which
  # lets ``smp_too_low`` stay relaxed (grace 50) for recovery learning. Without
  # this, the relaxed ``smp_too_low`` no longer resets flailing poses early
  # enough and the contact solver runs away to NaN in the actor observation.
  cfg.terminations["diverged"] = TerminationTermCfg(
    func=mdp.diverged,
    params={"max_lin_speed": 25.0, "max_ang_speed": 40.0},
  )

  # Truncate (time_out=True) once stably upright so value bootstraps correctly.
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": HEAD_STOOD_UP, "max_speed": 0.5, "hold_steps": 25},
  )

  cfg.episode_length_s = 5

  return cfg
