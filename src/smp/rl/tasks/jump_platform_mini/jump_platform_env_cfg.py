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

# Platform geometry (Mini_M1v1_platform.xml): box top at z=0.25, sized from the
# jump_form_box_to_safety_roll_* mocap clips (mean starting pelvis height
# 1.021 m vs. the robot's normal 0.77 m standing pelvis height).
PLATFORM_HEIGHT: float = 0.25
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

  # --- Events --------------------------------------------------------------
  # SMP prior trained on the jump_form_box_to_safety_roll_* mocap clips
  # (jumping off a box and recovering with a safety roll), so the sampled
  # GSI poses span standing-on-box → mid-air → impact → roll → stand-up.
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "dataset_mini/chuong_data/roll_platform_pretrained.pt"
  )
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # NOTE: the shared push (±0.5 m/s) is kept as-is. GSI already seeds a fraction
  # of episodes directly in fallen/rolling poses (the prior's manifold reaches
  # root_z≈0.12, fully inverted), which is the main roll-to-getup practice — so
  # we do NOT need an aggressive push, and a strong push only risks contact
  # blow-ups. Revisit (modestly) only once training is confirmed stable.

  # --- Rewards -------------------------------------------------------------
  # Ukemi + quick-standup reward (all terms ∈ [0,1], weights sum to 1).
  #
  # The reward is SMP-gated with a FLOOR (``smp_floor``): the gate is
  #   gate = smp_floor + (1 - smp_floor) · r_smp ∈ [smp_floor, 1].
  # A pure ×r_smp gate collapses to ~0 in fallen/off-manifold poses (the single
  # rolling clip's manifold is a thin tube dominated by standing), which leaves
  # the policy with NO gradient out of the exact states a getup task must escape.
  # The floor keeps ``smp_floor · task`` flowing off-manifold while on-manifold
  # rolling still earns the full ×1 style bonus.
  #
  # Term roles:
  #   upright_progress  — ALWAYS-ON monotonic potential (torso orientation).
  #                       This is the spine of the lying→standing gradient and
  #                       is never gated to a single phase, so the policy always
  #                       has a signal pulling it upright off the floor.
  #   track_head_height — ALWAYS-ON: stand tall.
  #   upward_velocity   — phase (head < HEAD_UP_THRESHOLD): rise QUICKLY.
  #   roll_momentum     — phase (head < HEAD_ROLL_THRESHOLD): roll to spread impact.
  #   soft_landing      — phase (head < HEAD_FLOOR_THRESHOLD): absorb the fall.
  #
  # Phase terms return 1.0 outside their window → no cross-phase interference;
  # the two always-on terms carry the global getup gradient.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "smp_floor": 0.3,
      "task_terms": (
        # Always-on: rotate the torso upright from ANY pose (core getup signal).
        (
          mdp.upright_progress,
          0.25,
          {"scale": 1.0},
        ),
        # Always-on: drive the head toward standing height.
        (
          mdp.track_head_height,
          0.20,
          {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0},
        ),
        # Rising phase: drive head upward quickly until near-standing height.
        (
          mdp.upward_velocity,
          0.15,
          {
            "target_velocity": 0.40,
            "head_height_threshold": HEAD_UP_THRESHOLD,
            "scale": 100.0,
          },
        ),
        # PROACTIVE roll: once the torso tilts past recovery (CoM outside the
        # support polygon), reward rolling angular momentum — triggered by TILT
        # while still up high, so the robot commits to a roll instead of a rigid
        # flat fall. This is the term that makes it roll when pushed over.
        (
          mdp.proactive_roll,
          0.20,
          {
            "tilt_threshold": 0.6,
            "target_ang_vel": 2.0,
            "scale": 0.5,
          },
        ),
        # On-ground rolling: keep rotating once already low (head < threshold).
        (
          mdp.roll_momentum,
          0.10,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # Impact phase: reward soft (rolling) landing — penalise hard fall speed.
        (
          mdp.soft_landing,
          0.10,
          {
            "scale": 4.0,
            "head_floor_threshold": HEAD_FLOOR_THRESHOLD,
          },
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
