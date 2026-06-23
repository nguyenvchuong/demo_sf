"""Mini_M1v1 getup task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

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
HEAD_TARGET_HEIGHT: float = 0.65  # track_head_height goal (just below full stand)
HEAD_UP_THRESHOLD: float = 0.50  # upward_velocity: drive while head below this
HEAD_STOOD_UP: float = 0.62  # stood_up: success threshold
# Ukemi-specific thresholds.
HEAD_FLOOR_THRESHOLD: float = 0.30  # soft_landing: below = impact/contact phase
HEAD_ROLL_THRESHOLD: float = 0.42  # roll_momentum: below = active rolling phase

GROUND_CONTACT_FORCE_SENSOR = ContactSensorCfg(
  name="ground_contact_force",
  primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="maxforce",
)


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
  cfg.scene.sensors = (*cfg.scene.sensors, GROUND_CONTACT_FORCE_SENSOR)

  # --- Events --------------------------------------------------------------
  # pretrained_getup_f2s2.pt is feature_dim=59 (G1, 29 DOF) — incompatible with
  # Mini (feature_dim=53, 23 DOF). Use the mini checkpoint until a dedicated
  # Mini getup pretrain is available. Replace this path once you have trained:
  #   uv run scripts/pretrain.py --data-dir dataset_mini/npz_getup ...
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "dataset_mini/cmu/new/pretrained.pt"
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
      "smp_floor": 0.0,
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
            "tilt_threshold": 0.15,
            "target_ang_vel": 2.0,
            "scale": 0.5,
          },
        ),
        # On-ground rolling: keep rotating once already low (head < threshold).
        (
          mdp.roll_momentum,
          0.05,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # Impact phase: reward soft (rolling) landing — penalise hard fall speed.
        (
          mdp.soft_landing,
          0.05,
          {
            "scale": 4.0,
            "head_floor_threshold": HEAD_FLOOR_THRESHOLD,
          },
        ),
        # Rolling phase: penalise high peak ground-contact force, so rolling
        # spreads impact across the body/time instead of slamming one link.
        (
          mdp.rolling_contact_force,
          0.10,
          {
            "sensor_name": GROUND_CONTACT_FORCE_SENSOR.name,
            "max_force": 150.0,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
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
