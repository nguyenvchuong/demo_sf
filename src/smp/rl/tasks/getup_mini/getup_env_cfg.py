"""Mini_M1v1 getup task with SMP guidance."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup_mini import mdp
from smp.robot.Mini_M1v1.mini_m11_constants import get_spec as _get_mini_spec

# Mini_M1v1 geometry (from Mini_M1v1.xml), MEASURED (not guessed):
#   torso_link at pos="0 0 0" relative to pelvis_link (same height).
#   head_link (commented-out) at pos="0 0 0.31" from torso_link, and its own
#   inertial/collision geom is centred a further "0 0 0.05" inside head_link —
#   so the head's actual centre is 0.31 + 0.05 = 0.36 m above torso (matching
#   how G1's HEAD_POS_IN_TORSO=0.43 is the head_collision geom's own centre,
#   not just its parent body's origin).
#   Standing pelvis height 0.77 m (KNEES_BENT_KEYFRAME) → head centre
#   0.77 + 0.36 = 1.13 m. (The old "~0.68 m" comment/thresholds were carried
#   over from a half-scale robot and never re-derived against the real 1.13 m.)
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.36)

# Reward / termination heights scaled to Mini's MEASURED 1.13 m standing head
# height, keeping the same relative fractions as the original (wrong-base)
# thresholds: 95.6% / 73.5% / 91.2% / 44.1% / 61.8% of standing head height.
HEAD_TARGET_HEIGHT: float = 1.08  # track_head_height goal (just below full stand)
HEAD_UP_THRESHOLD: float = 0.83  # upward_velocity: drive while head below this
HEAD_STOOD_UP: float = 1.03  # stood_up: success threshold
# Ukemi-specific thresholds.
HEAD_FLOOR_THRESHOLD: float = 0.50  # soft_landing: below = impact/contact phase
HEAD_ROLL_THRESHOLD: float = 0.70  # roll_momentum: below = active rolling phase

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

  # Stronger push_robot so the robot is genuinely TOPPLED and must roll to
  # recover, plus a longer interval so it has time to roll up before the next
  # push. Overridden locally so other mini tasks keep the gentle shared push.
  # (Tolerable now that ``diverged`` limits are raised below; play mode strips
  # the push events, hence the guard.)
  if "push_robot" in cfg.events:
    _push = cfg.events["push_robot"]
    _push.interval_range_s = (2.0, 5.0)  # more recovery time between pushes
    _push.params["velocity_range"] = {
      "x": (-1.2, 1.2),
      "y": (-1.2, 1.2),
      "z": (-0.6, 0.6),
      "roll": (-1.0, 1.0),
      "pitch": (-1.0, 1.0),
      "yaw": (-1.2, 1.2),
    }

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
          0.30,
          {
            "tilt_threshold": 0.4,
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

  # --- Smoothness penalties -------------------------------------------------
  # The task terms above all saturate to ~1.0 once standing, leaving a flat
  # reward gradient with nothing penalising high-frequency action/joint
  # chatter — the robot buzzes/vibrates when settled. These separate
  # negative-weight terms (summed by the reward manager alongside the [0,1]
  # task_smp_product) regularise the motion. Kept small so they don't fight the
  # getup/rolling phase; action_rate is the dominant anti-vibration term.
  cfg.rewards["action_rate"] = RewardTermCfg(
    func=base_mdp.action_rate_l2, weight=-0.03
  )
  cfg.rewards["action_acc"] = RewardTermCfg(
    func=base_mdp.action_acc_l2, weight=-0.001
  )
  cfg.rewards["joint_vel"] = RewardTermCfg(
    func=base_mdp.joint_vel_l2, weight=-1e-3
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
    params={"max_lin_speed": 40.0, "max_ang_speed": 60.0},
  )

  # Truncate (time_out=True) once stably upright so value bootstraps correctly.
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": HEAD_STOOD_UP, "max_speed": 0.5, "hold_steps": 25},
  )

  # --- Metrics (live plots in Viser + episode logs during training) ----------
  _force_sensor = GROUND_CONTACT_FORCE_SENSOR.name
  cfg.metrics = {
    "peak_contact_force_N": MetricsTermCfg(
      func=mdp.peak_ground_contact_force,
      params={"sensor_name": _force_sensor},
    ),
    "ground_contacting_bodies": MetricsTermCfg(
      func=mdp.ground_contacting_bodies,
      params={"sensor_name": _force_sensor},
    ),
    "head_height_m": MetricsTermCfg(func=mdp.head_height),
    "pelvis_downward_speed_mps": MetricsTermCfg(func=mdp.pelvis_downward_speed),
    **mdp.make_per_link_peak_force_metrics(_force_sensor),
  }

  cfg.episode_length_s = 5

  if play:
    # kéo/giật tay trong viewer tạo velocity spike giả — đừng reset
    cfg.terminations.pop("diverged", None)
    cfg.terminations.pop("smp_too_low", None)

  return cfg
