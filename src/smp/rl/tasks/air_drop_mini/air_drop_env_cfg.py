"""Mini_M1v1 air-drop task with SMP guidance.

The robot is respawned FLOATING in the air at a RANDOM height holding a RANDOM
pose (sampled from the SMP/GSI manifold, same source as ``getup_mini``), at rest.
Under gravity it free-falls:

  1. AIRBORNE — while it has not touched the ground, ``hold_initial_pose`` rewards
     keeping the joints at the dropped configuration (brace / hold the pose).
  2. TOUCHDOWN — a contact-force latch (``update_landed_latch`` →
     ``env._drop_landed``) detects ground contact and flips the reward phase.
  3. GROUNDED — the SMP-guided ukemi terms (``soft_landing`` + ``roll_momentum`` +
     ``rolling_contact_force``) roll the robot to spread/minimise impact, then
     ``track_head_height`` + ``upright_progress`` recover it to a smooth stand.

It shares the getup spec (flat plane + ``head`` site), the ukemi reward bodies, and
the getup terminations; only the airborne-drop reset, the landed latch, and the
hold-pose reward are new. Heights use Mini's MEASURED 1.08 m standing head (see
``jump_platform_env_cfg``), not the stale 0.68 m.
"""

from __future__ import annotations

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.events import reset_drop_in_air, update_landed_latch
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.air_drop_mini import mdp
from smp.rl.tasks.getup_mini.getup_env_cfg import get_mini_spec_with_head

# Reward / termination heights scaled to Mini's MEASURED 1.08 m standing head
# (standing pelvis 0.77 + head site +0.31), matching the jump-platform task.
HEAD_TARGET_HEIGHT: float = 0.95  # track_head_height goal (recovered floor stand)
HEAD_STOOD_UP: float = 0.88  # stood_up success threshold (clearly upright)
HEAD_FLOOR_THRESHOLD: float = 0.45  # soft_landing / rolling_contact_force: impact phase
HEAD_ROLL_THRESHOLD: float = 0.62  # (unused for roll_momentum here; kept for reference)
HEAD_UP_THRESHOLD: float = 0.75    # upward_velocity: drive head up while below this

# PD-gain softening, AIR-DROP ONLY (does NOT touch the shared robot constants, so
# locomotion tasks keep their stiff gains). A stiff body cannot tuck/curl into a
# roll — it catches the fall rigidly. These scale the actuator stiffness (kp) and
# damping (kd) down so the joints comply during the roll ("không cứng"). Kept high
# enough (~half) that the robot can still hold a stand after the roll; lower further
# for a softer roll, raise toward 1.0 if it can no longer stand.
AIR_DROP_KP_SCALE: float = 0.5
AIR_DROP_KD_SCALE: float = 0.65


def _soften_pd_gains(robot_cfg, kp_scale: float, kd_scale: float):
  """Return a copy of ``robot_cfg`` with every actuator's stiffness/damping scaled.
  Uses ``dataclasses.replace`` so the shared module-level actuator/articulation
  constants are left untouched (other tasks are unaffected)."""
  art = robot_cfg.articulation
  softened = tuple(
    dataclasses.replace(
      a, stiffness=a.stiffness * kp_scale, damping=a.damping * kd_scale
    )
    for a in art.actuators
  )
  return dataclasses.replace(
    robot_cfg, articulation=dataclasses.replace(art, actuators=softened)
  )
# roll_momentum covers the FULL 360° pitch rotation: head goes 0.95 → 0.1 → 0.95 m.
# Setting threshold above standing height keeps the reward active on the way BACK UP
# (second half of the roll), so the robot has incentive to complete the full rotation.
# SMP gate (smp_floor_grounded=0.30) prevents rewarding off-manifold spinning-in-place.
HEAD_FULL_ROLL_THRESHOLD: float = 1.2

# Air-drop spawn: pelvis world-z (above the floor) sampled per episode. Min 1.0 m
# keeps even inverted poses clear of the ground at spawn. With added downward
# z-velocity the robot reaches the ground faster than a pure free-fall, so the
# height range does not need to be extended — the impact energy comes from velocity,
# not height alone.
DROP_PELVIS_HEIGHT_RANGE: tuple[float, float] = (1.2, 1.8)

# Robot-vs-terrain peak contact-force sensor: drives BOTH the touchdown latch
# (update_landed_latch) and the rolling_contact_force penalty. Covers ALL bodies.
GROUND_CONTACT_FORCE_SENSOR = ContactSensorCfg(
  name="ground_contact_force",
  primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="maxforce",
)
# FEET-only contact sensor (ankle_roll bodies = the foot links). Used by
# ``feet_first_contact`` to reward the feet carrying the touchdown load.
FEET_CONTACT_FORCE_SENSOR = ContactSensorCfg(
  name="feet_contact_force",
  primary=ContactMatch(
    mode="body", pattern=r"(left|right)_ankle_roll_link", entity="robot"
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="maxforce",
)
# UPPER-body contact sensor (torso + pelvis + shoulders + arms = the "back").
# Used by ``avoid_upper_impact`` / ``feet_first_contact`` to penalise catching the
# fall on the back instead of the feet. Legs (hip/knee) are intentionally excluded
# — they may legitimately roll-contact during a proper ukemi.
UPPER_CONTACT_FORCE_SENSOR = ContactSensorCfg(
  name="upper_contact_force",
  primary=ContactMatch(
    mode="body",
    pattern=r"(torso_link|pelvis_link|(left|right)_(shoulder_(pitch|roll|yaw)|elbow|wrist_yaw)_link)",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="maxforce",
)
LANDED_FORCE_THRESHOLD: float = 5.0  # N; peak ground force that counts as touchdown


def mini_air_drop_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Mini_M1v1 air-drop env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Sim -----------------------------------------------------------------
  # Full-body roll contacts: 14 foot geoms + torso/pelvis/arms/legs all hitting
  # simultaneously easily exceeds the base nconmax=35 → dropped contacts →
  # robot sinks through the ground. 100 gives comfortable headroom.
  cfg.sim.nconmax = 100

  # --- Scene ---------------------------------------------------------------
  # Flat plane + a ``head`` site for the rewards (reuse the getup spec).
  cfg.scene.entities["robot"].spec_fn = get_mini_spec_with_head
  # Soften the PD gains (air-drop only) so the body can comply/tuck into the roll
  # instead of catching the fall rigidly ("không cứng"). Other tasks are unaffected.
  cfg.scene.entities["robot"] = _soften_pd_gains(
    cfg.scene.entities["robot"], AIR_DROP_KP_SCALE, AIR_DROP_KD_SCALE
  )
  cfg.scene.sensors = (
    *cfg.scene.sensors,
    GROUND_CONTACT_FORCE_SENSOR,
    FEET_CONTACT_FORCE_SENSOR,
    UPPER_CONTACT_FORCE_SENSOR,
  )

  # --- Events --------------------------------------------------------------
  # Use the focused 2-clip shoulder-roll prior (jump → ukemi roll → stand up): its
  # manifold spans the apex (pelvis≈1.50) down to the floor roll (≈0.08) and back
  # to a stand (≈0.65), so the post-touchdown roll+getup is well covered. floor
  # maps to floor (no z-offset) — the drop ends on the real ground at z=0.
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "dataset_mini/chuong_data/roll_safety_pretrained.pt"
  )
  cfg.events["init_smp_state"].params["smp_z_offset"] = 0.0
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # Respawn floating in the air simulating a hard throw: diagonal velocity kick
  # (lateral + downward) so the robot arrives at the ground with real impact energy
  # and must roll to absorb it — not just a passive free-fall.
  # lateral_speed_range [m/s]: horizontal kick in a random direction.
  # init_ang_vel_range [rad/s]: initial rotation about the axis perpendicular to the
  #   lateral velocity (pre-loading the shoulder roll before touchdown).
  # init_z_vel_range [m/s]: downward velocity injected at spawn (simulates being
  #   thrown down hard; adds impact energy on top of the free-fall).
  # Velocity mix for a FEET-FIRST landing → roll:
  #  * upright_spawn=True so the robot falls feet-down and CAN land on its feet.
  #  * lateral DOMINATES the downward kick: a ukemi roll needs HORIZONTAL momentum
  #    to roll *through* the contact; a vertical slam can't be rolled out of. The
  #    forward speed is what the robot converts into the roll AFTER the feet land.
  #  * init_ang_vel kept SMALL: a large pre-rotation in the air would tumble the
  #    robot off its upright orientation and ruin the feet-first landing. The roll
  #    must be generated on the ground (knees/hips) after touchdown, not in flight.
  cfg.events["reset_drop_in_air"] = EventTermCfg(
    func=reset_drop_in_air,
    mode="reset",
    params={
      "pelvis_height_range": DROP_PELVIS_HEIGHT_RANGE,
      "drop_fraction": 1.0,
      "lateral_speed_range": (2.0, 4.5),
      "init_ang_vel_range": (0.3, 0.8),
      "init_z_vel_range": (0.5, 2.0),
      "upright_spawn": True,
      "upright_joint_noise": 0.1,
      # Throw only along robot's forward/backward axis (sagittal plane ukemi).
      "forward_backward_only": True,
      # Randomise spawn yaw 360° each episode → full directional variety in world
      # frame while keeping the throw semantically forward/backward in robot frame.
      "random_yaw_spawn": True,
      # Always throw along robot's forward axis (never backward).
      "forward_only": True,
    },
  )
  # Per-step touchdown detection: latch env._drop_landed once ground contact occurs.
  cfg.events["update_landed_latch"] = EventTermCfg(
    func=update_landed_latch,
    mode="step",
    params={
      "sensor_name": GROUND_CONTACT_FORCE_SENSOR.name,
      "force_threshold": LANDED_FORCE_THRESHOLD,
    },
  )

  # PUSH is meaningless on a robot in free fall and only perturbs the held pose;
  # drop it for this task (it is already absent in play mode).
  cfg.events.pop("push_robot", None)

  # --- Rewards -------------------------------------------------------------
  # All terms ∈ [0,1], weights sum to 1.0. SMP-gated with a PHASE-DEPENDENT floor:
  #   AIRBORNE (smp_floor=0.5): a rigidly-held upright pose is off the prior's
  #     manifold, so a high floor keeps the hold-pose gradient alive.
  #   GROUNDED (smp_floor_grounded=0.30): higher than before (was 0.15) so that
  #     the getup gradient stays alive even when the robot is off-manifold after
  #     landing — at 0.15 the reward was too flat to learn standing up.
  #
  # Weight split (sums to 1.0):
  #   - AIRBORNE HOLD   (hold_initial_pose              = 0.08)
  #   - LANDING QUALITY (feet_first+avoid_upper+soft    = 0.22)
  #   - ROLL            (roll_momentum+contact_force    = 0.16)
  #   - GETUP SPINE     (upright+track_head+upward_vel  = 0.54) ← dominant
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "smp_floor": 0.5,   # AIRBORNE: rigid hold is off-manifold → keep its gradient.
      "smp_floor_grounded": 0.30,  # GROUNDED: doubled from 0.15 → stronger getup gradient.
      "task_terms": (
        # AIRBORNE HOLD (0.08): keep the upright ready stance until touchdown.
        (
          mdp.hold_initial_pose,
          0.057,
          {"scale": 8.0},
        ),
        # LANDING — FEET FIRST (0.070): feet carry the touchdown load, not the back.
        (
          mdp.feet_first_contact,
          0.070,
          {
            "feet_sensor_name": FEET_CONTACT_FORCE_SENSOR.name,
            "upper_sensor_name": UPPER_CONTACT_FORCE_SENSOR.name,
            "sharpness": 1.5,
          },
        ),
        # LANDING — FLAT SOLE (0.052): land on the flat foot sole, not toe/edge —
        # "mặt bàn chân chạm đất". Only scored while the feet are in contact.
        (
          mdp.foot_flat,
          0.052,
          {"feet_sensor_name": FEET_CONTACT_FORCE_SENSOR.name, "scale": 1.0},
        ),
        # LANDING — NO BACK (0.057): penalise hard upper-body contact. max_force=30 N.
        (
          mdp.avoid_upper_impact,
          0.057,
          {
            "sensor_name": UPPER_CONTACT_FORCE_SENSOR.name,
            "max_force": 30.0,
            "scale": 1.0,
          },
        ),
        # LANDING — SOFT (0.04): low downward speed at impact (head < 0.45 m).
        (
          mdp.soft_landing,
          0.028,
          {"scale": 4.0, "head_floor_threshold": HEAD_FLOOR_THRESHOLD},
        ),
        # ROLL FORWARD (0.086): body-frame FORWARD pitch ≥ 1.5 rad/s — "roll về đằng
        # trước". Directional (replaces the omnidirectional roll_momentum): a forward
        # somersault scores high, a backward roll / sideways topple scores ~0.
        # Active head < HEAD_FULL_ROLL_THRESHOLD (1.2 m) so it covers the whole roll
        # and the rise; SMP gate prevents off-manifold spinning-in-place when upright.
        (
          mdp.forward_roll,
          0.086,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_FULL_ROLL_THRESHOLD,
          },
        ),
        # MINIMISE CONTACT (0.08): low peak force while head < 0.45 m (impact only).
        # max_force=300 N leaves room for leg push during the roll→stand transition.
        (
          mdp.rolling_contact_force,
          0.057,
          {
            "sensor_name": GROUND_CONTACT_FORCE_SENSOR.name,
            "max_force": 300.0,
            "scale": 1.0,
            "head_roll_threshold": HEAD_FLOOR_THRESHOLD,
          },
        ),
        # ROLL SHAPE — KNEE BEND (0.10): flex the knees while low to absorb and drop
        # into the roll instead of a stiff straight-legged catch. "knee phải cong".
        # Head-gated (head < HEAD_ROLL_THRESHOLD) so it never fights the final stand.
        (
          mdp.knee_bend,
          0.070,
          {
            "target_flex": 1.2,
            "scale": 2.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # ROLL SHAPE — HEAD TUCK (0.084): dive the head below the pelvis so the body
        # rolls over the shoulders/back (one full rotation), not a stiff topple.
        # "đầu chúi xuống quay 1 vòng". Self-gating: 0 when standing (head above pelvis).
        (
          mdp.head_tuck,
          0.084,
          {"target_drop": 0.30},
        ),
        # ROLL SHAPE — HANDS FORWARD (0.060): both wrists reach forward to brace and
        # cushion the roll — "2 điểm cuối của tay cùng hướng về phía trước để đỡ".
        (
          mdp.hands_forward,
          0.060,
          {"target_reach": 0.20, "head_roll_threshold": HEAD_FULL_ROLL_THRESHOLD},
        ),
        # GETUP — torso upright from any pose, monotonic lying→standing.
        (
          mdp.upright_progress,
          0.141,
          {"scale": 1.0},
        ),
        # GETUP — drive the head toward 0.95 m standing height.
        (
          mdp.track_head_height,
          0.141,
          {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0},
        ),
        # GETUP — upward head velocity while low: breaks the "lie still" optimum —
        # ~0 for a stationary robot, only pays while rising.
        (
          mdp.upward_velocity,
          0.097,
          {
            "target_velocity": 0.40,
            "head_height_threshold": HEAD_UP_THRESHOLD,
            "scale": 100.0,
          },
        ),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  # Start pose is a fallen/airborne off-manifold pose — drop self_collision.
  cfg.terminations.pop("self_collision", None)

  # Relaxed SMP collapse guard. grace_steps=75 (~1.5 s at 50 Hz) gives the robot
  # time to absorb a hard throw impact and re-enter the manifold via rolling before
  # being terminated — harder throws produce longer off-manifold windows at landing.
  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.005, "ws": 6.0, "grace_steps": 75},
  )
  # Physics-divergence guard (contact blow-ups → NaN), independent of the SMP score.
  cfg.terminations["diverged"] = TerminationTermCfg(
    func=mdp.diverged,
    params={"max_lin_speed": 25.0, "max_ang_speed": 40.0},
  )
  # Truncate (time_out=True) once stably upright so the value bootstraps correctly.
  cfg.terminations["stood_up"] = TerminationTermCfg(
    func=mdp.stood_up,
    time_out=True,
    params={"head_height": HEAD_STOOD_UP, "max_speed": 0.5, "hold_steps": 25},
  )

  cfg.episode_length_s = 5

  return cfg
