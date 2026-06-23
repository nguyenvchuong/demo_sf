"""Mini_M1v1 jump-platform task with SMP guidance.

The robot starts standing on a staircase top tread (see ``Mini_M1v1_platform.xml``)
and must jump off it down to the floor, softly land, and roll to reduce impact
(recovering to a stand) — guided by an SMP prior trained on the two
``jump_to_shoulder_roll_R_10{1,2}__A416`` mocap clips (``minimal_pretrained.pt``).
The top-tread height is domain-randomized per episode (0.55–0.65 m) for
robustness. The reward/termination shape is shared with ``getup_mini`` (impact →
roll → stand-up phases): jumping off a platform and recovering from a fall is the
same getup problem with a different starting condition.
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.events import reset_stand_on_platform
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup_mini import mdp
from smp.robot.Mini_M1v1.mini_m11_constants import KNEES_BENT_KEYFRAME
from smp.sampling.feature_to_state import EE_BODY_NAMES

_PLATFORM_XML = Path("src/smp/robot/Mini_M1v1/Mini_M1v1_platform.xml")

# Mini_M1v1 geometry (from Mini_M1v1.xml), MEASURED by FK at the KNEES_BENT
# keyframe (not guessed): torso_link sits at pelvis height, the ``head`` site is
# +0.31 m above it, and the standing pelvis is 0.77 m, so the standing head site
# is at 0.77 + 0.31 = 1.08 m. (The old "~0.68 m" comment/thresholds were carried
# over from a half-scale robot and made every head-height term saturate while the
# robot was still in a deep crouch — re-derived below against the real 1.08 m.)
HEAD_POS_IN_TORSO: tuple[float, float, float] = (0.0, 0.0, 0.31)

# Reward / termination heights scaled to Mini's MEASURED 1.08 m standing head
# height (and the ~0.96 m head of the prior's floor-stand recovery, pelvis 0.65).
HEAD_TARGET_HEIGHT: float = 0.95  # track_head_height goal (the recovered floor stand)
HEAD_UP_THRESHOLD: float = 0.80  # upward_velocity: drive while head below this
HEAD_STOOD_UP: float = 0.88  # stood_up: success threshold (clearly upright, achievable)
# Ukemi-specific thresholds (head_z is low while rolling — torso horizontal).
HEAD_FLOOR_THRESHOLD: float = 0.45  # soft_landing: below = impact/contact phase
HEAD_ROLL_THRESHOLD: float = 0.62  # roll_momentum: below = active rolling phase

# Platform geometry (Mini_M1v1_platform.xml): the symmetric staircase from
# scene_Mini_M1v1_stair_jump.xml — the TOP tread (where the robot spawns) is at
# z=0.60, 1.5 m deep × 1.2 m wide, with 3 descending 0.15 m steps on each side.
PLATFORM_HEIGHT: float = 0.60
PLATFORM_SPAWN_POS: tuple[float, float, float] = (
  0.0,
  0.0,
  PLATFORM_HEIGHT + KNEES_BENT_KEYFRAME.pos[2],
)
# Per-episode domain randomization of the TOP tread surface height, for
# robustness to platform-height error at deployment. The top tread is the box
# geom ``stair_top_collision`` (default pos_z=0.30, half-height 0.30 → surface at
# 0.60). We randomize ONLY its pos_z to ``PLATFORM_HEIGHT_RANGE/2`` per reset so
# the tread surface lands uniformly in ±0.05 m of 0.60 m; the box bottom shifts
# with it (to ±0.05 m of the floor), which is harmless for a static support geom.
TOP_STAIR_GEOM: str = "stair_top_collision"
TOP_STAIR_HALF_HEIGHT: float = 0.30  # stair_top_collision size_z (kept fixed)
PLATFORM_HEIGHT_RANGE: tuple[float, float] = (0.55, 0.65)

# Stable standing spawn: fraction of envs that begin each episode standing STABLY
# on the (randomized) top tread instead of being GSI-seeded mid-motion. They start
# from the robot's neutral keyframe (KNEES_BENT, pelvis 0.77 m) with zero velocity,
# so the robot stands stably on the stair first and the diffusion reward then drives
# the jump-off. The rest keep GSI (fallen/rolling seeds) for land/roll practice.
# Set to 1.0 for every episode to start standing (e.g. for play); 0.0 disables it.
STAND_FRACTION: float = 0.5
STAND_PELVIS_HEIGHT: float = KNEES_BENT_KEYFRAME.pos[2]  # 0.77 m (measured stand)
# The robot's env-origin x (0.0) sits only ~0.15 m from the tread's +x drop edge
# (tread center env-local x≈-0.45, +x half-extent 0.6 → edge at +0.15), so a robot
# spawned there teeters at the lip and tips over the drop. Set the stable standing
# spawn BACK from the edge (negative = away from the +x drop) for full foot support.
STAND_SPAWN_X_OFFSET: float = -0.30


@functools.lru_cache(maxsize=1)
def _standing_ee_offsets() -> tuple[tuple[float, float, float], ...]:
  """End-effector positions relative to the pelvis at the neutral standing stance,
  via FK on the spec (KNEES_BENT = all-zero joints, identity orientation). Used to
  prime the SMP feature buffer for the stable standing spawn. Cached (FK is run once).
  """
  spec = get_mini_platform_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[0:3] = (0.0, 0.0, STAND_PELVIS_HEIGHT)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # joints stay at qpos0 = 0 (KNEES_BENT)
  mujoco.mj_forward(model, data)
  root = data.qpos[0:3]
  return tuple(
    tuple(float(v) for v in (data.xpos[model.body(name).id] - root))
    for name in EE_BODY_NAMES
  )

# Robot-vs-terrain peak contact-force sensor, used by ``rolling_contact_force``
# to penalise slamming a single link into the floor on landing / mid-roll.
GROUND_CONTACT_FORCE_SENSOR = ContactSensorCfg(
  name="ground_contact_force",
  primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="maxforce",
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
  cfg.scene.sensors = (*cfg.scene.sensors, GROUND_CONTACT_FORCE_SENSOR)

  # The collidable staircase generates many more simultaneous contacts than the
  # old single box (the robot can touch several steps + the platform at once,
  # especially mid-roll). The shared cfg's nconmax=35 overflows ("nconmax must
  # be >= 36"); raise the contact buffer to accommodate the stair geometry.
  cfg.sim.nconmax = 256

  # --- Events --------------------------------------------------------------
  # SMP prior trained on just the two jump_to_shoulder_roll_R_10{1,2}__A416
  # mocap clips (jump off a platform → shoulder/safety roll → stand up). Both
  # clips share the SAME arc (~6.3 s @ 50 Hz): start standing on the platform
  # (pelvis_z≈1.35), crouch+leap to an apex (≈1.50), drop to the floor and roll
  # (≈0.08), then stand back up (≈0.65). The embedded GSI quantiles match this
  # data (q_high[2]≈1.50), so sampled poses span the full jump→impact→roll→stand
  # sequence. ``minimal_pretrained.pt`` is the focused 2-clip prior; the older
  # ``roll_platform_pretrained.pt`` mixed in taller-platform clips (q_high[2]≈1.76).
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "dataset_mini/chuong_data/roll_platform/minimal_pretrained.pt"
  )
  # No z-offset: the task is "jump FROM the 0.60 m platform DOWN to the floor and
  # roll", so the prior's floor (z=0) must map to the real floor (z=0) and its
  # platform must map to the real platform. The data start (pelvis≈1.35) minus the
  # standing pelvis (KNEES_BENT z=0.77) ≈ 0.58 m ≈ the real PLATFORM_HEIGHT (0.60),
  # so heights already match — no shift. (A 0.60 offset would replay the whole
  # jump+roll ON TOP of the platform, so the robot would never reach the floor.)
  cfg.events["init_smp_state"].params["smp_z_offset"] = 0.0
  cfg.events["reset_stand_counter"] = EventTermCfg(
    func=mdp.reset_stand_counter, mode="reset"
  )

  # ROBUSTNESS: randomize the top-tread surface height per episode (reset mode)
  # so the policy does not overfit one platform height. ``dr.geom_pos`` with
  # operation="abs" SETS the geom's local z to a value sampled from the given
  # range; here we drive only axis 2 (z) of ``stair_top_collision`` to half the
  # target surface height, so the tread top lands uniformly in [0.55, 0.65] m
  # (its half-height stays 0.30, so surface = pos_z + 0.30). The ``model_fields``
  # decorator on ``dr.geom_pos`` auto-registers ``geom_pos`` for per-world memory.
  # (Reset mode = once per episode, the standard DR cadence; randomizing the
  # static support every physics step would teleport it under the robot's feet.)
  lo, hi = PLATFORM_HEIGHT_RANGE
  cfg.events["randomize_platform_height"] = EventTermCfg(
    func=dr.geom_pos,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=[TOP_STAIR_GEOM]),
      "ranges": {2: (lo - TOP_STAIR_HALF_HEIGHT, hi - TOP_STAIR_HALF_HEIGHT)},
      "operation": "abs",
    },
  )

  # STABLE STANDING SPAWN (runs AFTER gsi_reset + randomize_platform_height):
  # start a STAND_FRACTION of envs standing stably on the randomized tread (zero
  # velocity, feet on the surface) so the robot stands stably on the stair first
  # and then learns the jump-off from the diffusion reward — fixing the wobble
  # from GSI seeding every env mid-motion at a fixed (mismatched) height. The rest
  # keep GSI for in-fall/roll practice. ee_offsets prime the SMP buffer's standing
  # window so the SMP reward is consistent from step 0.
  cfg.events["reset_stand_on_platform"] = EventTermCfg(
    func=reset_stand_on_platform,
    mode="reset",
    params={
      "stand_fraction": STAND_FRACTION,
      "pelvis_height": STAND_PELVIS_HEIGHT,
      "ee_offsets": _standing_ee_offsets(),
      "top_geom_name": TOP_STAIR_GEOM,
      "top_geom_half_height": TOP_STAIR_HALF_HEIGHT,
      "spawn_x_offset": STAND_SPAWN_X_OFFSET,
    },
  )

  # PUSH: the shared push perturbs root velocity every 1–3 s. At full strength
  # (±0.5 m/s, ±0.52 rad/s roll/pitch) it topples the robot off the narrow tread
  # before it can commit to the jump, so for this task we DELAY the first push
  # (≥2.5 s — past the typical jump-off at ~1.7 s in the prior) and halve the
  # tread-toppling vertical/angular components. The GSI-seeded fallen envs still
  # get a meaningful push for roll robustness. (Absent in play mode.)
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (2.5, 4.0)
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.26, 0.26),
      "pitch": (-0.26, 0.26),
      "yaw": (-0.78, 0.78),
    }

  # --- Rewards -------------------------------------------------------------
  # Re-derived from scratch for the requested behaviour: JUMP off the top tread →
  # SOFT-LAND on the floor → ROLL to reduce impact (recover to a stand second).
  # All terms ∈ [0,1] and weights sum to 1; SMP-gated (gate = smp_floor +
  # (1−smp_floor)·r_smp). The 2-clip prior carries the jump+roll STYLE; the task
  # terms supply the phase incentives. Heights use the MEASURED 1.08 m standing
  # head (see threshold constants above), not the stale 0.68 m.
  #
  # Behaviour arc (base = pelvis world-z; head = head-site world-z):
  #   1. ON TREAD     (base ≈ 1.37): descend_off_platform penalises staying up
  #      → the robot must JUMP OFF the edge toward the open floor.
  #   2. FALL         (base ↓, torso tilts): proactive_roll rewards tucking into
  #      the rotation mid-fall instead of falling flat/rigid.
  #   3. LAND         (head < 0.45): soft_landing rewards a low vertical impact
  #      speed — absorb the drop rather than slam into it.
  #   4. ROLL         (head < 0.62): roll_momentum rewards rolling angular
  #      velocity and rolling_contact_force penalises peak contact force, so the
  #      drop energy is spread across body+time (the safety roll / ukemi).
  #   5. RECOVER      (head → 0.95): track_head_height + upright_progress bring it
  #      back to a floor stand.
  #
  # Weight emphasis follows the request: land+roll (soft_landing 0.20 + roll
  # terms 0.40 = 0.60) dominates, jump-off 0.25 gates it, recovery 0.15 trails.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "smp_floor": 0.0,
      "task_terms": (
        # JUMP-OFF driver: penalise remaining at platform height; 1.0 once the
        # base is at/below floor-standing height (0.80 ≈ standing pelvis 0.77).
        (
          mdp.descend_off_platform,
          0.25,
          {"target_height": 0.80, "scale": 4.0},
        ),
        # SOFT LANDING: absorb impact — penalise high downward speed at touchdown.
        (
          mdp.soft_landing,
          0.20,
          {"scale": 4.0, "head_floor_threshold": HEAD_FLOOR_THRESHOLD},
        ),
        # ROLL: reward horizontal angular velocity once low — the safety roll.
        (
          mdp.roll_momentum,
          0.15,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # ROLL → REDUCE IMPACT: penalise peak ground-contact force while rolling,
        # so the drop is dissipated across the body/time (the shoulder roll)
        # instead of slamming one link. Weighted up — this is the user's explicit
        # "roll to reduce impact" objective.
        (
          mdp.rolling_contact_force,
          0.15,
          {
            "sensor_name": GROUND_CONTACT_FORCE_SENSOR.name,
            "max_force": 150.0,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # Initiate the roll mid-fall once the torso tilts past upright (proactive,
        # tilt-gated so it fires while still high — commits to a roll, not a flop).
        (
          mdp.proactive_roll,
          0.10,
          {"tilt_threshold": 0.6, "target_ang_vel": 2.0, "scale": 0.5},
        ),
        # RECOVER: stand tall on the floor after the roll (head → 0.95).
        (
          mdp.track_head_height,
          0.10,
          {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0},
        ),
        # RECOVER: torso upright for the final floor stand.
        (
          mdp.upright_progress,
          0.05,
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
