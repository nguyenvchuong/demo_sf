"""Mini_M1v1 run-and-recover task with SMP guidance.

Scenario: robot runs forward at ~1 m/s. Periodic strong pushes knock it over.
The robot must roll to absorb the fall (ukemi), stand back up, and resume running.
Success is measured across the whole episode — there is no early termination for
standing; the policy must continuously run, fall, recover, and run again.

Reward structure (all terms ∈ [0, 1], weights sum to 1.0, SMP-gated):

  running_velocity   0.30  — velocity tracking ONLY when upright (head ≥ 0.62 m);
                             returns 1.0 when fallen so it doesn't block recovery.
  upright_progress   0.20  — always-on torso orientation potential (lying→standing).
  track_head_height  0.15  — always-on: stand tall.
  proactive_roll     0.15  — when tilt > 0.6 rad: reward committing to a roll.
  upward_velocity    0.10  — rising phase (head < 0.50 m): rise quickly.
  roll_momentum      0.05  — on-ground phase (head < 0.42 m): sustain the roll.
  soft_landing       0.05  — impact phase (head < 0.30 m): absorb the fall.

SMP floor = 0.3 so off-manifold fallen poses still receive 30% of task gradient.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from smp.rl.env_cfg import mini_smp_env_cfg
from smp.rl.rewards import task_smp_product
from smp.rl.tasks.getup_mini.getup_env_cfg import (
  HEAD_FLOOR_THRESHOLD,
  HEAD_ROLL_THRESHOLD,
  HEAD_STOOD_UP,
  HEAD_TARGET_HEIGHT,
  HEAD_UP_THRESHOLD,
  get_mini_spec_with_head,
)
from smp.rl.tasks.run_recover_mini import mdp


def mini_run_recover_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Mini_M1v1 run-and-recover env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  cfg.scene.entities["robot"].spec_fn = get_mini_spec_with_head

  # --- Commands ------------------------------------------------------------
  # Fixed forward direction (+x), fixed 1 m/s.  No resampling of direction;
  # speed is fixed so the policy always knows what "running" looks like.
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=1.0,
    tar_speed_max=1.0,
    debug_vis=True,
  )

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "steering"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Events --------------------------------------------------------------
  # Pretrained SMP prior: ideally a checkpoint trained on BOTH running and
  # getup/roll motions.  Fall back to the run prior — the smp_floor=0.3 below
  # ensures recovery still earns gradient even off the running manifold.
  # Replace with a mixed-motion checkpoint once available.
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_lafan_run.pt"
  )

  # Strong push: enough linear + angular impulse to knock over a robot running
  # at 1 m/s.  Interval is long enough (5–12 s) for a full fall-recover-run
  # cycle before the next push.
  cfg.events["push_robot"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(5.0, 12.0),
    params={
      "velocity_range": {
        "x": (-4.0, 4.0),
        "y": (-4.0, 4.0),
        "z": (-1.5, 1.5),
        "roll": (-3.0, 3.0),
        "pitch": (-3.0, 3.0),
        "yaw": (-2.0, 2.0),
      },
    },
  )

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      # smp_floor > 0 so off-manifold recovery still earns task gradient.
      "smp_floor": 0.3,
      "task_terms": (
        # Running phase: velocity tracking when upright, else 1.0 (no interference).
        (
          mdp.running_velocity,
          0.30,
          {
            "command_name": "steering",
            "vel_err_scale": 0.5,
            "head_run_threshold": HEAD_STOOD_UP,
          },
        ),
        # Always-on: rotate the torso upright from ANY pose (core recovery signal).
        (
          mdp.upright_progress,
          0.20,
          {"scale": 1.0},
        ),
        # Always-on: drive the head toward standing height.
        (
          mdp.track_head_height,
          0.15,
          {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0},
        ),
        # Proactive roll: reward committing to a roll once the topple is past the
        # point of recovery (tilt > threshold).  Fires while still up high so the
        # robot chooses to roll early rather than fighting the fall rigidly.
        (
          mdp.proactive_roll,
          0.15,
          {
            "tilt_threshold": 0.6,
            "target_ang_vel": 2.0,
            "scale": 0.5,
          },
        ),
        # Rising phase: drive head upward quickly until near-standing height.
        (
          mdp.upward_velocity,
          0.10,
          {
            "target_velocity": 0.40,
            "head_height_threshold": HEAD_UP_THRESHOLD,
            "scale": 100.0,
          },
        ),
        # On-ground rolling: sustain angular momentum while low.
        (
          mdp.roll_momentum,
          0.05,
          {
            "target_ang_vel": 1.5,
            "scale": 1.0,
            "head_roll_threshold": HEAD_ROLL_THRESHOLD,
          },
        ),
        # Impact phase: reward a soft (rolling) landing.
        # (
        #   mdp.soft_landing,
        #   0.05,
        #   {
        #     "scale": 4.0,
        #     "head_floor_threshold": HEAD_FLOOR_THRESHOLD,
        #   },
        # ),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  # The robot legitimately rolls on the ground — remove false-trigger guards.
  cfg.terminations.pop("self_collision", None)

  # No base_too_low: fallen pose is valid during recovery.
  # No stood_up truncation: success is running again, not just standing once.

  # Relax smp_too_low: wide grace window so recovery doesn't trigger a reset
  # (robot is off-manifold while rolling).  Only kills sustained degeneracy.
  cfg.terminations["smp_too_low"] = TerminationTermCfg(
    func=mdp.smp_too_low,
    params={"threshold": 0.005, "ws": 6.0, "grace_steps": 50},
  )

  # Physics-divergence guard: terminate runaway contact blow-ups before NaN.
  cfg.terminations["diverged"] = TerminationTermCfg(
    func=mdp.diverged,
    params={"max_lin_speed": 25.0, "max_ang_speed": 40.0},
  )

  # Long episode to allow multiple run → push → recover → run cycles.
  cfg.episode_length_s = 20.0

  return cfg
