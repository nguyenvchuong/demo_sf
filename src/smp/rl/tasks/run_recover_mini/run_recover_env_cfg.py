"""Mini_M1v1 run-and-recover task with SMP guidance.

Scenario: robot runs forward at 3 m/s continuously.  Periodic strong pushes
knock it over; it must perform ukemi (roll to absorb the fall), stand back up,
and resume running.  Stopping while upright → episode terminates.

─────────────────────────────────────────────────────────────────────────────
Reward architecture (two categories)
─────────────────────────────────────────────────────────────────────────────

[UNGATED — always full strength, never discounted by SMP gate]

  run_velocity      w=0.50  exp(-0.5·‖v_tar−v_act‖²) when upright, 0 when fallen.
                            Main running signal; 0 when down → cost for being fallen.

  gait_symmetry     w=0.30  exp(-0.1·mean‖dq_L+dq_R‖²) when upright, 1.0 when fallen.
                            Rewards anti-phase L/R joint velocity (beautiful gait).

  running_height    w=0.30  exp(-30·max(0.30−pelvis_z, 0)²) when upright, 1.0 fallen.
                            Penalises crouching on thighs; maintains proper stance height.

  head_safe_landing w=0.20  exp(-6·max(−head_vz, 0)²) when head < 0.25 m, else 0.
                            Penalises head diving into ground — rewards tucked ukemi.
                            Returns 0 outside impact phase (no free reward).

[SMP-GATED — run prior × (0.5 + 0.5·SMP), recovery style]

  task_smp_product  w=1.00  Σ wᵢ·termᵢ gated by (smp_floor + (1−smp_floor)·SMP).
    upright_progress  0.25  — always-on torso orientation potential (lying→standing).
    track_head_height 0.20  — always-on: stand tall.
    proactive_roll    0.20  — tilt > 0.6: commit to roll.
    upward_velocity   0.15  — head < 0.50 m: rise quickly.
    roll_momentum     0.10  — head < 0.42 m: sustain roll.
    soft_landing      0.10  — head < 0.30 m: low pelvis impact velocity.

─────────────────────────────────────────────────────────────────────────────
Reward values by state (approximate):
─────────────────────────────────────────────────────────────────────────────
  Running 3 m/s, symmetric, tall  (SMP≈0.8): 0.50+0.30+0.30+0+1.0×0.9 = 2.00
  Running 3 m/s, asymmetric/low   (SMP≈0.8): 0.50+0.15+0.15+0+0.7×0.9 = 1.43
  Lying still               (SMP≈0.1, floor): 0+0.30+0.30+0+0.5×0.55  = 0.88
  Rolling nicely ukemi      (SMP≈0.1, floor): 0+0.30+0.30+0.15+0.7×0.55= 1.14
  Good head-tucked landing  (head<0.25, good): 0+0.30+0.30+0.20+...     = ...

─────────────────────────────────────────────────────────────────────────────
Why run prior (not getup prior):
─────────────────────────────────────────────────────────────────────────────
  Run prior → SMP high while running → gate≈0.9 → style bonus for gait quality.
  Recovery works via smp_floor=0.5 + strong ungated cost for being fallen.
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

# Running target speed (m/s).
_TARGET_SPEED: float = 3.0

# Stall detection: upright + forward speed below this → stall counter increments.
_STALL_MIN_SPEED: float = 0.5   # 17% of 3 m/s target
_STALL_HOLD_STEPS: int = 30     # 30 × 0.02 s = 0.6 s before terminate


def mini_run_recover_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the Mini_M1v1 run-and-recover env cfg with SMP guidance."""
  cfg = mini_smp_env_cfg(play=play)

  # --- Scene ---------------------------------------------------------------
  cfg.scene.entities["robot"].spec_fn = get_mini_spec_with_head

  # --- Commands ------------------------------------------------------------
  cfg.commands["steering"] = mdp.SteeringCommandCfg(
    entity_name="robot",
    resampling_time_range=(5.0, 10.0),
    rand_tar_dir=False,
    rand_face_dir=False,
    tar_speed_min=_TARGET_SPEED,
    tar_speed_max=_TARGET_SPEED,
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
  # Run prior: high SMP while running → gate≈0.9 → style bonus for gait.
  # smp_floor=0.5 covers recovery poses being off the run manifold.
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "logs/pretrain/pretrain/20260606_121844/checkpoint_00300.pt"
  )

  # Strong push: enough to knock over a robot running at 3 m/s.
  # Interval 5–12 s → full fall-roll-standup-run cycle between pushes.
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

  # Reset stall counter on episode reset.
  cfg.events["reset_stall_counter"] = EventTermCfg(
    func=mdp.reset_stall_counter, mode="reset"
  )

  # --- Rewards -------------------------------------------------------------

  # [A] Ungated running signal: 0 when fallen → cost for being down.
  cfg.rewards["run_velocity"] = RewardTermCfg(
    func=mdp.running_velocity,
    weight=0.5,
    params={
      "command_name": "steering",
      "vel_err_scale": 0.5,
      "head_run_threshold": HEAD_STOOD_UP,
    },
  )

  # [B] Ungated gait symmetry: beautiful left-right anti-phase motion.
  #     Returns 1.0 when fallen → no penalty for asymmetric ukemi.
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=mdp.gait_symmetry,
    weight=0.3,
    params={
      "scale": 0.1,
      "head_run_threshold": HEAD_STOOD_UP,
    },
  )

  # [C] Ungated running height: penalise squatting on thighs while running.
  #     Mini standing pelvis ≈ 0.34–0.37 m; target 0.30 m allows slight crouch.
  #     Returns 1.0 when fallen → no penalty for being low during ukemi.
  cfg.rewards["running_height"] = RewardTermCfg(
    func=mdp.running_height,
    weight=0.3,
    params={
      "target_height": 0.30,
      "scale": 30.0,
      "head_run_threshold": HEAD_STOOD_UP,
    },
  )

  # [D] Ungated head protection: penalise head diving into ground.
  #     Returns 0 outside impact phase — no free reward when not falling.
  cfg.rewards["head_safe_landing"] = RewardTermCfg(
    func=mdp.head_safe_landing,
    weight=0.2,
    params={
      "scale": 6.0,
      "head_floor_threshold": 0.25,
    },
  )

  # [E] SMP-gated recovery style (run prior, smp_floor=0.5).
  #     Running: SMP≈0.8 → gate≈0.9 → all 6 terms return 1.0 → full style bonus.
  #     Fallen:  SMP≈0.1 → gate=0.55 → recovery gradient at 55% strength.
  cfg.rewards["task_smp_product"] = RewardTermCfg(
    func=task_smp_product,
    weight=1.0,
    params={
      "smp_floor": 0.5,
      "task_terms": (
        (mdp.upright_progress,  0.25, {"scale": 1.0}),
        (mdp.track_head_height, 0.20, {"target_height": HEAD_TARGET_HEIGHT, "scale": 1.0}),
        (mdp.proactive_roll,    0.20, {"tilt_threshold": 0.6, "target_ang_vel": 2.0, "scale": 0.5}),
        (mdp.upward_velocity,   0.15, {"target_velocity": 0.40, "head_height_threshold": HEAD_UP_THRESHOLD, "scale": 100.0}),
        (mdp.roll_momentum,     0.10, {"target_ang_vel": 1.5, "scale": 1.0, "head_roll_threshold": HEAD_ROLL_THRESHOLD}),
        (mdp.soft_landing,      0.10, {"scale": 4.0, "head_floor_threshold": HEAD_FLOOR_THRESHOLD}),
      ),
    },
  )

  # --- Terminations --------------------------------------------------------
  cfg.terminations.pop("self_collision", None)
  cfg.terminations.pop("smp_too_low", None)

  # "Must keep running": upright + forward speed < 0.5 m/s for 0.6 s → reset.
  cfg.terminations["stalled_while_upright"] = TerminationTermCfg(
    func=mdp.stalled_while_upright,
    params={
      "command_name": "steering",
      "min_forward_speed": _STALL_MIN_SPEED,
      "head_threshold": HEAD_STOOD_UP,
      "hold_steps": _STALL_HOLD_STEPS,
      "grace_steps": 50,
    },
  )

  # Physics-divergence guard.
  cfg.terminations["diverged"] = TerminationTermCfg(
    func=mdp.diverged,
    params={"max_lin_speed": 25.0, "max_ang_speed": 40.0},
  )

  cfg.episode_length_s = 20.0

  return cfg
