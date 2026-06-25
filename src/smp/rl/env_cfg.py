"""Shared G1 + SMP guidance env config.

The SMP feature buffer and frozen denoiser are attached to the stock
``ManagerBasedRlEnv`` via the startup/reset events in ``smp.rl.events``.
Per-task configs extend this with task-specific commands/observations/rewards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from smp.robot.Mini_M1v1 import (
  get_mini_m1v1_robot_cfg,
  MINI_M1V1_ACTION_SCALE
)
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import BuiltinSensor
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import illegal_contact
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from smp.rl.events import (
  gsi_refresh,
  gsi_reset,
  init_smp_state,
)
# --- Shared sensors ----------------------------------------------------------

G1_SELF_COLLISION = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

# Self-collision: upper body (torso subtree = shoulders/arms/wrists) vs
# lower body (pelvis + hips + legs). This catches arm-vs-leg or arm-vs-arm
# contacts without false-positives from feet touching the ground.
# NOTE: pelvis_link is the root body of Mini_M1v1 — subtree("pelvis_link")
# covers the ENTIRE robot including feet, which fire on every ground contact.
# Using subtree("torso_link") restricts to the upper body only.
MINI_SELF_COLLISION = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)


@dataclass(kw_only=True)
class G1SmpSceneCfg(SceneCfg):
  """Scene configuration for the G1 + SMP guidance environment."""

  num_envs: int = 1
  extent: float = 2.0
  terrain: TerrainEntityCfg | None = field(
    default_factory=lambda: TerrainEntityCfg(terrain_type="plane")
  )
  entities: dict = field(default_factory=lambda: {"robot": get_g1_robot_cfg()})
  sensors: tuple = field(default_factory=lambda: (G1_SELF_COLLISION,))


@dataclass(kw_only=True)
class MiniSmpSceneCfg(SceneCfg):
  """Scene configuration for the Mini_M1v1 + SMP guidance environment."""

  num_envs: int = 1
  extent: float = 2.0
  terrain: TerrainEntityCfg | None = field(
    default_factory=lambda: TerrainEntityCfg(terrain_type="plane")
  )
  entities: dict = field(default_factory=lambda: {"robot": get_mini_m1v1_robot_cfg()})
  sensors: tuple = field(default_factory=lambda: (MINI_SELF_COLLISION,))


def make_smp_viewer() -> ViewerConfig:
  """Shared viewer configuration following the robot's torso."""
  return ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso_link",
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
  )


def make_smp_sim() -> SimulationCfg:
  """Shared simulation configuration."""
  return SimulationCfg(
    nconmax=35,
    njmax=1500,
    mujoco=MujocoCfg(
      timestep=0.005,
      iterations=10,
      ls_iterations=20,
    ),
  )


def projected_gravity_imu(env, sensor_name: str = "robot/imu_lin_acc"):
  """Estimate projected gravity from an IMU accelerometer sensor.

  Computes ``-accelerometer_reading / 9.81``, matching the deployed FSM's
  MQEKF-derived ``-aBody / 9.81`` (see vm_ctrl FSMState_*.cpp) so the policy
  trains on the same gravity-direction signal it sees at runtime, instead of
  the ground-truth quaternion-rotated gravity vector.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return -sensor.data / 9.81


def make_g1_smp_observations() -> dict[str, ObservationGroupCfg]:
  """G1 + SMP observation configuration."""
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
    ),
  }


def make_mini_smp_observations() -> dict[str, ObservationGroupCfg]:
  """Mini_M1v1 + SMP observation configuration."""
  actor_terms = {
    # "base_lin_vel": ObservationTermCfg(
    #   func=mdp.builtin_sensor,
    #   params={"sensor_name": "robot/imu_lin_vel"},
    #   noise=Unoise(n_min=-0.5, n_max=0.5),
    # ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # "projected_gravity": ObservationTermCfg(
    #   func=mdp.projected_gravity,
    #   noise=Unoise(n_min=-0.05, n_max=0.05),
    # ),
    # Accelerometer-derived projected gravity (-aBody/9.81), matching the
    # deployed FSM's MQEKF estimate instead of ground-truth quat rotation.
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity_imu,
      params={"sensor_name": "robot/imu_lin_acc"},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {**actor_terms}

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
    ),
  }


def make_g1_smp_actions() -> dict[str, ActionTermCfg]:
  """G1 + SMP action specification."""
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }


def make_mini_smp_actions() -> dict[str, ActionTermCfg]:
  """Mini_M1v1 + SMP action specification."""
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=MINI_M1V1_ACTION_SCALE,
      use_default_offset=True,
    )
  }


def make_g1_smp_events() -> dict[str, EventTermCfg]:
  """G1 + SMP events config: SMP state init/refresh and domain randomization."""
  return {
    "init_smp_state": EventTermCfg(
      func=init_smp_state,
      mode="startup",
      params={
        "ckpt_path": "logs/pretrain/pretrained.pt",
        "gsi_buffer_size": 4096,
        "gsi_batch_size": 1024,
        "compile_model": True,
        "compile_mode": "max-autotune",
      },
    ),
    "gsi_reset": EventTermCfg(func=gsi_reset, mode="reset", params={}),
    "gsi_refresh": EventTermCfg(
      func=gsi_refresh,
      mode="step",
      params={"num_samples": 1024, "step_interval": 2400},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }


def make_mini_smp_events() -> dict[str, EventTermCfg]:
  """Mini_M1v1 + SMP events config: SMP state init/refresh and domain randomization."""
  return {
    "init_smp_state": EventTermCfg(
      func=init_smp_state,
      mode="startup",
      params={
        "ckpt_path": "logs/pretrain/pretrained.pt",
        "gsi_buffer_size": 4096,
        "gsi_batch_size": 1024,
        "compile_model": True,
        "compile_mode": "max-autotune",
      },
    ),
    "gsi_reset": EventTermCfg(func=gsi_reset, mode="reset", params={}),
    "gsi_refresh": EventTermCfg(
      func=gsi_refresh,
      mode="step",
      params={"num_samples": 1024, "step_interval": 2400},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    # Sustained external pushes with randomized strength AND duration. Force
    # magnitude and impulse length are sampled INDEPENDENTLY per impulse, so a
    # single event continuously spans every regime the getup/rolling policy must
    # be robust to: strong+short (sharp shove), strong+long (sustained hard push
    # that tips the robot over → roll → get up), weak+short (minor balance
    # nudge), weak+long (slow lean/drift). The robot is not free-floating — feet
    # are planted and actuators + ground friction resist — so it genuinely
    # fights each push and only loses balance on the harder samples.
    #
    # body_point_offset lifts the application point 25 cm above the torso CoM:
    # cross(offset, force) adds a tipping torque so horizontal pushes ROLL the
    # robot rather than just sliding it, which is the disturbance getup must
    # recover from. Mini total mass ~35 kg (torso ~10 kg).
    "push_impulse": EventTermCfg(
      func=mdp.apply_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "force_range": (-200.0, 200.0),   # N/component: ~0 (weak) → 200 (strong)
        "torque_range": (-20.0, 20.0),    # Nm: extra spin to provoke rolling
        "duration_s": (0.05, 1.0),        # short snap → long sustained lean
        "cooldown_s": (1.5, 4.0),         # recovery gap (getup needs settle time)
        "body_point_offset": (0.0, 0.0, 0.25),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=r"^(left|right)_foot[1-7]_collision$"
        ),
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "joint_default_pos": EventTermCfg(
      mode="startup",
      func=dr.joint_default_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (-0.01, 0.01),
        "operation": "add",
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        "operation": "add",
        "ranges": {
          0: (-0.04, 0.04),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "pd_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "operation": "scale",
      },
    ),
  }


def make_g1_smp_terminations() -> dict[str, TerminationTermCfg]:
  """G1 + SMP termination terms."""
  return {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "self_collision": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": G1_SELF_COLLISION.name},
    ),
  }


def make_mini_smp_terminations() -> dict[str, TerminationTermCfg]:
  """Mini_M1v1 + SMP termination terms."""
  return {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "self_collision": TerminationTermCfg(
      func=illegal_contact,
      params={"sensor_name": MINI_SELF_COLLISION.name},
    ),
  }


@dataclass(kw_only=True)
class G1SmpEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for the G1 + SMP guidance environment.

  Rewards are intentionally left empty: each task adds its own
  ``task_smp_product`` term (task reward x SMP guidance).
  """

  scene: G1SmpSceneCfg = field(default_factory=G1SmpSceneCfg)
  observations: dict = field(default_factory=make_g1_smp_observations)
  actions: dict = field(default_factory=make_g1_smp_actions)
  events: dict = field(default_factory=make_g1_smp_events)
  rewards: dict = field(default_factory=dict)
  terminations: dict = field(default_factory=make_g1_smp_terminations)
  viewer: ViewerConfig = field(default_factory=make_smp_viewer)
  sim: SimulationCfg = field(default_factory=make_smp_sim)
  decimation: int = 4
  episode_length_s: float = 20.0


@dataclass(kw_only=True)
class MiniSmpEnvCfg(ManagerBasedRlEnvCfg):
  """Configuration for the Mini_M1v1 + SMP guidance environment.

  Rewards are intentionally left empty: each task adds its own
  ``task_smp_product`` term (task reward x SMP guidance).
  """

  scene: MiniSmpSceneCfg = field(default_factory=MiniSmpSceneCfg)
  observations: dict = field(default_factory=make_mini_smp_observations)
  actions: dict = field(default_factory=make_mini_smp_actions)
  events: dict = field(default_factory=make_mini_smp_events)
  rewards: dict = field(default_factory=dict)
  terminations: dict = field(default_factory=make_mini_smp_terminations)
  viewer: ViewerConfig = field(default_factory=make_smp_viewer)
  sim: SimulationCfg = field(default_factory=make_smp_sim)
  decimation: int = 4
  episode_length_s: float = 20.0


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """Strip training-only events and shrink the SMP buffer for play mode."""
  cfg.episode_length_s = int(1e9)
  cfg.events.pop("push_robot", None)
  cfg.events.pop("push_impulse", None)
  cfg.events.pop("gsi_refresh", None)
  cfg.events["init_smp_state"].params["compile_model"] = False
  cfg.events["init_smp_state"].params["gsi_buffer_size"] = 1024


def g1_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the shared G1 + SMP env cfg (denoiser ckpt path set on
  ``init_smp_state`` below; override it from the task config)."""
  cfg = G1SmpEnvCfg()
  if play:
    _apply_play_overrides(cfg)
  return cfg


def mini_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the shared Mini + SMP env cfg (denoiser ckpt path set on
  ``init_smp_state`` below; override it from the task config)."""
  cfg = MiniSmpEnvCfg()
  if play:
    _apply_play_overrides(cfg)
  return cfg
