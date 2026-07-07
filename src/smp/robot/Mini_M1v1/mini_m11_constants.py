"""M23 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##
#Mini_M1v1.xml: 23dof
MINI_M1V1_XML: Path = Path(
  "src/smp/robot/Mini_M1v1/Mini_M1v1.xml"
)
assert MINI_M1V1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(MINI_M1V1_XML))


##
# Actuator config.
##

# Motor specs (from Unitree).
ARMATURE_EC_A8116_P1_18H = 0.063828       # hip_pitch, knee
# ARMATURE_EC_A6416_P2_30_25H = 0.093956787  # hip_roll
ARMATURE_EC_A6416_P2_30_25H = 0.063828   # hip_roll
ARMATURE_EC_A6408_P2_30_25H = 0.057648938    # hip_yaw, waist
ARMATURE_EC_A4310_P2_36H = 0.023328       # shoulder, elbow, wrist, ankle

# ACTUATOR_EC04 = ElectricActuator(
#   reflected_inertia=ARMATURE_EC04,
#   velocity_limit=37.0,
#   effort_limit=25.0,
# )
# ACTUATOR_EC03 = ElectricActuator(
#   reflected_inertia=ARMATURE_EC03,
#   velocity_limit=32.0,
#   effort_limit=88.0,
# )
# ACTUATOR_EC06 = ElectricActuator(
#   reflected_inertia=ARMATURE_EC06,
#   velocity_limit=20.0,
#   effort_limit=36.0,
# )
# ACTUATOR_EC00 = ElectricActuator(
#   reflected_inertia=ARMATURE_EC00,
#   velocity_limit=22.0,
#   effort_limit=5.0,
# )
# ACTUATOR_EC05 = ElectricActuator(
#   reflected_inertia=ARMATURE_EC05,
#   velocity_limit=22.0,
#   effort_limit=5.0,
# )
NATURAL_FREQ = 4.0 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 1.2
 
STIFFNESS_EC_A8116_P1_18H = ARMATURE_EC_A8116_P1_18H * NATURAL_FREQ**2
STIFFNESS_EC_A6416_P2_30_25H = ARMATURE_EC_A6416_P2_30_25H * NATURAL_FREQ**2
STIFFNESS_EC_A6408_P2_30_25H = ARMATURE_EC_A6408_P2_30_25H * NATURAL_FREQ**2
STIFFNESS_EC_A4310_P2_36H = ARMATURE_EC_A4310_P2_36H * NATURAL_FREQ**2
 
DAMPING_EC_A8116_P1_18H = 2.0 * DAMPING_RATIO * ARMATURE_EC_A8116_P1_18H * NATURAL_FREQ
DAMPING_EC_A6416_P2_30_25H = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6416_P2_30_25H * NATURAL_FREQ
DAMPING_EC_A6408_P2_30_25H = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6408_P2_30_25H * NATURAL_FREQ
DAMPING_EC_A4310_P2_36H = 2.0 * DAMPING_RATIO * ARMATURE_EC_A4310_P2_36H * NATURAL_FREQ

# DC motor torque-speed curve parameters.
# velocity_limit   : no-load joint-level speed [rad/s]  – torque drops to 0 here.
# saturation_effort: stall (peak) torque at zero speed [Nm] – must be >= effort_limit.
# effort_limit     : continuous torque limit [Nm] kept same as before.
#
# Derived from G1/Unitree public proxies and peer motor families (no public datasheet
# exists for these exact Encos IDs). Update when hardware bench data is available.
VELOCITY_LIMIT_EC_A4310_P2_36H  = 18.5   # rad/s – small arm/ankle motor
VELOCITY_LIMIT_ANKLE_PITCH  = 10.0   # rad/s – small arm/ankle motor
VELOCITY_LIMIT_ANKLE_ROLL  = 15.0   # rad/s – small arm/ankle motor
VELOCITY_LIMIT_EC_A8116_P1_18H  = 13.0   # rad/s – large hip-pitch / knee motor
VELOCITY_LIMIT_EC_A6416_P2_30_25H = 13.0 # rad/s – hip-roll motor
VELOCITY_LIMIT_EC_A6408_P2_30_25H = 10.0 # rad/s – hip-yaw / waist motor

SATURATION_EFFORT_EC_A4310_P2_36H   =  30.0  # Nm – equal to effort_limit (conservative)
SATURATION_EFFORT_EC_ANKLE_PITCH   =  24.0  # Nm – equal to effort_limit (conservative)
SATURATION_EFFORT_EC_ANKLE_ROLL   =  36.0  # Nm – equal to effort_limit (conservative)
SATURATION_EFFORT_EC_A8116_P1_18H   = 125.0  # Nm – ~1.23x continuous (mild peak boost)
SATURATION_EFFORT_EC_A6416_P2_30_25H = 118.0  # Nm – ~1.21x continuous
SATURATION_EFFORT_EC_A6408_P2_30_25H =  60.0  # Nm – ~1.14x continuous

#uncomment for 27dof
# MINI_M1V1_ACTUATOR_EC00 = BuiltinPositionActuatorCfg(
#   target_names_expr=(
#     ".*_wrist_yaw_joint",
#   ),
#   stiffness=STIFFNESS_EC00,
#   damping=DAMPING_EC00,
#   effort_limit=8.0,
#   armature=ARMATURE_EC00,
# )

# Actuator command latency (sim2real), modeling comm + motor response delay
# between the policy's position command and the motor acting on it. Mirrors
# vm_lab_rr, which models all latency at the actuator/position level (see
# m2v3.py DelayedInstinctActuatorCfg + sync_actuator_delays, lag_range=(1,3)).
#
# Lag is in PHYSICS steps (5 ms each), sampled in [0, 3] = 0-15 ms. Unlike the
# previous config (which resampled the lag EVERY physics step -> unrealistic
# 200 Hz jitter that the policy just learns to ignore), the lag here is HELD per
# robot and drifts slowly: an env only reconsiders its lag every
# delay_update_period steps, and even then keeps the old value with probability
# delay_hold_prob. With period=50 (0.25 s) and hold_prob=0.9 the effective
# latency stays roughly constant over multi-second windows -- a faithful stand-in
# for a real robot's near-constant comm/motor dead-time, which is what actually
# makes an instant-feedback policy react sluggishly on hardware. per_env_phase
# staggers the (rare) updates across envs so they don't change in lockstep.
_ACTUATOR_DELAY = dict(
  delay_min_lag=0,
  delay_max_lag=3,
  delay_update_period=50,
  delay_hold_prob=0.9,
  delay_per_env_phase=True,
)


MINI_M1V1_ACTUATOR_EC_A4310_P2_36H = DcMotorActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=STIFFNESS_EC_A4310_P2_36H,
  damping=DAMPING_EC_A4310_P2_36H,
  effort_limit=36.0,
  saturation_effort=SATURATION_EFFORT_EC_A4310_P2_36H,
  velocity_limit=VELOCITY_LIMIT_EC_A4310_P2_36H,
  armature=ARMATURE_EC_A4310_P2_36H,
  **_ACTUATOR_DELAY,
)


MINI_M1V1_ACTUATOR_EC_A8116_P1_18H = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint", ".*_knee_joint"),
  stiffness=STIFFNESS_EC_A8116_P1_18H,
  damping=DAMPING_EC_A8116_P1_18H,
  effort_limit=130.0,
  saturation_effort=SATURATION_EFFORT_EC_A8116_P1_18H,
  velocity_limit=VELOCITY_LIMIT_EC_A8116_P1_18H,
  armature=ARMATURE_EC_A8116_P1_18H,
  **_ACTUATOR_DELAY,
)

MINI_M1V1_ACTUATOR_EC_A6416_P2_30_25H = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_roll_joint",),
  stiffness=STIFFNESS_EC_A6416_P2_30_25H,
  damping=DAMPING_EC_A6416_P2_30_25H,
  effort_limit=132.0,
  saturation_effort=SATURATION_EFFORT_EC_A6416_P2_30_25H,
  velocity_limit=VELOCITY_LIMIT_EC_A6416_P2_30_25H,
  armature=ARMATURE_EC_A6416_P2_30_25H,
  **_ACTUATOR_DELAY,
)

MINI_M1V1_ACTUATOR_EC_A6408_P2_30_25H = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", "waist_joint"),
  stiffness=STIFFNESS_EC_A6408_P2_30_25H,
  damping=DAMPING_EC_A6408_P2_30_25H,
  effort_limit=70.0,
  saturation_effort=SATURATION_EFFORT_EC_A6408_P2_30_25H,
  velocity_limit=VELOCITY_LIMIT_EC_A6408_P2_30_25H,
  armature=ARMATURE_EC_A6408_P2_30_25H,
  **_ACTUATOR_DELAY,
)

# M23_ACTUATOR_EC05 = BuiltinPositionActuatorCfg(
#   target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_roll_joint"),
#   stiffness=STIFFNESS_EC05,
#   damping=DAMPING_EC05,
#   effort_limit=5.0,
#   armature=ARMATURE_EC05,
# )

# Waist pitch/roll and ankles are 4-bar linkages with 2 5020 actuatoEC.
# Due to the parallel linkage, the effective armature at the ankle and waist joints
# is configuration dependent. Since the exact geometry of the linkage is unknown, we
# assume a nominal 1:1 gear ratio. Under this assumption, the joint armature in the
# nominal configuration is approximated as the sum of the 2 actuatoEC' armatures.

# MINI_M1V1_ACTUATOR_EC_ANKLE = DcMotorActuatorCfg(
#   target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
#   stiffness=STIFFNESS_EC_A4310_P2_36H * 2.0,
#   damping=DAMPING_EC_A4310_P2_36H * 2.0,
#   effort_limit=36.0 * 2.0,
#   # Ankle: 2 A4310 motors via 4-bar linkage, sum saturation effort.
#   saturation_effort=SATURATION_EFFORT_EC_A4310_P2_36H * 2.0,
#   velocity_limit=VELOCITY_LIMIT_EC_A4310_P2_36H,
#   armature=ARMATURE_EC_A4310_P2_36H * 2.0,
# )

MINI_M1V1_ACTUATOR_EC_ANKLE_PITCH = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=STIFFNESS_EC_A4310_P2_36H ,
  damping=DAMPING_EC_A4310_P2_36H * 2.0,
  effort_limit=24.0,
  # Ankle: 2 A4310 motors via 4-bar linkage, sum saturation effort.
  saturation_effort=SATURATION_EFFORT_EC_ANKLE_PITCH ,
  velocity_limit=VELOCITY_LIMIT_ANKLE_PITCH,
  armature=ARMATURE_EC_A4310_P2_36H,
  **_ACTUATOR_DELAY,
)

MINI_M1V1_ACTUATOR_EC_ANKLE_ROLL = DcMotorActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=STIFFNESS_EC_A4310_P2_36H ,
  damping=DAMPING_EC_A4310_P2_36H * 2.0,
  effort_limit=36.0,
  # Ankle: 2 A4310 motors via 4-bar linkage, sum saturation effort.
  saturation_effort=SATURATION_EFFORT_EC_ANKLE_ROLL ,
  velocity_limit=VELOCITY_LIMIT_ANKLE_ROLL,
  armature=ARMATURE_EC_A4310_P2_36H ,
  **_ACTUATOR_DELAY,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.7),
  joint_pos={
    ".*_hip_pitch_joint": -0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_pitch_joint": -0.0,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.77),
  joint_pos={
    ".*_hip_pitch_joint": -0.0,
    ".*_knee_joint": 0.0,
    ".*_ankle_pitch_joint": -0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

MINI_M1V1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MINI_M1V1_ACTUATOR_EC_A4310_P2_36H,
    MINI_M1V1_ACTUATOR_EC_A8116_P1_18H,
    MINI_M1V1_ACTUATOR_EC_A6416_P2_30_25H,
    MINI_M1V1_ACTUATOR_EC_A6408_P2_30_25H,
    MINI_M1V1_ACTUATOR_EC_ANKLE_PITCH,
    MINI_M1V1_ACTUATOR_EC_ANKLE_ROLL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_mini_m1v1_robot_cfg() -> EntityCfg:
  """Get a fresh Mini M1V1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MINI_M1V1_ARTICULATION,
  )


MINI_M1V1_ACTION_SCALE: dict[str, float] = {}
for a in MINI_M1V1_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    MINI_M1V1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_mini_m1v1_robot_cfg())

  viewer.launch(robot.spec.compile())
