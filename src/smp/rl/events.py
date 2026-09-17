"""Startup + reset events for SMP RL.

Run from mjlab's event manager so the task stays a plain ``ManagerBasedRlEnv``.
Motion features carry no absolute root pose, so GSI writes a default root frame
(each env's origin, identity yaw) to sim and primes the feature buffer in an
env-origin-relative frame, so the SMP reward is invariant to env placement.
"""

from __future__ import annotations

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from smp.rl.utils import DiffNormalizer, MotionFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import (
  EE_BODY_NAMES,
  NUM_EE,
  rot6d_to_quat,
  slice_features,
)

NUM_JOINTS = 23


def _maybe_compile(model, compile_model: bool, compile_mode: str | None):
  """``torch.compile`` ``model`` (no-op if ``compile_model`` false), working
  around the Inductor ``pad_mm`` TF32 crash by disabling shape padding."""
  if not compile_model:
    return model
  torch.set_float32_matmul_precision("high")
  try:
    import torch._inductor.config as _ic

    _ic.shape_padding = False
  except ImportError:
    pass
  if compile_mode is not None:
    return torch.compile(model, fullgraph=True, mode=compile_mode)
  return torch.compile(model, fullgraph=True)


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
  gsi_buffer_size: int = 4096,
  gsi_batch_size: int = 256,
  compile_model: bool = True,
  compile_mode: str | None = None,
  smp_z_offset: float = 0.0,
) -> None:
  """Startup-mode event: load the frozen denoiser, allocate the feature buffer +
  ``DiffNormalizer`` (stashed on the env), and pre-generate the GSI pool of
  ``gsi_buffer_size`` windows that ``gsi_reset`` samples from (amortizes the DDPM
  cost).  If ``compile_model``, the denoiser is ``torch.compile``-d and pre-warmed
  so Inductor compiles here, not on the first sim step.

  ``smp_z_offset`` raises the GSI-placed robot by this many metres in the SIM
  while keeping the SMP feature world floor at 0 (the reward subtracts it back
  out). Use it when the robot stands on a raised surface (e.g. a platform of
  height ``smp_z_offset``): the whole prior motion then plays out ON TOP of that
  surface instead of inside it, without the prior seeing an off-manifold z."""
  del env_ids
  env._smp_z_offset = float(smp_z_offset)  # type: ignore[attr-defined]
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  model, scheduler, q_low, q_high, feature_dim, window_size = load_denoiser(
    ckpt_path, env.device
  )
  model = _maybe_compile(model, compile_model, compile_mode)
  env._smp_bundle = (  # type: ignore[attr-defined]
    model,
    scheduler,
    q_low,
    q_high,
    feature_dim,
    window_size,
  )
  robot = env.scene["robot"]
  env._smp_ee_indexes = torch.tensor(  # type: ignore[attr-defined]
    robot.find_bodies(list(EE_BODY_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=env.device,
  )
  env._smp_buffer = MotionFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    num_ee=NUM_EE,
    device=env.device,
  )
  env._smp_normalizer = DiffNormalizer(scheduler.num_timesteps, env.device)  # type: ignore[attr-defined]

  if gsi_buffer_size <= 0:
    msg = f"gsi_buffer_size must be positive, got {gsi_buffer_size}."
    raise ValueError(msg)
  pool_chunks: list[torch.Tensor] = []
  for start in range(0, gsi_buffer_size, gsi_batch_size):
    bsz = min(gsi_batch_size, gsi_buffer_size - start)
    pool_chunks.append(_ddpm_sample(env, bsz))
  env._smp_gsi_pool = torch.cat(pool_chunks, dim=0)  # type: ignore[attr-defined]

  if compile_model and env.num_envs != gsi_batch_size:
    # Warm the reward-path shape so its Inductor compile happens here.
    with torch.no_grad():
      dummy_x = torch.randn(env.num_envs, window_size, feature_dim, device=env.device)
      dummy_t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
      _ = model(dummy_x, dummy_t)

  gsi_reset(env)


def _prime_sim_and_buffer(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  window: torch.Tensor,
) -> None:
  """Common GSI tail: write the window's last frame to sim, fill the feature
  buffer.  The buffer is env-origin-RELATIVE (placement-invariant features) while
  the sim write adds each env's origin so robots spread across the grid.
  ``joint_vel`` is finite-differenced from ``joint_pos`` (not in the window)."""
  n, W, _ = window.shape
  E = NUM_EE
  parts = slice_features(window)
  root_pos_local = parts["root_pos"]
  root_rot_6d = parts["root_rot"]
  joint_pos = parts["joint_pos"]
  ee_pos_local = parts["ee_pos"].reshape(n, W, E, 3)
  root_lin_vel_local = parts["root_lin_vel"]
  root_ang_vel_local = parts["root_ang_vel"]

  control_dt = float(env.cfg.sim.mujoco.timestep) * float(env.cfg.decimation)
  if W > 1:
    joint_vel = torch.zeros_like(joint_pos)
    joint_vel[:, :-1] = (joint_pos[:, 1:] - joint_pos[:, :-1]) / control_dt
    joint_vel[:, -1] = joint_vel[:, -2]
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot = env.scene["robot"]
  default_root = robot.data.default_root_state[env_ids].clone()
  default_pos = default_root[:, 0:3]
  default_quat = default_root[:, 3:7]
  yaw_T = yaw_quat(default_quat)
  yaw_T_W = yaw_T[:, None, :].expand(n, W, 4).reshape(-1, 4)

  local_xy = root_pos_local.clone()
  local_xy[..., 2] = 0.0
  world_offset_xy = quat_apply(yaw_T_W, local_xy.reshape(-1, 3)).reshape(n, W, 3)
  pelvis_pos_w = world_offset_xy.clone()
  pelvis_pos_w[..., 0] += default_pos[:, None, 0]
  pelvis_pos_w[..., 1] += default_pos[:, None, 1]
  pelvis_pos_w[..., 2] = root_pos_local[..., 2]

  root_rot_local_quat = rot6d_to_quat(root_rot_6d.reshape(-1, 6)).reshape(n, W, 4)
  pelvis_quat_w = quat_mul(yaw_T_W, root_rot_local_quat.reshape(-1, 4)).reshape(n, W, 4)

  lin_vel_w = quat_apply(yaw_T_W, root_lin_vel_local.reshape(-1, 3)).reshape(n, W, 3)
  ang_vel_w = quat_apply(yaw_T_W, root_ang_vel_local.reshape(-1, 3)).reshape(n, W, 3)

  yaw_T_E = yaw_T[:, None, None, :].expand(n, W, E, 4).reshape(-1, 4)
  ee_offset_w = quat_apply(yaw_T_E, ee_pos_local.reshape(-1, 3)).reshape(n, W, E, 3)
  ee_pos_w = ee_offset_w + pelvis_pos_w[:, :, None, :]

  # Buffer stays env-relative (SMP world, floor=0); the sim write is offset to
  # each env's origin and raised by smp_z_offset so the robot is placed ON a
  # raised surface of that height (the reward subtracts the offset back out).
  origins = env.scene.env_origins[env_ids]
  z_off = getattr(env, "_smp_z_offset", 0.0)
  sim_root_pos = pelvis_pos_w[:, -1] + origins
  if z_off:
    sim_root_pos = sim_root_pos.clone()
    sim_root_pos[:, 2] += z_off
  last_root_state = torch.cat(
    [
      sim_root_pos,
      pelvis_quat_w[:, -1],
      lin_vel_w[:, -1],
      ang_vel_w[:, -1],
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  buf: MotionFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    pelvis_pos_w,
    pelvis_quat_w,
    lin_vel_w,
    ang_vel_w,
    ee_pos_w,
    joint_pos,
    joint_vel,
  )


@torch.no_grad()
def _ddpm_sample(env: ManagerBasedRlEnv, n: int) -> torch.Tensor:
  """Run DDPM ancestral sampling and return ``n`` denormalized windows."""
  model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle  # type: ignore[attr-defined]
  x_t = torch.randn(n, window_size, feature_dim, device=env.device)
  for t_int in reversed(range(scheduler.num_timesteps)):
    t = torch.full((n,), t_int, dtype=torch.long, device=env.device)
    eps = model(x_t, t)
    x_t = scheduler.step(eps, x_t, t_int)
  return (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def gsi_refresh(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  num_samples: int = 1024,
  step_interval: int = 2400,
) -> None:
  """Step-mode event: every ``step_interval`` steps, FIFO-replace ``num_samples``
  GSI-pool windows with fresh DDPM samples so the init distribution stays fresh."""
  del env_ids
  cur = int(env.common_step_counter)
  if cur == 0 or (cur % step_interval) != 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  pool_size = pool.shape[0]
  if num_samples > pool_size:
    msg = f"num_samples ({num_samples}) cannot exceed pool size ({pool_size})"
    raise ValueError(msg)

  new_windows = _ddpm_sample(env, num_samples)
  head = int(getattr(env, "_smp_gsi_head", 0))
  end = head + num_samples
  if end <= pool_size:
    pool[head:end] = new_windows
  else:
    first = pool_size - head
    pool[head:] = new_windows[:first]
    pool[: end - pool_size] = new_windows[first:]
  env._smp_gsi_head = end % pool_size  # type: ignore[attr-defined]


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization: sample ``n`` windows from the GSI pool and
  prime sim + feature buffer from them.  Must run AFTER mjlab's ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  idx = torch.randint(0, pool.shape[0], (n,), device=env.device)
  window = pool[idx]
  _prime_sim_and_buffer(env, env_ids, window)


@torch.no_grad()
def reset_drop_in_air(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  pelvis_height_range: tuple[float, float] = (1.0, 1.8),
  drop_fraction: float = 1.0,
  lateral_speed_range: tuple[float, float] = (0.0, 0.0),
  init_ang_vel_range: tuple[float, float] = (0.0, 0.0),
  init_z_vel_range: tuple[float, float] = (0.0, 0.0),
  upright_spawn: bool = False,
  upright_joint_noise: float = 0.1,
  forward_backward_only: bool = False,
  random_yaw_spawn: bool = False,
  forward_only: bool = False,
) -> None:
  """Reset-mode event: respawn envs FLOATING in the air at a RANDOM height holding a
  RANDOM pose, with optional lateral velocity so the robot falls at an angle.

  Must run AFTER ``gsi_reset`` (it overrides GSI's placement for the chosen envs).
  A ``drop_fraction`` subset is sampled fresh from the GSI pool — giving the random
  joint configuration / orientation (``các dáng ngẫu nhiên``, same source as the
  getup task) — then the window is rewritten so that:
    * the root world-z of EVERY frame is set to a per-env height sampled uniformly
      from ``pelvis_height_range`` (the ``độ cao ngẫu nhiên``), and
    * a random HORIZONTAL velocity is injected in a uniformly-random direction with
      speed ∈ ``lateral_speed_range`` [m/s] — making the robot fall diagonally,
      which gives it forward angular momentum and makes ukemi rolling much easier
      to learn than from a pure vertical drop, and
    * an initial ANGULAR velocity ∈ ``init_ang_vel_range`` [rad/s] is injected
      about the axis perpendicular to the lateral velocity (coordinated so the
      rotation is in the fall direction, pre-loading the shoulder roll), and
    * an initial DOWNWARD velocity ∈ ``init_z_vel_range`` [m/s] is injected on
      the world-z axis (negative = down) — simulating a hard throw rather than a
      passive free-fall, so the robot arrives at the ground with higher impact
      energy and must roll to absorb it, and
    * if ``upright_spawn`` the root orientation is overridden to UPRIGHT (feet
      pointing down) and the joints to the default landing-ready crouch (with
      ±``upright_joint_noise`` rad of noise) — so the robot can land FEET-FIRST
      and then roll, instead of catching the fall on its back,
    * if ``random_yaw_spawn`` (requires ``upright_spawn``) the upright orientation
      is additionally given a uniformly-random yaw angle ∈ [0, 2π] so the robot
      faces a different direction every episode, adding 360° rotational variety
      while keeping the feet-down stance intact,
    * if ``forward_backward_only`` the lateral throw direction is constrained to
      the robot's local forward (+x) or backward (-x) axis chosen randomly, so
      the impact and roll are always along the robot's sagittal plane — the only
      axis a natural ukemi roll can absorb.  Combined with ``random_yaw_spawn``
      this gives full 360° directional variety in world-frame while keeping the
      throw semantically forward/backward in the robot's frame every episode.
  so the primed sim + SMP buffer describe the pose at that height with those
  initial velocities.  Under gravity the robot then free-falls, tilts, and learns
  to roll on touchdown (``update_landed_latch`` flips ``env._drop_landed``).

  Side effects stashed on ``env`` for the reward/latch:
    * ``env._drop_spawn_joint_pos`` — the spawn joint configuration the policy must
      hold while airborne (read by ``hold_initial_pose``).
    * ``env._drop_landed`` — per-env "has touched the ground" latch, reset to False
      here and OR-accumulated each step by ``update_landed_latch``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  robot = env.scene["robot"]
  num_joints = robot.data.joint_pos.shape[-1]
  if not hasattr(env, "_drop_spawn_joint_pos"):
    env._drop_spawn_joint_pos = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, num_joints, device=env.device
    )
  if not hasattr(env, "_drop_landed"):
    env._drop_landed = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
  # The landed latch always resets for the whole reset batch (even envs left on GSI).
  env._drop_landed[env_ids] = False  # type: ignore[attr-defined]

  # Per-env Bernoulli selection so it works for any num_envs (incl. play's 1 env).
  if drop_fraction >= 1.0:
    drop_ids = env_ids
  else:
    drop_ids = env_ids[torch.rand(n, device=env.device) < drop_fraction]
  k = int(drop_ids.numel())
  if k == 0:
    return

  pool: torch.Tensor = env._smp_gsi_pool  # type: ignore[attr-defined]
  idx = torch.randint(0, pool.shape[0], (k,), device=env.device)
  window = pool[idx].clone()  # [k, W, F]

  # root_pos world-z is feature index 2 (see feature_to_state.slice_features).
  lo, hi = pelvis_height_range
  height = lo + (hi - lo) * torch.rand(k, device=env.device)
  window[..., 2] = height[:, None]

  # UPRIGHT SPAWN: override the (possibly inverted) sampled orientation with an
  # upright one so the robot falls feet-down and can land FEET-FIRST. root_rot is
  # the 6D [col0=x-axis, col2=z-axis] of the rotation matrix.
  # With random_yaw_spawn the yaw is drawn uniformly from [0, 2π] so the robot
  # faces a different direction every episode — 6D = [cos θ, sin θ, 0, 0, 0, 1].
  # Without random_yaw_spawn identity (robot faces world +x) is used.
  spawn_yaw = torch.zeros(k, device=env.device)  # default: face world +X
  if upright_spawn:
    J_ = NUM_JOINTS
    if random_yaw_spawn:
      spawn_yaw = 2.0 * math.pi * torch.rand(k, device=env.device)  # [k]
    rot_6d = torch.zeros(k, 6, device=env.device, dtype=window.dtype)
    rot_6d[:, 0] = torch.cos(spawn_yaw)  # col0_x (robot forward, world x-component)
    rot_6d[:, 1] = torch.sin(spawn_yaw)  # col0_y (robot forward, world y-component)
    rot_6d[:, 5] = 1.0                    # col2_z = 1 (up axis unchanged)
    window[..., 3:9] = rot_6d[:, None, :]
    default_jp = robot.data.default_joint_pos[drop_ids]  # [k, J]
    if upright_joint_noise > 0.0:
      noise = (2.0 * torch.rand_like(default_jp) - 1.0) * upright_joint_noise
      default_jp = default_jp + noise
    window[..., 9 : 9 + J_] = default_jp[:, None, :]

  # Zero the root linear+angular velocity features, then inject lateral + angular
  # velocity so the robot falls diagonally and pre-rotates toward a shoulder roll.
  # Feature layout (feature_to_state.slice_features):
  #   [0:9]  root_pos(3) + root_rot_6d(6)
  #   [9:32] joint_pos (J=23)
  #   [32:47] ee_pos (E=5 × 3)
  #   [47:50] root_lin_vel (x, y, z)  ← lin_start = 9+J+E*3 = 47
  #   [50:53] root_ang_vel (x, y, z)
  J, E = NUM_JOINTS, NUM_EE
  lin_start = 9 + J + E * 3  # = 47
  window[..., lin_start : lin_start + 6] = 0.0

  # LATERAL VELOCITY: throw direction in world XY.
  # forward_backward_only=True → throw along ±robot-forward (spawn_yaw or spawn_yaw+π)
  #   so impact and roll are always in the robot's sagittal plane — the only axis a
  #   ukemi roll can absorb.  Combined with random_yaw_spawn this gives full 360°
  #   world-frame variety while keeping the throw forward/backward in robot frame.
  # forward_backward_only=False → uniformly random direction (original behaviour).
  lo_v, hi_v = lateral_speed_range
  if hi_v > 0.0:
    speed = lo_v + (hi_v - lo_v) * torch.rand(k, device=env.device)
    if forward_backward_only:
      if forward_only:
        # Always throw forward (along spawn_yaw, never backward).
        angle = spawn_yaw.clone()
      else:
        # Each env randomly picks forward (0) or backward (π) relative to spawn_yaw.
        flip = torch.randint(0, 2, (k,), device=env.device).float() * math.pi
        angle = spawn_yaw + flip
    else:
      angle = 2.0 * math.pi * torch.rand(k, device=env.device)
    vx = speed * torch.cos(angle)   # world x
    vy = speed * torch.sin(angle)   # world y
    window[:, :, lin_start] = vx[:, None]
    window[:, :, lin_start + 1] = vy[:, None]

    # INITIAL ANGULAR VELOCITY: ∈ init_ang_vel_range rad/s about the axis
    # perpendicular to the lateral velocity → rotates in the fall direction,
    # pre-loading the shoulder roll before touchdown.
    # Perpendicular to (cos θ, sin θ) in world xy = (-sin θ, cos θ, 0).
    lo_w, hi_w = init_ang_vel_range
    if hi_w > 0.0:
      ang_speed = lo_w + (hi_w - lo_w) * torch.rand(k, device=env.device)
      window[:, :, lin_start + 3] = (ang_speed * (-torch.sin(angle)))[:, None]
      window[:, :, lin_start + 4] = (ang_speed * torch.cos(angle))[:, None]

  # DOWNWARD VELOCITY: ∈ init_z_vel_range m/s, always negative (downward).
  # Independent of lateral_speed so it can be set even with no horizontal throw.
  lo_z, hi_z = init_z_vel_range
  if hi_z > 0.0:
    z_speed = lo_z + (hi_z - lo_z) * torch.rand(k, device=env.device)
    window[:, :, lin_start + 2] = -z_speed[:, None]

  _prime_sim_and_buffer(env, drop_ids, window)

  # Record the pose just written to sim as the target the policy must hold midair.
  env._drop_spawn_joint_pos[drop_ids] = robot.data.joint_pos[drop_ids].clone()  # type: ignore[attr-defined]


@torch.no_grad()
def update_landed_latch(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  sensor_name: str = "ground_contact_force",
  force_threshold: float = 5.0,
) -> None:
  """Step-mode event: detect the ground touchdown and LATCH it per env.

  Reads the robot-vs-terrain contact-force sensor and flips ``env._drop_landed``
  to True (sticky) once the peak contact force exceeds ``force_threshold`` newtons.
  This is the "phát hiện chạm đất" signal that switches the reward from the airborne
  hold-pose phase to the ukemi roll + getup phase. Using the contact sensor is more
  robust than inferring contact from joint deltas (a PD-tracked joint barely moves
  on a soft touch), and the env already carries the sensor for ``rolling_contact_force``.
  """
  del env_ids
  if not hasattr(env, "_drop_landed"):
    env._drop_landed = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.bool, device=env.device
    )
  sensor = env.scene.sensors[sensor_name]
  peak_force = torch.norm(sensor.data.force, dim=-1).max(dim=-1)[0]  # [B]
  env._drop_landed |= peak_force > force_threshold  # type: ignore[attr-defined]


@torch.no_grad()
def reset_stand_on_platform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  stand_fraction: float = 0.5,
  pelvis_height: float = 0.77,
  ee_offsets: tuple[tuple[float, float, float], ...] | None = None,
  top_geom_name: str = "stair_top_collision",
  top_geom_half_height: float = 0.30,
  spawn_x_offset: float = 0.0,
) -> None:
  """Reset-mode event: place a ``stand_fraction`` subset of envs STABLY standing on
  the (randomized) top tread, so the episode begins from a clean platform stance and
  the diffusion reward drives the jump-off — instead of GSI seeding every env in a
  random motion phase (mid-air / falling), which wobbles on the narrow tread.

  Must run AFTER ``gsi_reset`` (it overrides GSI for the chosen envs) and AFTER the
  ``randomize_platform_height`` event (it reads each env's randomized tread height so
  the feet rest exactly on the surface). The stance is the robot's default keyframe
  (``KNEES_BENT``, all-zero joints) with ZERO root/joint velocity, pelvis at
  ``tread_surface + pelvis_height``. The SMP feature buffer is re-primed with a static
  standing window (env-origin-relative, floor at 0) so the SMP reward is consistent
  from step 0. ``ee_offsets`` are the end-effector positions relative to the pelvis at
  this stance (precomputed by FK; see ``jump_platform_env_cfg``). ``spawn_x_offset``
  shifts the stance along the robot's facing (+x) in the env-local frame: use a
  NEGATIVE value to set the robot BACK from the drop edge so it has full foot
  support and stands stably (rather than teetering at the lip of the platform)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0 or stand_fraction <= 0.0:
    return

  # Per-env Bernoulli selection so it works for any num_envs (incl. play's 1 env).
  stand_ids = env_ids[torch.rand(n, device=env.device) < stand_fraction]
  k = int(stand_ids.numel())
  if k == 0:
    return

  robot = env.scene["robot"]

  # Global geom index of the top tread (cache across resets).
  gidx = getattr(env, "_top_stair_gidx", None)
  if gidx is None:
    gnames = list(robot.geom_names)
    gidx = int(robot.indexing.geom_ids[gnames.index(top_geom_name)].item())
    env._top_stair_gidx = gidx  # type: ignore[attr-defined]

  # Per-env tread surface height (local to env origin), from the randomized geom.
  surface = env.sim.model.geom_pos[stand_ids, gidx, 2] + top_geom_half_height
  origins = env.scene.env_origins[stand_ids]

  # --- sim state: stable upright stance, feet on the tread, zero velocity --------
  jp = robot.data.default_joint_pos[stand_ids].clone()  # KNEES_BENT (all-zero) stance
  jv = torch.zeros_like(jp)
  pos = origins.clone()
  pos[:, 0] = origins[:, 0] + spawn_x_offset
  pos[:, 2] = origins[:, 2] + surface + pelvis_height
  quat = torch.zeros(k, 4, device=env.device)
  quat[:, 0] = 1.0  # identity: upright, facing +x (the prior's jump direction)
  root_state = torch.cat([pos, quat, torch.zeros(k, 6, device=env.device)], dim=-1)
  robot.write_root_state_to_sim(root_state, env_ids=stand_ids)
  robot.write_joint_state_to_sim(jp, jv, env_ids=stand_ids)

  # --- re-prime the SMP buffer with a static standing window ----------------------
  buf = getattr(env, "_smp_buffer", None)
  if buf is None:
    return
  W = buf.window_size
  z_off = getattr(env, "_smp_z_offset", 0.0)
  # Buffer frame is env-origin-relative with floor at 0 (matches _update_buffer_from_sim).
  root_pos_rel = torch.zeros(k, 3, device=env.device)
  root_pos_rel[:, 0] = spawn_x_offset
  root_pos_rel[:, 2] = surface + pelvis_height - z_off
  root_pos_win = root_pos_rel[:, None, :].expand(k, W, 3).contiguous()
  quat_win = torch.zeros(k, W, 4, device=env.device)
  quat_win[..., 0] = 1.0
  zero3 = torch.zeros(k, W, 3, device=env.device)
  jp_win = jp[:, None, :].expand(k, W, jp.shape[-1]).contiguous()
  jv_win = torch.zeros_like(jp_win)
  if ee_offsets is not None:
    ee_off = torch.tensor(ee_offsets, device=env.device, dtype=root_pos_win.dtype)
    ee_pos_win = root_pos_win[:, :, None, :] + ee_off[None, None, :, :]
  else:
    ee_pos_win = root_pos_win[:, :, None, :].expand(k, W, NUM_EE, 3).contiguous()
  buf.reset(
    stand_ids,
    root_pos_win,
    quat_win,
    zero3,
    zero3,
    ee_pos_win,
    jp_win,
    jv_win,
  )
