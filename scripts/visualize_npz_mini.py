"""Visualize a Mini M1V1 dataset NPZ file in a viser viewer.

Loads a windowed NPZ file (produced by csv_to_npz_mini.py) and lets you
interactively browse windows and scrub frames.  Each window is rendered
with the default robot standing pose as the world-frame anchor for the
last window frame.

Usage:
  uv run scripts/visualize_npz_mini.py --npz-path dataset_mini/npz/walk1_subject1.npz
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.sampling.feature_to_state import (
  NUM_EE,
  window_to_ee_trajectories,
  window_to_pelvis_trajectory,
)
from smp.utils import detect_device


@dataclass
class Cfg:
  npz_path: str = ""
  """Path to a windowed NPZ file from dataset_mini/npz/."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _setup_mini_sim(device: str):
  from mjlab.scene import Scene
  from mjlab.scene.scene import SceneCfg
  from mjlab.sim.sim import Simulation, SimulationCfg

  from smp.robot.Mini_M1v1.mini_m11_constants import get_mini_m1v1_robot_cfg

  scene_cfg = SceneCfg(entities={"robot": get_mini_m1v1_robot_cfg()})
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _decode_window(
  window: torch.Tensor,
  anchor_pos: torch.Tensor,
  anchor_quat: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return (pelvis_pos, pelvis_quat, joint_pos, ee_pos) numpy arrays for one window."""
  p_pos, p_quat, p_joint = window_to_pelvis_trajectory(window, anchor_pos, anchor_quat)
  ee_pos = window_to_ee_trajectories(window, p_pos, p_quat)
  return (
    p_pos.cpu().numpy(),
    p_quat.cpu().numpy(),
    p_joint.cpu().numpy(),
    ee_pos.cpu().numpy(),
  )


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  if not cfg.npz_path:
    msg = "Provide --npz-path pointing to a dataset_mini/npz/*.npz file"
    raise ValueError(msg)

  npz_path = Path(cfg.npz_path)
  if not npz_path.exists():
    msg = f"File not found: {npz_path}"
    raise FileNotFoundError(msg)

  data = np.load(npz_path)
  windows_np: np.ndarray = data["windows"]  # (N, W, F)
  num_windows, window_size, feature_dim = windows_np.shape
  fps = float(data["fps"][0]) if "fps" in data else cfg.fps
  print(f"Loaded {npz_path.name}: {num_windows} windows, W={window_size}, F={feature_dim}, fps={fps}")

  device_str = cfg.device or detect_device()
  device = torch.device(device_str)

  sim, scene = _setup_mini_sim(device_str)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  anchor_pos = robot.data.default_root_state[0, 0:3].detach().cpu()
  anchor_quat = robot.data.default_root_state[0, 3:7].detach().cpu()

  # Cache decoded windows on demand.
  cache: dict[int, tuple] = {}

  def get_decoded(win_idx: int):
    if win_idx not in cache:
      window = torch.from_numpy(windows_np[win_idx]).float()
      cache[win_idx] = _decode_window(window, anchor_pos, anchor_quat)
    return cache[win_idx]

  state: dict = {"win": 0}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  ee_points = server.scene.add_point_cloud(
    name="/fixed_bodies/ee_positions",
    points=np.zeros((NUM_EE, 3), dtype=np.float32),
    colors=np.tile(np.array([255, 80, 0], dtype=np.uint8), (NUM_EE, 1)),
    point_size=0.03,
  )

  with server.gui.add_folder("Dataset"):
    win_slider = server.gui.add_slider(
      "Window", min=0, max=num_windows - 1, step=1, initial_value=0
    )
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    info_text = server.gui.add_markdown(
      f"**File:** {npz_path.name}  \n**Windows:** {num_windows}  \n**FPS:** {fps}"
    )

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @win_slider.on_update
  def _(_evt) -> None:
    state["win"] = int(win_slider.value)
    frame_slider.value = 0

  def render(win_idx: int, frame: int) -> None:
    p_pos, p_quat, p_joint, ee_pos = get_decoded(win_idx)
    _write_pose_to_robot(robot, p_pos[frame], p_quat[frame], p_joint[frame], device_str)
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    ee_points.points = ee_pos[frame]
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt_play = 1.0 / cfg.fps
  try:
    while True:
      win_idx = state["win"]
      frame = int(frame_slider.value)
      render(win_idx, frame)
      if playing["v"]:
        nxt_frame = frame + 1
        if nxt_frame >= window_size:
          nxt_frame = 0
          next_win = (win_idx + 1) % num_windows
          win_slider.value = next_win
          state["win"] = next_win
        frame_slider.value = nxt_frame
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
