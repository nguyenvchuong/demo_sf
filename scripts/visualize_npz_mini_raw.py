"""Visualize a raw Mini M1V1 motion NPZ file in a viser viewer.

Unlike ``visualize_npz_mini.py`` (which plays back *windowed feature* NPZ
files produced by ``csv_to_npz_mini.py``), this script plays back *raw*
motion NPZ files such as ``dataset_mini/chuong_data/ori/**/*.npz``, which
store per-frame ``base_pos_w`` (T,3), ``base_quat_w`` (T,4, wxyz),
``joint_pos`` (T,J), and ``joint_names`` (J,) directly.

Usage:
  uv run scripts/visualize_npz_mini_raw.py --npz-path dataset_mini/chuong_data/ori/231110/jump_form_box_to_safety_roll_180_R_001__A500_M.npz
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

from smp.utils import detect_device


@dataclass
class Cfg:
  npz_path: str = ""
  """Path to a raw motion NPZ file (base_pos_w/base_quat_w/joint_pos/joint_names)."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 0.0
  """Playback frame rate. 0 = use the FPS stored in the NPZ file."""


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


def main(cfg: Cfg) -> None:
  if not cfg.npz_path:
    msg = "Provide --npz-path pointing to a raw motion NPZ file"
    raise ValueError(msg)

  npz_path = Path(cfg.npz_path)
  if not npz_path.exists():
    msg = f"File not found: {npz_path}"
    raise FileNotFoundError(msg)

  data = np.load(npz_path, allow_pickle=True)
  base_pos = data["base_pos_w"]  # (T, 3)
  base_quat = data["base_quat_w"]  # (T, 4) wxyz
  joint_pos = data["joint_pos"]  # (T, J)
  joint_names = list(data["joint_names"])
  num_frames = base_pos.shape[0]
  fps = float(data["fps"]) if (cfg.fps <= 0 and "fps" in data) else (cfg.fps or 30.0)
  print(
    f"Loaded {npz_path.name}: {num_frames} frames, {len(joint_names)} joints, fps={fps}"
  )

  device_str = cfg.device or detect_device()
  device = torch.device(device_str)

  sim, scene = _setup_mini_sim(device_str)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  joint_indexes = torch.tensor(
    robot.find_joints(joint_names, preserve_order=True)[0],
    dtype=torch.long,
    device=device_str,
  )

  base_pos_t = torch.as_tensor(base_pos, device=device, dtype=torch.float32)
  base_quat_t = torch.as_tensor(base_quat, device=device, dtype=torch.float32)
  joint_pos_t = torch.as_tensor(joint_pos, device=device, dtype=torch.float32)

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  with server.gui.add_folder("Dataset"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=num_frames - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    info_text = server.gui.add_markdown(
      f"**File:** {npz_path.name}  \n**Frames:** {num_frames}  \n**FPS:** {fps}"
    )

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  def render(frame: int) -> None:
    root = robot.data.default_root_state.clone()
    root[:, 0:3] = base_pos_t[frame]
    root[:, 3:7] = base_quat_t[frame]
    robot.write_root_state_to_sim(root)

    jp = robot.data.default_joint_pos.clone()
    jv = robot.data.default_joint_vel.clone()
    jp[:, joint_indexes] = joint_pos_t[frame]
    robot.write_joint_state_to_sim(jp, jv)

    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt_play = 1.0 / fps
  try:
    while True:
      frame = int(frame_slider.value)
      render(frame)
      if playing["v"]:
        nxt_frame = (frame + 1) % num_frames
        frame_slider.value = nxt_frame
      time.sleep(dt_play)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
