"""Open scene_Mini_M1v1_stair_jump.xml with the robot pre-placed on the
platform (the XML's `on_platform_perpendicular` keyframe), instead of the
default pose which sinks into the platform.

Usage:
  uv run scripts/view_stair_jump.py
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import mujoco.viewer

_XML = (
  Path(__file__).resolve().parent.parent
  / "src/smp/robot/Mini_M1v1/scene_Mini_M1v1_stair_jump.xml"
)


def main() -> None:
  model = mujoco.MjModel.from_xml_path(str(_XML))
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  mujoco.viewer.launch(model, data)


if __name__ == "__main__":
  main()
