"""Play wrapper: registers SMP tasks then delegates to mjlab.scripts.play.main.

Also doubles as a reference for converting a trained `.pt` checkpoint to
ONNX for deployment, e.g.:

    uv run scripts/play.py --to-onnx \
      --task Smp-Forward-mini \
      --checkpoint-file logs/rsl_rl/smp_forward_mini/<run>/model_2000.pt

Writes `<checkpoint_dir>/exported/<checkpoint_stem>.onnx` (override with
`--output`).
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
import mjlab.scripts.play as mjlab_play
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry
from smp.rl.viewer import SmpViserPlayViewer


@dataclass(frozen=True)
class OnnxExportConfig:
  task: str
  """Registered task id, e.g. 'Smp-Forward-mini'."""
  checkpoint_file: str
  """Path to the `.pt` checkpoint, e.g. logs/rsl_rl/.../model_2000.pt."""
  output: str | None = None
  """Output directory for the .onnx file (default: '<checkpoint_dir>/exported')."""


def export_onnx(cfg: OnnxExportConfig) -> None:
  """Load a trained checkpoint and export the actor to ONNX with metadata.

  Mirrors what `VelocityOnPolicyRunner` / `MotionTrackingOnPolicyRunner` do
  automatically on `save()` during training, as a standalone post-hoc
  conversion. SMP tasks use the base `MjlabOnPolicyRunner`, which has no
  auto-export, so this is the way to get a deployable `policy.onnx`.
  """
  configure_torch_backends()
  device = "cpu"

  checkpoint_path = Path(cfg.checkpoint_file).resolve()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  env_cfg = load_env_cfg(cfg.task, play=True)
  agent_cfg = load_rl_cfg(cfg.task)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
  )

  export_dir = Path(cfg.output) if cfg.output else checkpoint_path.parent / "exported"
  filename = f"{checkpoint_path.stem}.onnx"
  runner.export_policy_to_onnx(str(export_dir), filename)

  onnx_path = export_dir / filename
  metadata = get_base_metadata(env.unwrapped, run_path=checkpoint_path.stem)
  attach_metadata_to_onnx(str(onnx_path), metadata)

  print(f"[INFO] Exported ONNX policy -> {onnx_path}")
  env.close()


if __name__ == "__main__":
  if "--to-onnx" in sys.argv:
    sys.argv.remove("--to-onnx")
    export_onnx(tyro.cli(OnnxExportConfig, config=mjlab.TYRO_FLAGS))
  else:
    # Use native MuJoCo contact-force decor in Viser (not simple debug arrows).
    mjlab_play.ViserPlayViewer = SmpViserPlayViewer
    mjlab_play.main()
