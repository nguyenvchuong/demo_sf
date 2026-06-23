"""Batch safety metrics for getup / ukemi policies.

Computes quantitative contact-force statistics over many rollouts and writes
a CSV summary. Use this for paper tables; use Viser **Metrics** tab during
``scripts/play.py`` for live numbers.

Example::

    uv run scripts/eval_getup_safety.py Smp-Getup-mini \\
      --checkpoint-file logs/rsl_rl/smp_getup_mini/<run>/model_25500.pt \\
      --num-episodes 50
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

import smp.rl.tasks  # noqa: F401
from smp.rl.tasks.getup_mini.getup_env_cfg import HEAD_STOOD_UP
from smp.rl.tasks.getup_mini.mdp.metrics import (
  ground_contacting_bodies,
  head_height,
  peak_ground_contact_force,
)


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: str
  num_episodes: int = 50
  num_envs: int = 16
  device: str | None = None
  sensor_name: str = "ground_contact_force"
  output: str | None = None
  """CSV path (default: '<checkpoint_dir>/eval_safety.csv')."""
  head_success_height: float = HEAD_STOOD_UP
  max_stand_speed: float = 0.5


def _load_policy(task_id: str, cfg: EvalConfig):
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=False)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  checkpoint_path = Path(cfg.checkpoint_file).resolve()
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)
  return env, policy, checkpoint_path, device


def _episode_done_mask(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
  return terminated | truncated


def run_eval(task_id: str, cfg: EvalConfig) -> Path:
  env, policy, checkpoint_path, device = _load_policy(task_id, cfg)
  unwrapped: ManagerBasedRlEnv = env.unwrapped
  unwrapped.cfg.auto_reset = False
  step_dt = unwrapped.step_dt

  num_envs = cfg.num_envs
  episodes_target = cfg.num_episodes
  episodes_done = 0

  rows: list[dict[str, float | int | bool]] = []

  obs = env.get_observations()

  ep_peak = torch.zeros(num_envs, device=device)
  ep_impulse = torch.zeros(num_envs, device=device)
  ep_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
  ep_max_contacts = torch.zeros(num_envs, device=device)

  robot = unwrapped.scene["robot"]

  try:
    while episodes_done < episodes_target:
      with torch.inference_mode():
        actions = policy(obs)
        obs, _rew, terminated, truncated, _info = env.step(actions)

      done = _episode_done_mask(terminated, truncated)
      peak = peak_ground_contact_force(unwrapped, cfg.sensor_name)
      contacts = ground_contacting_bodies(unwrapped, cfg.sensor_name)
      head_z = head_height(unwrapped)
      speed = torch.linalg.norm(robot.data.root_link_lin_vel_w, dim=-1)

      ep_peak = torch.maximum(ep_peak, peak)
      ep_impulse += peak * step_dt
      ep_steps += 1
      ep_max_contacts = torch.maximum(ep_max_contacts, contacts)

      if not done.any():
        continue

      done_ids = done.nonzero(as_tuple=False).flatten()
      for idx in done_ids.tolist():
        if episodes_done >= episodes_target:
          break
        recovered = bool(
          head_z[idx].item() >= cfg.head_success_height
          and speed[idx].item() < cfg.max_stand_speed
        )
        rows.append(
          {
            "episode": episodes_done,
            "steps": int(ep_steps[idx].item()),
            "peak_contact_force_N": float(ep_peak[idx].item()),
            "contact_impulse_Ns": float(ep_impulse[idx].item()),
            "max_ground_contacting_bodies": int(ep_max_contacts[idx].item()),
            "final_head_height_m": float(head_z[idx].item()),
            "final_speed_mps": float(speed[idx].item()),
            "recovered": recovered,
          }
        )
        episodes_done += 1

      ep_peak[done_ids] = 0.0
      ep_impulse[done_ids] = 0.0
      ep_steps[done_ids] = 0
      ep_max_contacts[done_ids] = 0.0

      unwrapped.reset(env_ids=done_ids)
      obs = env.get_observations()

  finally:
    env.close()

  out_path = (
    Path(cfg.output)
    if cfg.output
    else checkpoint_path.parent / "eval_safety.csv"
  )
  out_path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else []
  with out_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  peaks = np.array([r["peak_contact_force_N"] for r in rows], dtype=np.float64)
  impulses = np.array([r["contact_impulse_Ns"] for r in rows], dtype=np.float64)
  recovered = np.array([r["recovered"] for r in rows], dtype=bool)

  print(f"[INFO] Wrote per-episode metrics -> {out_path}")
  print(f"[INFO] Episodes: {len(rows)}")
  print(
    f"[INFO] peak_contact_force_N: "
    f"mean={peaks.mean():.1f}  max={peaks.max():.1f}  std={peaks.std():.1f}"
  )
  print(
    f"[INFO] contact_impulse_Ns: "
    f"mean={impulses.mean():.1f}  max={impulses.max():.1f}  std={impulses.std():.1f}"
  )
  print(f"[INFO] recovery_rate: {100.0 * recovered.mean():.1f}%")
  return out_path


def main() -> None:
  maybe_print_top_level_help("eval_getup_safety")
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    EvalConfig,
    args=remaining_args,
    config=mjlab.TYRO_FLAGS,
    prog=sys.argv[0] + f" {chosen_task}",
  )
  run_eval(chosen_task, cfg)


if __name__ == "__main__":
  main()
