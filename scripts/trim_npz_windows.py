"""Trim leading/trailing windows from a windowed motion NPZ file.

Drops ``n_first`` windows from the start and ``n_last`` windows from the end
of the ``windows`` array, then re-saves under a new path. All other arrays
(``fps``, ``window_size``, ``stride``, ``ee_body_names``, ``feature_dims``)
are copied through unchanged.

Usage:
  uv run scripts/trim_npz_windows.py \
    --input-path dataset_mini/cmu/127_23_stageii.npz \
    --output-path dataset_mini/cmu/127_23_stageii_trimmed.npz \
    --n-first 5 --n-last 5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class Cfg:
  input_path: str
  """Path to the source windowed NPZ file."""
  output_path: str
  """Path to write the trimmed NPZ file."""
  n_first: int = 0
  """Number of windows to drop from the start."""
  n_last: int = 0
  """Number of windows to drop from the end."""


def main(cfg: Cfg) -> None:
  in_path = Path(cfg.input_path)
  out_path = Path(cfg.output_path)

  data = np.load(in_path, allow_pickle=True)
  windows = data["windows"]
  num_windows = windows.shape[0]

  if cfg.n_first + cfg.n_last >= num_windows:
    msg = (
      f"n_first ({cfg.n_first}) + n_last ({cfg.n_last}) >= num_windows "
      f"({num_windows}) — nothing would be left"
    )
    raise ValueError(msg)

  end = num_windows - cfg.n_last
  trimmed = windows[cfg.n_first : end]

  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out_path,
    **{k: (trimmed if k == "windows" else data[k]) for k in data.files},
  )

  print(f"Loaded {in_path}: windows={tuple(windows.shape)}")
  print(f"Dropped first {cfg.n_first}, last {cfg.n_last} windows")
  print(f"Saved {out_path}: windows={tuple(trimmed.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
