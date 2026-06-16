"""Convert NPZ motion files (produced by csv_to_npz.py) back to CSV format.

Output CSV columns:
  [base_pos_x, base_pos_y, base_pos_z,
   base_rot_x, base_rot_y, base_rot_z, base_rot_w,   ← xyzw (matching original)
   dof_pos_joint0, ..., dof_pos_jointN]
"""

from pathlib import Path

import numpy as np
import tyro


_DEFAULT_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
]


def _npz_to_csv(npz_path: str | Path, csv_path: str | Path, joint_names: list[str]) -> None:
    data = np.load(npz_path, allow_pickle=True)

    all_joint_names: list[str] = list(data["joint_names"])

    # joint_pos has one column per actuated joint (floating_base_joint excluded).
    actuated_names = [n for n in all_joint_names if n != "floating_base_joint"]
    name_to_col = {n: i for i, n in enumerate(actuated_names)}
    try:
        joint_idxs = [name_to_col[j] for j in joint_names]
    except KeyError as e:
        raise ValueError(
            f"Joint not found in NPZ: {e}. Available joints: {actuated_names}"
        ) from e

    base_pos = data["base_pos_w"]          # (T, 3)
    base_quat_wxyz = data["base_quat_w"]   # (T, 4) wxyz

    # Convert wxyz → xyzw to match original CSV convention.
    base_quat_xyzw = base_quat_wxyz[:, [1, 2, 3, 0]]  # (T, 4)

    dof_pos = data["joint_pos"][:, joint_idxs]  # (T, n_joints)

    rows = np.concatenate([base_pos, base_quat_xyzw, dof_pos], axis=1)

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csv_path, rows, delimiter=",")
    print(f"[INFO] Saved {rows.shape[0]} frames × {rows.shape[1]} cols → {csv_path}")


def main(
    input_file: str,
    output_file: str,
) -> None:
    """Convert a single NPZ motion file to CSV.

    Args:
        input_file: Path to the input .npz file.
        output_file: Path for the output .csv file.
    """
    _npz_to_csv(input_file, output_file, _DEFAULT_JOINT_NAMES)


def main_dir(
    input_dir: str,
    output_dir: str = "csv_output",
    pattern: str = "*.npz",
    skip_existing: bool = True,
) -> None:
    """Batch-convert a folder of NPZ files to CSV.

    Args:
        input_dir: Directory containing .npz files.
        output_dir: Directory for output .csv files.
        pattern: Glob pattern for selecting files (default: ``*.npz``).
        skip_existing: Skip files whose output CSV already exists.
    """
    npz_files = sorted(Path(input_dir).glob(pattern))
    if not npz_files:
        raise FileNotFoundError(f"No files matching '{pattern}' in: {input_dir}")

    print(f"[INFO] Found {len(npz_files)} file(s) in {input_dir}")

    pending = []
    for f in npz_files:
        out = Path(output_dir) / f"{f.stem}.csv"
        if skip_existing and out.exists():
            print(f"[SKIP] {f.name} → {out} already exists")
        else:
            pending.append(f)

    if not pending:
        print("[INFO] All files already converted. Nothing to do.")
        return

    for i, npz_path in enumerate(pending, 1):
        out_path = Path(output_dir) / f"{npz_path.stem}.csv"
        print(f"[{i}/{len(pending)}] {npz_path.name} → {out_path}")
        _npz_to_csv(npz_path, out_path, _DEFAULT_JOINT_NAMES)

    print(f"\n[INFO] Done. {len(pending)} file(s) saved to '{output_dir}/'.")


_app = tyro.extras.SubcommandApp()


@_app.command(name="single")
def _cmd_main(
    input_file: str,
    output_file: str,
) -> None:
    """Convert a single NPZ motion file to CSV."""
    main(input_file=input_file, output_file=output_file)


@_app.command(name="dir")
def _cmd_main_dir(
    input_dir: str,
    output_dir: str = "csv_output",
    pattern: str = "*.npz",
    skip_existing: bool = True,
) -> None:
    """Batch-convert a folder of NPZ files to CSV."""
    main_dir(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=pattern,
        skip_existing=skip_existing,
    )


def cli() -> None:
    _app.cli()


if __name__ == "__main__":
    cli()
