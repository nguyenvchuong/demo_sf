"""SMP play-viewer tweaks on top of mjlab's Viser backend."""

from __future__ import annotations

from typing_extensions import override

from mjlab.viewer.viser.viewer import ViserPlayViewer

from smp.rl.viewer.contact_force_plotter import LinkContactForcePlotter


class SmpViserPlayViewer(ViserPlayViewer):
  """Viser play viewer with native MuJoCo contact visualization enabled.

  mjviser halves contact/force overlay scales by default, which makes arrows
  much smaller than the desktop MuJoCo viewer. This restores native sizing and
  turns on MuJoCo's own contact-point / contact-force decor (via
  ``mjv_updateScene``), matching the native viewer rather than simple debug
  arrows.
  """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._link_force_plotter: LinkContactForcePlotter | None = None

  @override
  def setup(self) -> None:
    super().setup()
    self._enable_native_contact_viz()
    self._setup_link_force_plotter()

  def _setup_link_force_plotter(self) -> None:
    env = self.env.unwrapped
    if not hasattr(env.scene, "sensors"):
      return
    if "ground_contact_force" not in env.scene.sensors:
      return
    self._link_force_plotter = LinkContactForcePlotter(
      self._server,
      env,
      sensor_name="ground_contact_force",
      get_env_idx=lambda: self._scene.env_idx,
      history_length=150,
    )
    self._link_force_plotter.setup()

  def _enable_native_contact_viz(self) -> None:
    scene = self._scene
    vis = scene.mj_model.vis

    # mjviser scales these down by 0.5 in ViserMujocoScene.__init__.
    vis.scale.contactwidth *= 2.0
    vis.scale.contactheight *= 2.0
    vis.scale.forcewidth *= 2.0

    scene.show_contact_points = True
    scene.show_contact_forces = True
    scene.needs_update = True

    env = self.env.unwrapped
    if hasattr(env, "metrics_manager"):
      self.log(
        "[INFO] Native MuJoCo contact viz ON. "
        "Metrics tab: per-link force_* terms. "
        "Controls: Link contact forces panel.",
      )
    else:
      self.log(
        "[INFO] Native MuJoCo contact visualization enabled "
        "(Visualization tab to adjust scale/colors).",
      )

  @override
  def _update_env_dependent_plots(self) -> None:
    super()._update_env_dependent_plots()
    if self._link_force_plotter is not None:
      self._link_force_plotter.update(self._is_paused)

  @override
  def reset_environment(self) -> None:
    super().reset_environment()
    if self._link_force_plotter is not None:
      self._link_force_plotter.clear_histories()

  @override
  def close(self) -> None:
    if self._link_force_plotter is not None:
      self._link_force_plotter.cleanup()
      self._link_force_plotter = None
    super().close()


__all__ = ["SmpViserPlayViewer"]
