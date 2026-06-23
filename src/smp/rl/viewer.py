"""SMP play-viewer tweaks on top of mjlab's Viser backend."""

from __future__ import annotations

from typing_extensions import override

from mjlab.viewer.viser.viewer import ViserPlayViewer


class SmpViserPlayViewer(ViserPlayViewer):
  """Viser play viewer with native MuJoCo contact visualization enabled.

  mjviser halves contact/force overlay scales by default, which makes arrows
  much smaller than the desktop MuJoCo viewer. This restores native sizing and
  turns on MuJoCo's own contact-point / contact-force decor (via
  ``mjv_updateScene``), matching the native viewer rather than simple debug
  arrows.
  """

  @override
  def setup(self) -> None:
    super().setup()
    self._enable_native_contact_viz()

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

    self.log(
      "[INFO] Native MuJoCo contact visualization enabled "
      "(Visualization tab to adjust scale/colors).",
    )
