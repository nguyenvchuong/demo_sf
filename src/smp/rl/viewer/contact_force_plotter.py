"""Multi-link peak contact-force plot for the Viser play viewer."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
import torch
import viser
import viser.uplot

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor.contact_sensor import ContactSensor

_PALETTE = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
  "#aec7e8",
  "#ffbb78",
]


class LinkContactForcePlotter:
  """One uplot with a time series per robot link (peak contact force, N)."""

  def __init__(
    self,
    server: viser.ViserServer,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    get_env_idx: Callable[[], int],
    history_length: int = 150,
  ) -> None:
    self._server = server
    self._env = env
    self._sensor_name = sensor_name
    self._get_env_idx = get_env_idx
    self._history_length = history_length
    self._x_array = np.arange(-history_length + 1, 1, dtype=np.float64)
    self._empty = np.array([], dtype=np.float64)

    self._link_names: list[str] = []
    self._histories: list[deque[float]] = []
    self._enabled: list[bool] = []
    self._checkboxes: list[viser.GuiInputHandle] = []
    self._plot: viser.GuiUplotHandle | None = None
    self._plot_folder = server.gui.add_folder(
      "Link contact forces", expand_by_default=True
    )

  def setup(self) -> None:
    sensor: ContactSensor | None = self._env.scene.sensors.get(self._sensor_name)
    if sensor is None:
      with self._plot_folder:
        self._server.gui.add_markdown(
          "<small>No ground contact sensor found for per-link force plot.</small>"
        )
      return

    self._link_names = list(sensor.primary_names)
    self._histories = [deque(maxlen=self._history_length) for _ in self._link_names]
    self._enabled = [False] * len(self._link_names)

    with self._plot_folder:
      self._server.gui.add_markdown(
        "<small>Peak ground contact force per link (N). "
        "Enable links below; active ones share one plot.</small>"
      )
      bulk = self._server.gui.add_button_group("Links", options=["All", "None"])

      @bulk.on_click
      def _(event) -> None:
        enable = event.target.value == "All"
        for i, cb in enumerate(self._checkboxes):
          self._enabled[i] = enable
          cb.value = enable
        self._sync_plot()

      for i, name in enumerate(self._link_names):
        short = name.removesuffix("_link") if name.endswith("_link") else name
        cb = self._server.gui.add_checkbox(
          short,
          initial_value=False,
          hint=f"Color: {_PALETTE[i % len(_PALETTE)]}",
        )
        self._checkboxes.append(cb)

        @cb.on_update
        def _(event, idx=i) -> None:
          self._enabled[idx] = event.target.value
          self._sync_plot()

      self._plots_subfolder = self._server.gui.add_folder("Plot", expand_by_default=True)

  def update(self, paused: bool) -> None:
    if paused or not self._link_names:
      return

    sensor: ContactSensor = self._env.scene.sensors[self._sensor_name]
    data = sensor.data
    if data.force is None:
      return

    env_idx = self._get_env_idx()
    force_norm = torch.norm(data.force, dim=-1)[env_idx].detach().cpu().numpy()

    for i, value in enumerate(force_norm):
      if np.isfinite(value):
        self._histories[i].append(float(value))

    if not any(self._enabled):
      return

    if self._plot is None:
      self._sync_plot()
    elif self._plot is not None:
      self._plot.data = self._build_plot_data()

  def clear_histories(self) -> None:
    for history in self._histories:
      history.clear()
    if self._plot is not None:
      self._plot.data = self._empty_plot_data()

  def cleanup(self) -> None:
    if self._plot is not None:
      self._plot.remove()
      self._plot = None
    for cb in self._checkboxes:
      cb.remove()
    self._checkboxes.clear()
    self._plot_folder.remove()

  def _empty_plot_data(self) -> tuple[np.ndarray, ...]:
    return (self._empty,)

  def _build_plot_data(self) -> tuple[np.ndarray, ...]:
    enabled_indices = [i for i, on in enumerate(self._enabled) if on]
    if not enabled_indices:
      return self._empty_plot_data()

    hist_len = max((len(self._histories[i]) for i in enabled_indices), default=0)
    if hist_len == 0:
      return self._empty_plot_data()

    x = self._x_array[-hist_len:]
    ys = []
    for i in enabled_indices:
      history = self._histories[i]
      ys.append(np.fromiter(history, dtype=np.float64, count=len(history)))
    return (x, *ys)

  def _sync_plot(self) -> None:
    if self._plot is not None:
      self._plot.remove()
      self._plot = None

    enabled_indices = [i for i, on in enumerate(self._enabled) if on]
    if not enabled_indices:
      return

    series: list[viser.uplot.Series] = [viser.uplot.Series(label="Steps")]
    for i in enabled_indices:
      short = self._link_names[i].removesuffix("_link")
      series.append(
        viser.uplot.Series(
          label=short,
          stroke=_PALETTE[i % len(_PALETTE)],
          width=2,
        )
      )

    with self._plots_subfolder:
      self._plot = self._server.gui.add_uplot(
        data=self._build_plot_data(),
        series=tuple(series),
        scales={
          "x": viser.uplot.Scale(
            time=False, auto=False, range=(-self._history_length, 0)
          ),
          "y": viser.uplot.Scale(auto=True),
        },
        legend=viser.uplot.Legend(show=True),
        title="Peak contact force per link (N)",
        aspect=2.5,
        visible=True,
      )
