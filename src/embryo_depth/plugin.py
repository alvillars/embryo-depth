"""napari plugin entry point: a dock widget wrapping the depth-shell viewer.

``viewer.py`` is the standalone command-line form; this is the same machinery exposed
through napari's plugin menu so an already-open viewer can load shells into itself.

Note the deliberate absence of ``from __future__ import annotations``. It would turn every
annotation into a string, and magicgui resolves annotations at widget-construction time --
it cannot resolve a bare ``'Path'`` forward reference and the widget fails to build. The
types below must stay concrete and importable at module scope.
"""

from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from qtpy.QtWidgets import QVBoxLayout, QWidget

from .store import DEFAULT_STORE, Dataset
from .viewer import DEFAULT_LEVEL, DEFAULT_MODE, ShellSource


def depth_shell_widget() -> QWidget:
    state: dict = {"source": None, "layer": None, "mode": DEFAULT_MODE}

    @magicgui(
        call_button="Load depth index",
        store={"widget_type": "FileEdit", "mode": "d"},
        level={"choices": ["1", "2", "3", "4"]},
        mode={"choices": ["volume", "points"]},
        timepoints={"widget_type": "SpinBox", "min": 1, "max": 10000},
    )
    def loader(
        viewer: napari.Viewer,
        store: Path = DEFAULT_STORE,
        level: str = DEFAULT_LEVEL,
        mode: str = DEFAULT_MODE,
        timepoints: int = 132,
    ) -> None:
        ds = Dataset(store)
        if not ds.has_depth():
            raise RuntimeError("no depth/ group -- run `python -m embryo_depth.depth` first")
        src = ShellSource(ds, level, range(min(timepoints, ds.n_timepoints)))
        src.load()
        state["source"] = src
        state["mode"] = mode
        scale = src.lv.voxel_size_um
        t0 = src.timepoints[0]

        if state["layer"] is not None and state["layer"] in viewer.layers:
            viewer.layers.remove(state["layer"])
        if mode == "volume":
            state["layer"] = viewer.add_image(
                src.shell_volume(t0, shell.depth_um.value, shell.half_width_um.value),
                name="shell",
                scale=scale,
                rendering="attenuated_mip",
                contrast_limits=[32, 6000],
            )
        else:
            coords, values = src.shell(t0, shell.depth_um.value, shell.half_width_um.value)
            state["layer"] = viewer.add_points(
                coords * np.array(scale),
                name="shell",
                size=max(scale) * 1.5,
                features={"intensity": values},
                face_color="intensity",
                face_colormap="gray",
                border_width=0,
                shading="none",
            )
        viewer.dims.ndisplay = 3
        shell.depth_um.max = max(src.max_depth_unit, 1)
        shell.timepoint.max = max(len(src.timepoints) - 1, 0)

    @magicgui(
        auto_call=True,
        timepoint={"widget_type": "Slider", "min": 0, "max": 131},
        depth_um={"widget_type": "Slider", "min": 0, "max": 255, "label": "depth (um)"},
        half_width_um={"widget_type": "Slider", "min": 1, "max": 20, "label": "+/- (um)"},
    )
    def shell(
        timepoint: int = 0,
        depth_um: int = 10,
        half_width_um: int = 3,
        hide_crop_bounded: bool = False,
    ) -> None:
        src = state["source"]
        layer = state["layer"]
        if src is None or layer is None:
            return
        t = src.timepoints[min(timepoint, len(src.timepoints) - 1)]
        if state["mode"] == "volume":
            layer.data = src.shell_volume(t, depth_um, half_width_um, hide_crop_bounded)
        else:
            coords, values = src.shell(t, depth_um, half_width_um, hide_crop_bounded)
            scale = np.array(src.lv.voxel_size_um)
            layer.data = coords * scale
            layer.features = {"intensity": values}
            layer.face_color = "intensity"

    container = QWidget()
    layout = QVBoxLayout(container)
    layout.addWidget(loader.native)
    layout.addWidget(shell.native)
    container._loader = loader  # keep references alive and reachable for tests
    container._shell = shell
    return container
