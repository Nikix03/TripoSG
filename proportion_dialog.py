"""Interactive preview of the anatomical proportion controls."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aethryx_anatomy import PROPORTIONS, anime_proportions


class ProportionDialog(tk.Toplevel):
    def __init__(self, master, mesh, values):
        super().__init__(master)
        self.title("Proporciones · Vista 3D de la plantilla")
        self.geometry("960x680")
        self.transient(master)
        self.grab_set()
        self.result = None
        self.pending = None
        self.bounds = mesh.bounds.copy()
        if len(mesh.faces) > 6500:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=6500)
            except (ImportError, ValueError, RuntimeError, TypeError):
                pass
        self.mesh = mesh
        self.values = {key: tk.DoubleVar(value=values.get(key, 1)) for key in PROPORTIONS}
        controls = ttk.Frame(self, padding=10)
        controls.pack(side="left", fill="y")
        for key, label in PROPORTIONS.items():
            ttk.Label(controls, text=label).pack(anchor="w")
            tk.Scale(controls, from_=.75, to=1.35, resolution=.01, orient="horizontal",
                     showvalue=True, variable=self.values[key], length=180,
                     command=self._schedule).pack(fill="x")
        ttk.Button(controls, text="Aplicar proporciones", command=self._apply).pack(fill="x", pady=8)
        figure = Figure(facecolor="#11141b")
        self.axes = figure.add_subplot(projection="3d", proj_type="ortho")
        self.axes.view_init(elev=8, azim=-63)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._render()

    def _factors(self):
        return {key: value.get() for key, value in self.values.items()}

    def _schedule(self, _value):
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(100, self._render)

    def _render(self):
        self.pending = None
        axes = self.axes
        elev, azim = axes.elev, axes.azim
        axes.clear()
        axes.view_init(elev=elev, azim=azim)
        mesh = self.mesh
        vertices = anime_proportions(mesh.vertices, self.bounds, self._factors())[:, [0, 2, 1]]
        triangles = vertices[mesh.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shade = .45 + .55 * np.maximum(normals @ np.array([.24, .76, .58]), 0)
        colors = shade[:, None] * np.array([.40, .80, .96])
        axes.add_collection3d(Poly3DCollection(triangles, facecolors=colors, linewidths=0))
        axes.set_facecolor("#11141b")
        axes.set_axis_off()
        limits = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        for setter, bounds in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), limits.T):
            setter(*bounds)
        axes.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-5), zoom=1.1)
        self.canvas.draw_idle()

    def _apply(self):
        self.result = self._factors()
        self.destroy()

    def destroy(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        super().destroy()
