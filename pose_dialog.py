"""Interactive AETHRYX pose and expression inspection."""

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from aethryx_rig import PosePreview, demo_rotations


class PoseDialog(tk.Toplevel):
    def __init__(self, parent, scene, hidden=()):
        self.preview = PosePreview(scene, simplify=True)
        super().__init__(parent)
        self.title("Genesis · Pose y expresiones")
        self.geometry("1060x680")
        self.transient(parent)
        self.grab_set()
        self.hidden = set(hidden)
        self.rotations = {}
        self.pending = None
        self.bone = tk.StringVar(value="LeftUpperArm")
        self.expression = tk.StringVar(value="Neutra")
        self.strength = tk.DoubleVar(value=0)
        self.show_bones = tk.BooleanVar(value=True)
        self.angles = [tk.DoubleVar(value=0) for _ in range(3)]
        controls = ttk.Frame(self, padding=12)
        controls.pack(side="left", fill="y")
        ttk.Label(controls, text="Pose").pack(anchor="w")
        for text, get_pose in (("Reposo de la malla", dict), ("Aproximar T", self.preview.t_pose),
                               ("Pose de movimiento", demo_rotations)):
            ttk.Button(controls, text=text, command=lambda p=get_pose: self._preset(p())).pack(fill="x", pady=2)
        ttk.Checkbutton(controls, text="Mostrar articulaciones", variable=self.show_bones,
                        command=self._schedule).pack(anchor="w", pady=8)
        ttk.Label(controls, text="Rotación local por hueso").pack(anchor="w")
        combo = ttk.Combobox(controls, textvariable=self.bone, values=self.preview.skeleton[0], state="readonly")
        combo.pack(fill="x")
        combo.bind("<<ComboboxSelected>>", self._sync_joint)
        for axis, variable in zip("XYZ", self.angles):
            ttk.Label(controls, text=axis).pack(anchor="w")
            tk.Scale(controls, variable=variable, from_=-90, to=90, resolution=1, orient="horizontal",
                     command=self._rotate).pack(fill="x")
        ttk.Label(controls, text="Expresión o visema").pack(anchor="w", pady=(10, 0))
        choices = ttk.Combobox(controls, textvariable=self.expression, state="readonly",
                               values=["Neutra", *self.preview.morphs])
        choices.pack(fill="x")
        choices.bind("<<ComboboxSelected>>", self._schedule)
        tk.Scale(controls, variable=self.strength, from_=0, to=1, resolution=.05, orient="horizontal",
                 command=self._schedule).pack(fill="x")
        figure = Figure(facecolor="#11141b")
        self.axes = figure.add_subplot(projection="3d", proj_type="ortho")
        self.axes.view_init(elev=7, azim=-79)
        self.canvas = FigureCanvasTkAgg(figure, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._render()

    def _preset(self, rotations):
        self.rotations = rotations
        self._sync_joint()
        self._schedule()

    def _sync_joint(self, _event=None):
        for variable, angle in zip(self.angles, self.rotations.get(self.bone.get(), (0, 0, 0))):
            variable.set(angle)

    def _rotate(self, _value):
        self.rotations[self.bone.get()] = tuple(value.get() for value in self.angles)
        self._schedule()

    def _schedule(self, _event=None):
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(125, self._render)

    @staticmethod
    def _axes_frame(points):
        return points[:, [0, 2, 1]] * [1, -1, 1]

    def _render(self):
        self.pending = None
        expression = self.expression.get()
        parts, points = self.preview.evaluate(self.rotations, expression if expression != "Neutra" else None,
                                              self.strength.get())
        axes = self.axes
        elev, azim = axes.elev, axes.azim
        axes.clear()
        axes.view_init(elev=elev, azim=azim)
        visible = []
        for name, vertices in parts.items():
            if name in self.hidden:
                continue
            mesh = self.preview.parts[name]
            material = getattr(mesh.visual, "material", None)
            color = getattr(material, "baseColorFactor", None)
            color = np.asarray(color, dtype=float)[:3] / 255 if color is not None else (.60, .69, .78)
            frame = self._axes_frame(vertices)
            visible.append(frame)
            axes.add_collection3d(Poly3DCollection(
                frame[mesh.faces], facecolors=[color], edgecolor="none", linewidths=0, shade=True,
            ))
        joints = self._axes_frame(points)
        if self.show_bones.get():
            names, parents, _ = self.preview.skeleton
            for index, parent in enumerate(parents):
                if parent >= 0:
                    bone = joints[[parent, index]]
                    axes.plot(*bone.T, color="#ff268f" if names[index] == self.bone.get() else "#00d9ff",
                              linewidth=1.1, zorder=20)
        merged = np.vstack([*visible, joints])
        bounds = np.array([merged.min(axis=0), merged.max(axis=0)])
        for setter, limits in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim), bounds.T):
            setter(*limits)
        axes.set_box_aspect(np.maximum(np.ptp(merged, axis=0), 1e-5))
        axes.set_facecolor("#11141b")
        axes.set_axis_off()
        self.canvas.draw_idle()

    def destroy(self):
        if self.pending is not None:
            self.after_cancel(self.pending)
        super().destroy()
