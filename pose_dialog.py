"""Interactive AETHRYX pose and expression inspection."""

import tkinter as tk
from tkinter import messagebox, ttk

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
        self.geometry("1060x790")
        self.transient(parent)
        self.grab_set()
        self.hidden = set(hidden)
        self.source = scene
        self.offsets = dict(scene.metadata.get("aethryx_joint_offsets", {}))
        self.result = None
        self.rotations = {}
        self.pending = None
        self.speech_job = None
        self.bone = tk.StringVar(value="LeftUpperArm")
        self.expression = tk.StringVar(value="Neutra")
        self.strength = tk.DoubleVar(value=0)
        self.expression_two = tk.StringVar(value="Neutra")
        self.strength_two = tk.DoubleVar(value=0)
        self.sequence = tk.StringVar(value="aa e ih oh ou")
        self.rest_fields = [tk.StringVar(value="0") for _ in range(3)]
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
        ttk.Label(controls, text="Desplazamiento de reposo X Y Z").pack(anchor="w", pady=(8, 0))
        rest = ttk.Frame(controls)
        rest.pack(fill="x")
        for variable in self.rest_fields:
            ttk.Entry(rest, textvariable=variable, width=7).pack(side="left", padx=2)
        ttk.Button(controls, text="Ajustar articulación", command=self._offset).pack(fill="x")
        ttk.Label(controls, text="Expresión o visema").pack(anchor="w", pady=(10, 0))
        choices = ttk.Combobox(controls, textvariable=self.expression, state="readonly",
                               values=["Neutra", *self.preview.morphs])
        choices.pack(fill="x")
        choices.bind("<<ComboboxSelected>>", self._schedule)
        tk.Scale(controls, variable=self.strength, from_=0, to=1, resolution=.05, orient="horizontal",
                 command=self._schedule).pack(fill="x")
        ttk.Label(controls, text="Mezclar con una segunda expresión").pack(anchor="w")
        second = ttk.Combobox(controls, textvariable=self.expression_two, state="readonly",
                              values=["Neutra", *self.preview.morphs])
        second.pack(fill="x")
        second.bind("<<ComboboxSelected>>", self._schedule)
        tk.Scale(controls, variable=self.strength_two, from_=0, to=1, resolution=.05, orient="horizontal",
                 command=self._schedule).pack(fill="x")
        sequence = ttk.Frame(controls)
        sequence.pack(fill="x")
        ttk.Entry(sequence, textvariable=self.sequence, width=16).pack(side="left", fill="x", expand=True)
        ttk.Button(sequence, text="Ver visemas", command=self._speech).pack(side="right")
        ttk.Button(controls, text="Guardar ajustes del esqueleto", command=self._apply).pack(fill="x", pady=7)
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
        for variable, value in zip(self.rest_fields, self.offsets.get(self.bone.get(), (0, 0, 0))):
            variable.set(str(value))

    def _offset(self):
        try:
            name = self.bone.get()
            values = [float(field.get().replace(",", ".")) for field in self.rest_fields]
            amended = dict(self.offsets)
            if any(values):
                amended[name] = values
            else:
                amended.pop(name, None)
            preview = PosePreview(self.source, simplify=True, offsets=amended)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Ajustar articulación", str(exc), parent=self)
            return
        self.preview, self.offsets = preview, amended
        self._schedule()

    def _speech(self):
        sequence = [f"vrc.v_{item}" for item in self.sequence.get().lower().split()]
        if not sequence or any(name not in self.preview.morphs for name in sequence):
            messagebox.showerror("Previsualizar visemas", "Usa nombres de visemas, por ejemplo aa e ih oh ou.", parent=self)
            return
        if self.speech_job is not None:
            self.after_cancel(self.speech_job)
        if self.pending is not None:
            self.after_cancel(self.pending)
            self.pending = None

        def frame(index):
            self.speech_job = None
            self.expression.set(sequence[index] if index < len(sequence) else "Neutra")
            self.strength.set(.85)
            self._render()
            if index < len(sequence):
                self.speech_job = self.after(280, frame, index + 1)

        frame(0)

    def _apply(self):
        self.result = self.source.copy()
        self.result.metadata["aethryx_joint_offsets"] = dict(self.offsets)
        self.destroy()

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
        blend = {}
        for expression, value in ((self.expression.get(), self.strength.get()),
                                  (self.expression_two.get(), self.strength_two.get())):
            if expression != "Neutra":
                blend[expression] = min(1, blend.get(expression, 0) + value)
        parts, points = self.preview.evaluate(self.rotations, blend)
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
        if self.speech_job is not None:
            self.after_cancel(self.speech_job)
        super().destroy()
