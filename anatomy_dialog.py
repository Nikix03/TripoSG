"""Visual calibration and landmark editor for Genesis."""

from copy import deepcopy
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from aethryx_anatomy import (
    PROPORTIONS, VIEWS, blocking_geometry_issues, export_studio_reconstruction,
    load_body_template, orthographic_projection,
    reconstruct_data,
)
from aethryx_engines import suggest_anatomical_points
from aethryx_profiles import BODY_PROFILES, BUNDLED_BODY, POSE_VERTICES
from aethryx_reference import (
    AnimeSilhouette, SEE_THROUGH_COLORS, SEMANTIC_COLORS, compare_silhouette, import_see_through_v2, profile_opposite,
    foreground_crop_diagnostics, named_reference_views, reference_silhouette, refine_visible_mask,
    pose_points_for_view, segment_anime_reference, vertical_pose_disagreements,
    segment_see_through_v2, semantic_reference_regions, silhouette_calibration,
)
from proportion_dialog import ProportionDialog
from mask_dialog import MaskDialog
from character_sheet import export_confirmed_panels


VIEW_LABELS = dict(zip(VIEWS, ("Frontal", "Lateral", "Posterior")))
POSE_LABELS = {
    "Nariz": 0, "Oreja izquierda": 7, "Oreja derecha": 8,
    "Hombro izquierdo": 11, "Hombro derecho": 12,
    "Codo izquierdo": 13, "Codo derecho": 14,
    "Muñeca izquierda": 15, "Muñeca derecha": 16,
    "Cadera izquierda": 23, "Cadera derecha": 24,
    "Rodilla izquierda": 25, "Rodilla derecha": 26,
    "Tobillo izquierdo": 27, "Tobillo derecho": 28,
}
IMAGE_FILES = [("Imagen", "*.png *.jpg *.jpeg *.webp *.bmp")]


class AnatomyDialog(tk.Toplevel):
    def __init__(self, master, reference_regions=()):
        super().__init__(master)
        self.title("AETHRYX · Reconstrucción anatómica")
        self.geometry("1160x790")
        self.minsize(880, 660)
        self.transient(master)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.result = None
        self.draft = None
        self.triposg_crop = None
        self.reference_regions = deepcopy(reference_regions)
        self.profile = tk.StringVar(value="Anime neutro")
        self.template_path = None
        self.template = None
        self.indices = []
        self.pose_links = {}
        self.views = {name: {"points": {}} for name in VIEWS}
        self.palette = {}
        self.images = {}
        self.comparisons = {}
        self.comparison_mode = tk.BooleanVar(value=False)
        self.visual_hull = tk.BooleanVar(value=False)
        self.export_rig = tk.BooleanVar(value=False)
        self.export_hybrid = tk.BooleanVar(value=False)
        self.infer_occlusion_depth = tk.BooleanVar(value=False)
        self.segmenter = None
        self.see_through = None
        self.segment_cancel = Event()
        self.next_calibration = None
        self.busy = False
        self.photo = {}
        self.image_frame = None
        self.template_pixels = None
        self.side = tk.BooleanVar(value=False)
        self.side_manual = False
        self.overlay = tk.BooleanVar(value=True)
        self.view_label = tk.StringVar(value=VIEW_LABELS["front"])
        self.pose_label = tk.StringVar()
        self.status = tk.StringVar(value="Carga una plantilla corporal y las tres imágenes.")
        self.factors = {name: tk.StringVar(value="1.00") for name in PROPORTIONS}
        self._build()
        if BUNDLED_BODY.is_file():
            self._use_profile()

    @property
    def view(self):
        return VIEWS[list(VIEW_LABELS.values()).index(self.view_label.get())]

    def _build(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        mask_tools = ttk.Frame(self, padding=(8, 0, 8, 6))
        mask_tools.pack(fill="x")
        for label, command in (
            ("Plantilla corporal…", self._load_template),
            ("Abrir ajuste…", self._open_setup),
            ("Guardar ajuste…", self._save_setup),
            ("Cargar imagen…", self._load_image),
            ("Cargar conjunto…", self._load_reference_set),
            ("Usar paneles confirmados…", self._import_confirmed_panels),
        ):
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=3)
        for label, command in (
            ("Máscara RGB…", self._load_mask),
            ("Máscara corporal…", self._load_body_mask),
            ("Paleta…", self._edit_palette),
            ("Segmentar ONNX…", self._segment_current),
            ("Reparar silueta…", self._repair_body_mask),
        ):
            ttk.Button(mask_tools, text=label, command=command).pack(side="left", padx=3)
        ttk.Label(mask_tools, textvariable=self.status, wraplength=390).pack(side="left", padx=10)
        views = ttk.Frame(self, padding=(8, 0, 8, 6))
        views.pack(fill="x")
        choose = ttk.Combobox(views, textvariable=self.view_label, values=list(VIEW_LABELS.values()),
                              state="readonly", width=11)
        choose.pack(side="left")
        choose.bind("<<ComboboxSelected>>", self._change_view)
        self.profile_combo = ttk.Combobox(views, state="readonly", width=16, textvariable=self.profile,
                                          values=[*BODY_PROFILES, "Personalizado"])
        self.profile_combo.pack(side="left", padx=4)
        self.profile_combo.bind("<<ComboboxSelected>>", self._select_profile)
        ttk.Button(views, text="Marcar suelo y coronilla", command=self._start_calibration).pack(side="left", padx=6)
        self.profile_toggle = ttk.Checkbutton(views, text="Perfil opuesto", variable=self.side,
                                              command=self._reverse_side)
        self.profile_toggle.pack(side="left")
        self.profile_toggle.state(["disabled"])
        ttk.Checkbutton(views, text="Proyectar vértices", variable=self.overlay,
                        command=self._redraw).pack(side="left")
        ttk.Button(views, text="Sugerir puntos (MediaPipe)", command=self._suggest).pack(side="left", padx=6)
        ttk.Button(views, text="Alinear y detectar vistas", command=self._auto_align).pack(side="left", padx=6)
        ttk.Button(views, text="Alinear con ISNet", command=lambda: self._auto_align(use_isnet=True)).pack(side="left")
        ttk.Button(views, text="Importar See-through V2…", command=self._import_see_through).pack(side="left")
        ttk.Button(views, text="Ejecutar See-through…", command=self._run_see_through).pack(side="left")
        ttk.Button(views, text="Cancelar segmentación", command=self.segment_cancel.set).pack(side="left")

        panels = ttk.Panedwindow(self, orient="horizontal")
        panels.pack(fill="both", expand=True, padx=8)
        for name, title, click in (
            ("image", "Referencia · clic para colocar o corregir el punto", self._image_click),
            ("template", "Plantilla · clic para elegir un vértice", self._template_click),
        ):
            panel = ttk.Labelframe(panels, text=title)
            canvas = tk.Canvas(panel, bg="#11141b", highlightthickness=0, width=420, height=300)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Button-1>", click)
            canvas.bind("<Configure>", self._redraw)
            panels.add(panel, weight=1)
            setattr(self, f"{name}_canvas", canvas)

        controls = ttk.Frame(self, padding=8)
        controls.pack(fill="x")
        self.points = tk.Listbox(controls, height=5, width=36, exportselection=False)
        self.points.pack(side="left")
        self.points.bind("<<ListboxSelect>>", self._point_selected)
        point_actions = ttk.Frame(controls)
        point_actions.pack(side="left", padx=8)
        ttk.Button(point_actions, text="Marcar oculto en esta vista", command=self._clear_point).pack(fill="x")
        ttk.Button(point_actions, text="Eliminar vértice del ajuste", command=self._remove_point).pack(fill="x")
        self.pose_combo = ttk.Combobox(point_actions, textvariable=self.pose_label,
                                       values=["", *POSE_LABELS], state="readonly")
        self.pose_combo.pack(fill="x", pady=(4, 0))
        self.pose_combo.bind("<<ComboboxSelected>>", self._link_pose)
        ttk.Label(point_actions, text="Correspondencia para sugerencias").pack()
        ttk.Button(controls, text="Proporciones 3D…", command=self._proportions).pack(side="left", padx=6)
        ttk.Checkbutton(controls, text="Comparar siluetas", variable=self.comparison_mode,
                        command=self._redraw).pack(side="left")
        ttk.Checkbutton(controls, text="Volumen semántico", variable=self.visual_hull).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Exportar rig GLB", variable=self.export_rig).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Avatar con cara, cabello y camiseta", variable=self.export_hybrid).pack(side="left", padx=4)
        ttk.Checkbutton(controls, text="Inferir puntos ocluidos", variable=self.infer_occlusion_depth).pack(side="left", padx=4)
        ttk.Button(controls, text="Reconstruir y validar…", command=self._fit).pack(side="right")

    def _close(self):
        if not self.busy:
            if self.images and self.template_path:
                try:
                    self.draft = self._build_manifest(False)
                except ValueError as exc:
                    self._error(exc)
                    return
            self.destroy()

    def _load_template(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Malla corporal", "*.ply *.obj *.glb")])
        if not path:
            return
        try:
            mesh = load_body_template(path)
            if mesh.extents[1] <= 0:
                raise ValueError("La plantilla requiere altura en el eje Y.")
        except Exception as exc:
            self._error(exc)
            return
        if any(view["points"] or "projection" in view for view in self.views.values()) and not messagebox.askyesno(
                "Cambiar plantilla", "¿Reiniciar los vértices y la calibración?", parent=self):
            return
        self.profile.set("Personalizado")
        self._replace_template(path, mesh)

    def _replace_template(self, path, mesh):
        self.template_path, self.template = str(path), mesh
        self.indices.clear()
        self.pose_links.clear()
        for view in self.views.values():
            view["points"].clear()
            view.pop("projection", None)
            view.pop("calibration", None)
        self._invalidate_comparison()
        self._change_view()

    def _select_profile(self, _event=None):
        if not self.busy and self.profile.get() != "Personalizado":
            self._use_profile()

    def _use_profile(self):
        profile = self.profile.get()
        if self.template_path != str(BUNDLED_BODY):
            if any(view["points"] or "projection" in view for view in self.views.values()) and not messagebox.askyesno(
                    "Cambiar plantilla", "¿Reiniciar las correspondencias y la calibración?", parent=self):
                self.profile.set("Personalizado")
                return
            try:
                template = load_body_template(BUNDLED_BODY)
            except Exception as exc:
                self.profile.set("Personalizado")
                self._error(exc)
                return
            self._replace_template(BUNDLED_BODY, template)
            self.indices = list(POSE_VERTICES.values())
            self.pose_links = {vertex: joint for joint, vertex in POSE_VERTICES.items()}
        for key, value in self.factors.items():
            value.set(BODY_PROFILES[profile].get(key, 1))
        self._invalidate_comparison()
        self._refresh_points()
        self.status.set("Perfil cargado. Añade las tres referencias y usa la alineación automática.")

    def _proportions(self):
        if self.busy or self.template is None:
            return
        dialog = ProportionDialog(self, self.template, {key: float(var.get()) for key, var in self.factors.items()})
        self.wait_window(dialog)
        self.grab_set()
        if dialog.result:
            for key, value in dialog.result.items():
                self.factors[key].set(value)
            self._invalidate_comparison()
            self._redraw()

    def _load_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=IMAGE_FILES)
        if not path:
            return
        if self.views[self.view]["points"] and not messagebox.askyesno(
                "Cambiar referencia", "¿Reiniciar los puntos de esta vista?", parent=self):
            return
        try:
            with Image.open(path) as image:
                preview = image.convert("RGBA")
        except OSError as exc:
            self._error(exc)
            return
        self.images[self.view] = preview
        self.views[self.view] = {"image": path, "points": {}}
        self._invalidate_comparison()
        self._change_view()

    def _load_reference_set(self):
        if self.busy:
            return
        paths = filedialog.askopenfilenames(parent=self, filetypes=IMAGE_FILES,
                                            title="Referencias con nombre frontal, lateral o posterior")
        if not paths:
            return
        try:
            chosen = named_reference_views(paths)
            previews = {}
            for name, path in chosen.items():
                with Image.open(path) as image:
                    previews[name] = image.convert("RGBA")
        except (OSError, ValueError) as exc:
            self._error(exc)
            return
        assignments = "\n".join(f"{VIEW_LABELS[name]} · {Path(path).name}" for name, path in chosen.items())
        if not messagebox.askyesno("Confirmar perspectivas", assignments + "\n\n¿Cargar estas vistas y reiniciar sus ajustes?",
                                   parent=self):
            return
        for name, path in chosen.items():
            self.views[name] = {"image": path, "points": {}}
        self.images.update(previews)
        self._invalidate_comparison()
        self.view_label.set(VIEW_LABELS[next(name for name in VIEWS if name in chosen)])
        self._change_view()

    def _import_confirmed_panels(self):
        if self.busy:
            return
        if not self.reference_regions:
            self.status.set("Confirma los recortes de cuerpo completo en el editor de hojas de Studio.")
            return
        if not messagebox.askyesno("Importar paneles",
                                   "¿Guardar los paneles confirmados y reiniciar los ajustes de esas vistas?",
                                   parent=self):
            return
        directory = filedialog.askdirectory(parent=self)
        if not directory:
            return
        try:
            panels = export_confirmed_panels(self.reference_regions, directory)
            for name in VIEWS:
                if name in panels:
                    with Image.open(panels[name]) as source:
                        self.images[name] = source.convert("RGBA")
                    self.views[name] = {"image": panels[name], "points": {}}
            self._invalidate_comparison()
            self._change_view()
            self.status.set(f"{len(panels)} paneles guardados. Calibra las vistas ortogonales cargadas.")
        except (OSError, ValueError) as exc:
            self._error(exc)

    def _load_mask(self):
        self._select_mask("mask", "Máscara semántica")

    def _load_body_mask(self):
        self._select_mask("body_mask", "Silueta corporal")

    def _repair_body_mask(self):
        view = self.view
        if self.busy or view not in self.images:
            return
        image = self.images[view]
        state = self.views[view]

        def edit(mask):
            editor = MaskDialog(self, image, mask, {"background": (0, 0, 0), "silueta": (255, 255, 255)})
            self.wait_window(editor)
            if editor.result is None:
                return
            if not editor.result.getbbox():
                self._error(ValueError("La silueta quedó vacía."))
                return
            path = filedialog.asksaveasfilename(parent=self, initialfile=f"{view}-silueta-revisada.png",
                                                 defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not path:
                return
            try:
                with Path(path).open("xb") as file:
                    editor.result.save(file, format="PNG")
            except OSError as exc:
                self._error(exc)
                return
            state["body_mask"] = path
            self._invalidate_comparison()
            edges = foreground_crop_diagnostics(editor.result)["edge_pixels"]
            warning = " Todavía toca un borde. Revisa el recorte o calibra a mano." if any(edges.values()) else ""
            self.status.set(f"Silueta revisada guardada para {VIEW_LABELS[view]}.{warning}")

        if state.get("body_mask"):
            try:
                with Image.open(state["body_mask"]) as loaded:
                    mask = loaded.convert("RGB")
                if mask.size != image.size:
                    raise ValueError("La máscara debe tener las dimensiones de la referencia.")
            except (OSError, ValueError) as exc:
                self._error(exc)
                return
            edit(mask)
        else:
            self._work("Separando primer plano con ISNet…", lambda: AnimeSilhouette().predict(image), edit)

    def _select_mask(self, key, label):
        if self.busy or self.view not in self.images:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Máscara RGB", "*.png")])
        if path:
            try:
                with Image.open(path) as mask:
                    if mask.size != self.images[self.view].size:
                        raise ValueError("La máscara debe tener las dimensiones de la referencia.")
            except (OSError, ValueError) as exc:
                self._error(exc)
                return
            self.views[self.view][key] = path
            self._invalidate_comparison()
            self.status.set(f"{label} cargada en {self.view_label.get()}.")

    def _import_see_through(self):
        if self.busy or self.view not in self.images:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Máscaras booleanas V2", "*.npy")])
        if path:
            self._segment_current(external=path)

    def _run_see_through(self):
        if self.busy or self.view not in self.images:
            return
        if self.see_through is None:
            source = filedialog.askdirectory(parent=self, title="Código fuente oficial de See-through")
            if not source:
                return
            weights = filedialog.askopenfilename(parent=self, title="Checkpoint V2 oficial",
                                                 filetypes=[("Pesos V2", "*.pt")])
            if not weights:
                return
            python = filedialog.askopenfilename(parent=self, title="Python con PyTorch y CUDA",
                                                initialdir=str(Path(sys.executable).parent),
                                                filetypes=[("Python", "python.exe")])
            if not python:
                return
            self.see_through = python, source, weights
        self.segment_cancel.clear()
        self._segment_current(run_see_through=True)

    def _segment_current(self, external=None, *, run_see_through=False):
        view = self.view
        if self.busy or view not in self.images:
            return
        if external is None and not run_see_through and self.segmenter is None:
            model = filedialog.askopenfilename(parent=self, filetypes=[("ONNX multiclase", "*.onnx")])
            if not model:
                return
            raw = simpledialog.askstring(
                "Canales del modelo", "Introduce las clases en el orden de salida del modelo.",
                initialvalue=",".join(SEMANTIC_COLORS), parent=self,
            )
            if raw is None:
                return
            self.segmenter = model, [label.strip().lower() for label in raw.split(",")]
        image = self.images[view]
        colors = SEE_THROUGH_COLORS if external or run_see_through else SEMANTIC_COLORS
        def accept(mask):
            if mask is None:
                self.status.set("Segmentación cancelada.")
                return
            self.status.set(f"Máscara propuesta para {VIEW_LABELS[view]}.")
            preview = tk.Toplevel(self)
            preview.title("Revisar segmentación multiclase")
            preview.transient(self)
            preview.grab_set()
            scaled = mask.copy()
            scaled.thumbnail((620, 560), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(scaled)
            label = ttk.Label(preview, image=photo)
            label.pack(padx=8, pady=8)
            def reopen(new_mask):
                preview.destroy()
                self.after_idle(lambda: accept(new_mask))

            def edit_mask():
                editor = MaskDialog(preview, image, mask, colors)
                preview.wait_window(editor)
                if editor.result is not None:
                    reopen(editor.result)
                else:
                    preview.grab_set()

            def refine_mask():
                preview.destroy()
                self._work("Combinando See-through e ISNet…",
                           lambda: refine_visible_mask(image, mask, AnimeSilhouette().predict(image)),
                           accept)

            edit_actions = ttk.Frame(preview)
            edit_actions.pack()
            ttk.Button(edit_actions, text="Editar máscara…", command=edit_mask).pack(side="left")
            if colors is SEE_THROUGH_COLORS:
                ttk.Button(edit_actions, text="Refinar piel visible con ISNet", command=refine_mask).pack(side="left")
            regions = semantic_reference_regions(image, mask, colors)
            legend = ttk.Frame(preview)
            legend.pack()
            for key, caption in zip(colors, ("Fondo", "Cuello", "Rostro", "Cabello", "Ropa", "Accesorios", "Ojos")):
                if key != "background":
                    color = "#%02x%02x%02x" % colors[key]
                    if key == "body" and colors is SEMANTIC_COLORS:
                        caption = "Cuerpo"
                    elif key == "body":
                        caption = "Piel visible"
                    if key in regions:
                        caption += f" {regions[key]['color']}"
                    tk.Label(legend, text=caption, bg=color, fg="#11141b", padx=6).pack(side="left")
            def export_crops():
                directory = filedialog.askdirectory(parent=preview)
                if not directory:
                    return
                from tempfile import mkdtemp
                folder = Path(mkdtemp(prefix="recortes-", dir=directory))
                try:
                    for name, region in regions.items():
                        region["crop"].save(folder / f"{name}.png")
                    (folder / "regiones.json").write_text(json.dumps({
                        name: {key: value for key, value in region.items() if key != "crop"}
                        for name, region in regions.items()
                    }, indent=2), encoding="utf-8")
                    self.status.set(f"Recortes transparentes guardados en {folder.name}.")
                except OSError as exc:
                    self._error(exc)
            ttk.Button(preview, text="Exportar recortes transparentes…", command=export_crops,
                       state="normal" if regions else "disabled").pack(pady=4)
            if regions:
                row = ttk.Frame(preview)
                row.pack(pady=4)
                chosen = tk.StringVar(master=preview, value=next(iter(regions)))
                ttk.Combobox(row, textvariable=chosen, values=list(regions), width=15,
                             state="readonly").pack(side="left")

                def send_crop():
                    name = chosen.get()
                    path = filedialog.asksaveasfilename(parent=preview, defaultextension=".png",
                                                        initialfile=f"recorte-{name}.png",
                                                        filetypes=[("PNG con transparencia", "*.png")])
                    if not path:
                        return
                    try:
                        with Path(path).open("xb") as file:
                            regions[name]["crop"].save(file, format="PNG")
                        self.draft = self._build_manifest(False)
                    except (OSError, ValueError) as exc:
                        self._error(exc)
                        return
                    self.triposg_crop = path
                    preview.destroy()
                    self.destroy()

                ttk.Button(row, text="Continuar en TripoSG…", command=send_crop).pack(side="left", padx=4)
            def save():
                palette = {name: list(color) for name, color in colors.items() if name != "background"}
                used = {tuple(color): name for name, color in palette.items()}
                if any((name in palette and tuple(color) != tuple(palette[name]))
                       or (tuple(color) in used and used[tuple(color)] != name)
                       for name, color in self.palette.items()):
                    self._error(ValueError("La paleta existente entra en conflicto con la segmentación ONNX."))
                    return
                path = filedialog.asksaveasfilename(parent=preview, defaultextension=".png",
                                                     filetypes=[("Máscara RGB", "*.png")])
                if path:
                    try:
                        reference_paths = {Path(state["image"]).resolve() for state in self.views.values()
                                           if "image" in state}
                        if Path(path).resolve() in reference_paths:
                            raise ValueError("Guarda la máscara en un archivo separado de la referencia.")
                        mask.save(path, format="PNG")
                    except (OSError, ValueError) as exc:
                        self._error(exc)
                        return
                    self.palette.update(palette)
                    self.views[view]["mask"] = path
                    self._invalidate_comparison()
                    self.status.set(f"Máscara multiclase aceptada para {VIEW_LABELS[view]}.")
                    preview.destroy()
            ttk.Button(preview, text="Guardar y usar máscara", command=save).pack(pady=(0, 8))
            self.wait_window(preview)
            if self.winfo_exists():
                self.grab_set()
        if external is not None:
            self._work("Leyendo las máscaras externas de See-through V2…",
                       lambda: import_see_through_v2(external, image.size, include_eyes=True), accept)
            return
        if run_see_through:
            path = self.views[view]["image"]
            resolution = image.size
            self._work("Segmentando en CUDA con See-through…",
                       lambda: segment_see_through_v2(path, resolution, *self.see_through,
                                                      cancelled=self.segment_cancel), accept)
            return
        model, labels = self.segmenter
        # Prompt again on the next request if the selected provider fails.
        def segment():
            return segment_anime_reference(image, model, labels)
        self.segmenter = None
        def finished(mask):
            self.segmenter = model, labels
            accept(mask)
        self._work("Segmentando la referencia con ONNX en CPU…", segment, finished)

    def _edit_palette(self):
        if self.busy:
            return
        value = simpledialog.askstring("Paleta de máscaras", 'Nombres y colores RGB en JSON. Ejemplo {"ropa":[255,0,0]}',
                                       initialvalue=json.dumps(self.palette, ensure_ascii=False), parent=self)
        if value is not None:
            try:
                palette = json.loads(value)
                if not isinstance(palette, dict):
                    raise ValueError("Usa un objeto JSON para la paleta.")
                self.palette = palette
                self._invalidate_comparison()
            except ValueError as exc:
                self._error(exc)

    def _change_view(self, _event=None):
        self.next_calibration = None
        view = self.views[self.view]
        self.profile_toggle.state(["!disabled" if self.view == "side" else "disabled"])
        if not self.busy:
            self.status.set(f"{self.view_label.get()} · {'calibrada' if 'projection' in view else 'sin calibrar'}")
        self._refresh_points(self._selected())

    def _reverse_side(self):
        view = self.views["side"]
        if self.busy or self.view != "side":
            self.side.set(not self.side.get())
            return
        self.side_manual = True
        if "calibration" in view:
            self._apply_calibration(*view["calibration"])
        elif "projection" in view:
            view.pop("projection")
            self.status.set("Marca de nuevo suelo y coronilla para el perfil opuesto.")
        self._redraw()

    def _start_calibration(self):
        if self.busy or self.template is None or self.view not in self.images:
            return
        self.next_calibration = ()
        self.status.set("Marca el centro del cuerpo en el suelo, a la altura de las plantas.")

    def _apply_calibration(self, center, crown):
        projection = orthographic_projection(
            self.view, center, center[1], crown, self.template.bounds,
            side=-1 if self.side.get() else 1,
        )
        view = self.views[self.view]
        view["calibration"] = [center, crown]
        view["projection"] = projection.tolist()
        self._invalidate_comparison()
        self.status.set(f"{self.view_label.get()} calibrada. Revisa la alineación y los puntos.")

    def _selected(self):
        selection = self.points.curselection()
        return int(selection[0]) if selection else None

    def _refresh_points(self, selected=None):
        points = self.views[self.view]["points"]
        self.points.delete(0, "end")
        for index, vertex in enumerate(self.indices):
            point = points.get(vertex)
            if point is None:
                status = "pendiente"
            elif point[0] is None:
                status = "oculto"
            else:
                status = "sugerido" if point[1] else "marcado"
            self.points.insert("end", f"{index + 1:02d}  Vértice {vertex} · {status}")
        if selected is not None and 0 <= selected < len(self.indices):
            self.points.selection_set(selected)
            self.points.see(selected)
        self._point_selected()

    def _point_selected(self, _event=None):
        choice = self._selected()
        pose_id = self.pose_links.get(self.indices[choice]) if choice is not None else None
        self.pose_label.set(next((name for name, index in POSE_LABELS.items() if index == pose_id), ""))
        self._redraw()

    def _clear_point(self):
        selected = self._selected()
        if selected is not None and not self.busy:
            self.views[self.view]["points"][self.indices[selected]] = (None, False)
            self._invalidate_comparison()
            self._refresh_points(selected)

    def _remove_point(self):
        selected = self._selected()
        if selected is None or self.busy:
            return
        vertex = self.indices.pop(selected)
        self.pose_links.pop(vertex, None)
        for view in self.views.values():
            view["points"].pop(vertex, None)
        self._invalidate_comparison()
        self._refresh_points(min(selected, len(self.indices) - 1))

    def _link_pose(self, _event=None):
        selected = self._selected()
        if selected is not None and not self.busy:
            vertex = self.indices[selected]
            joint = POSE_LABELS.get(self.pose_label.get())
            if self.pose_links.get(vertex) == joint:
                return
            if joint is None:
                self.pose_links.pop(vertex, None)
            else:
                self.pose_links[vertex] = joint
            for view in self.views.values():
                if view["points"].get(vertex, (None, False))[1]:
                    view["points"].pop(vertex)
            self._invalidate_comparison()
            self._refresh_points(selected)

    def _image_click(self, event):
        if self.image_frame is None or self.busy:
            return
        x, y, scale = self.image_frame
        point = [(event.x - x) / scale, (event.y - y) / scale]
        width, height = self.images[self.view].size
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            return
        if self.next_calibration is not None:
            if not self.next_calibration:
                self.next_calibration = point
                self.status.set("Marca el punto más alto de la coronilla.")
                return
            try:
                self._apply_calibration(self.next_calibration, point[1])
            except ValueError as exc:
                self._error(exc)
                return
            self.next_calibration = None
        elif self._selected() is not None:
            self.views[self.view]["points"][self.indices[self._selected()]] = (point, False)
            self._invalidate_comparison()
        self._refresh_points(self._selected())

    def _template_click(self, event):
        if self.template_pixels is None or self.busy:
            return
        pixels, depth = self.template_pixels
        distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
        closest = distances.min()
        if closest > 12:
            return
        candidates = np.flatnonzero(distances <= closest + 2)
        vertex = int(candidates[np.argmax(depth[candidates])])
        if vertex not in self.indices:
            self.indices.append(vertex)
            self._invalidate_comparison()
        self._refresh_points(self.indices.index(vertex))

    def _redraw(self, _event=None):
        if not hasattr(self, "points"):
            return
        image_canvas, template_canvas = self.image_canvas, self.template_canvas
        for canvas in (image_canvas, template_canvas):
            canvas.delete("all")
        self.image_frame = self.template_pixels = None
        view = self.views[self.view]
        if self.view in self.images:
            image = self.images[self.view]
            width, height = image_canvas.winfo_width(), image_canvas.winfo_height()
            scale = min(width / image.width, height / image.height)
            x, y = (width - image.width * scale) / 2, (height - image.height * scale) / 2
            self.image_frame = x, y, scale
            self.photo["image"] = ImageTk.PhotoImage(image.resize((
                max(1, round(image.width * scale)), max(1, round(image.height * scale)),
            ), Image.Resampling.LANCZOS))
            image_canvas.create_image(x, y, image=self.photo["image"], anchor="nw")
            if "projection" in view:
                matrix = np.asarray(view["projection"])
                bounds = self.template.bounds
                mid = bounds.mean(axis=0)
                center = (matrix[0, :3] @ mid + matrix[0, 3]) * scale + x
                levels = matrix[1, 1] * np.linspace(*bounds[:, 1], 5) + matrix[1, 3]
                for level in levels:
                    pos = y + level * scale
                    image_canvas.create_line(x, pos, width - x, pos, fill="#00d9ff", dash=(4, 4))
                image_canvas.create_line(center, y, center, height - y, fill="#00d9ff", dash=(4, 4))
                if self.overlay.get():
                    self._draw_projected_points(image_canvas, view, matrix, x, y, scale)
            for order, vertex in enumerate(self.indices):
                point, suggested = view["points"].get(vertex, (None, False))
                if point is not None:
                    px, py = point
                    self._draw_point(image_canvas, x + px * scale, y + py * scale, order, suggested)
        if self.comparison_mode.get() and self.view in self.comparisons:
            comparison = self.comparisons[self.view].copy()
            comparison.thumbnail((max(1, template_canvas.winfo_width()), max(1, template_canvas.winfo_height())),
                                 Image.Resampling.LANCZOS)
            self.photo["comparison"] = ImageTk.PhotoImage(comparison)
            template_canvas.create_image(
                template_canvas.winfo_width() / 2, template_canvas.winfo_height() / 2,
                image=self.photo["comparison"],
            )
        elif self.template is not None:
            width, height = template_canvas.winfo_width(), template_canvas.winfo_height()
            bounds = self.template.bounds
            axis = 2 if self.view == "side" else 0
            extent = self.template.extents
            pixel_height = max(1, min(height - 24, (width - 24) * extent[1] / max(extent[axis], 1e-8)))
            matrix = orthographic_projection(
                self.view, [width / 2, 0], (height + pixel_height) / 2,
                (height - pixel_height) / 2, bounds, side=-1 if self.side.get() else 1,
            )
            vertices = np.asarray(self.template.vertices)
            pixels = vertices @ matrix[:, :3].T + matrix[:, 3]
            depth = vertices @ -np.cross(*matrix[:, :3])
            self.template_pixels = pixels, depth
            picture = Image.new("RGB", (width, height), "#11141b")
            ImageDraw.Draw(picture).point([tuple(p) for p in pixels], fill="#8a99ab")
            self.photo["template"] = ImageTk.PhotoImage(picture)
            template_canvas.create_image(0, 0, image=self.photo["template"], anchor="nw")
            for order, vertex in enumerate(self.indices):
                self._draw_point(template_canvas, *pixels[vertex], order)

    def _draw_point(self, canvas, x, y, order, suggested=False):
        color = "#ff268f" if self._selected() == order else "#ffd767" if suggested else "#00d9ff"
        canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="#11141b")
        canvas.create_text(x + 8, y - 8, text=order + 1, fill=color, anchor="w")

    def _draw_projected_points(self, canvas, view, matrix, x, y, scale):
        pixels = self.template.vertices[self.indices] @ matrix[:, :3].T + matrix[:, 3]
        width, height = self.images[self.view].size
        for vertex, (px, py) in zip(self.indices, pixels):
            if not (0 <= px < width and 0 <= py < height):
                continue
            cx, cy = x + px * scale, y + py * scale
            canvas.create_rectangle(cx - 3, cy - 3, cx + 3, cy + 3, outline="#f3f5f8")
            point, _ = view["points"].get(vertex, (None, False))
            if point is not None:
                qx, qy = point
                canvas.create_line(cx, cy, x + qx * scale, y + qy * scale, fill="#a5aeb9", dash=(2, 3))

    def _suggest(self):
        view = self.view
        if self.busy or view not in self.images or not self.pose_links:
            self.status.set("Asocia al menos un vértice de plantilla con una articulación MediaPipe.")
            return
        image = self.views[view]["image"]
        def apply_suggestions(found):
            points = self.views[view]["points"]
            additions = {
                vertex: (found[joint], True) for vertex, joint in self.pose_links.items()
                if joint in found and vertex not in points
            }
            points.update(additions)
            if additions:
                self._invalidate_comparison()
            self._refresh_points(self._selected())
            self.status.set(f"{len(additions)} sugerencias en {VIEW_LABELS[view]}. Corrige con un clic.")
        self._work("Detectando puntos con MediaPipe…",
                   lambda: pose_points_for_view(suggest_anatomical_points(image), view), apply_suggestions)

    def _invalidate_comparison(self):
        self.comparisons.clear()
        self.comparison_mode.set(False)

    def _auto_align(self, *, use_isnet=False):
        if self.busy or self.template is None or not self.images:
            return
        sources = {
            name: (image, self.views[name]["image"], self.views[name].get("body_mask"))
            for name, image in self.images.items()
        }
        def detect():
            results = {}
            segmenter = AnimeSilhouette() if use_isnet and any(not row[2] for row in sources.values()) else None
            for name, (image, source, mask_path) in sources.items():
                try:
                    found = pose_points_for_view(suggest_anatomical_points(source), name)
                except (OSError, ImportError, RuntimeError, ValueError):
                    found = {}
                mask = None
                if mask_path:
                    with Image.open(mask_path) as loaded:
                        mask = loaded.convert("RGB")
                    if mask.size != image.size:
                        raise ValueError(f"La silueta de {VIEW_LABELS[name]} debe coincidir con la referencia.")
                elif segmenter is not None:
                    mask = segmenter.predict(image)
                report = foreground_crop_diagnostics(mask) if mask is not None else None
                results[name] = silhouette_calibration(reference_silhouette(image, mask), found), found, report
            return results
        def apply(results):
            aligned = added = 0
            issues = []
            pose_evidence = []
            edge_labels = {"top": "superior", "bottom": "inferior",
                           "left": "izquierdo", "right": "derecho"}
            for name, (calibration, found, report) in results.items():
                view = self.views[name]
                if report is not None:
                    edges = [edge_labels[edge] for edge, pixels in report["edge_pixels"].items() if pixels]
                    if edges:
                        issues.append(f"{VIEW_LABELS[name]} toca el borde ({', '.join(edges)})")
                    if report["secondary_fraction"] >= .02:
                        issues.append(f"{VIEW_LABELS[name]} tiene {report['secondary_fraction']:.0%} de área aislada")
                if calibration and "projection" not in view:
                    if name == "side" and not self.side_manual:
                        opposite = profile_opposite(found, self.images[name].width)
                        if opposite is not None:
                            self.side.set(opposite)
                    center, crown = calibration
                    view["calibration"] = [center, crown]
                    view["projection"] = orthographic_projection(
                        name, center, center[1], crown, self.template.bounds,
                        side=-1 if self.side.get() else 1,
                    ).tolist()
                    aligned += 1
                for vertex, joint in self.pose_links.items():
                    if joint in found and vertex not in view["points"]:
                        view["points"][vertex] = (found[joint], True)
                        added += 1
                pose_evidence.append((view.get("calibration", calibration), found))
            disagreements = vertical_pose_disagreements(pose_evidence)
            if disagreements:
                issues.append(f"{len(disagreements)} puntos con alturas distintas entre vistas")
            if aligned or added:
                self._invalidate_comparison()
            self._refresh_points(self._selected())
            summary = f"{aligned} vistas alineadas · {added} puntos sugeridos."
            self.status.set(" · ".join([summary, *issues, "Revisa o calibra manualmente."]))
        self._work("Analizando siluetas y puntos anatómicos…", detect, apply)

    def _build_manifest(self, require_ready):
        if not self.template_path:
            raise ValueError("Carga una plantilla corporal.")
        indices = self.indices.copy()
        bundled = self.template_path == str(BUNDLED_BODY)
        active = {name for name, state in self.views.items() if "image" in state}
        partial = 0 < len(active) < 3
        may_infer = partial or self.infer_occlusion_depth.get()
        if require_ready and (bundled or partial):
            visible = {
                name: {vertex for vertex, (point, _) in state["points"].items() if point is not None}
                for name, state in self.views.items()
            }
            usable = set().union(*visible.values()) if may_infer else visible["side"] & (visible["front"] | visible["back"])
            indices = [vertex for vertex in indices if vertex in usable or
                       (not may_infer and vertex not in POSE_VERTICES.values())]
        views = {}
        for name, state in self.views.items():
            if require_ready and partial and name not in active:
                continue
            if require_ready and ("image" not in state or "projection" not in state):
                raise ValueError(f"Carga y calibra la vista {VIEW_LABELS[name].lower()}.")
            view = {key: deepcopy(value) for key, value in state.items() if key != "points"}
            view["landmarks"] = [
                state["points"].get(vertex, (None, False))[0] or [None, None] for vertex in indices
            ]
            view["suggested"] = [vertex for vertex, (_, proposed) in state["points"].items() if proposed]
            view["hidden"] = [vertex for vertex, (point, _) in state["points"].items() if point is None]
            views[name] = view
        return {
            "template": self.template_path, "vertex_indices": indices, "views": views,
            "palette": deepcopy(self.palette), "pose_links": self.pose_links.copy(),
            "side_opposite": self.side.get(),
            "side_manual": self.side_manual,
            "profile": self.profile.get(),
            "bundled_template": bundled,
            "allow_inferred_depth": may_infer,
            "export_rig": self.export_rig.get(),
            "export_hybrid": self.export_hybrid.get(),
            "smoothing": .008 if bundled else 0,
            "proportions": {key: float(var.get().replace(",", ".")) for key, var in self.factors.items()},
        }

    def _save_setup(self):
        if self.busy:
            return
        try:
            data = self._build_manifest(False)
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", filetypes=[("Ajuste", "*.json")])
            if path:
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            self._error(exc)

    def _open_setup(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self, filetypes=[("Ajuste", "*.json")])
        if not path:
            return
        try:
            self._restore_setup(Path(path))
        except Exception as exc:
            self._error(exc)

    def _restore_setup(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.load_setup(data, path.parent)

    def load_setup(self, data, root=Path(".")):
        template_path = str(BUNDLED_BODY if data.get("bundled_template")
                            else (root / data["template"]).resolve())
        template = load_body_template(template_path)
        indices = data["vertex_indices"]
        if (template.extents[1] <= 0 or len(indices) != len(set(indices))
                or any(type(index) is not int or not 0 <= index < len(template.vertices) for index in indices)):
            raise ValueError("La plantilla o sus índices no son válidos.")
        views, images = {}, {}
        for name in VIEWS:
            view = dict(data["views"].get(name, {}))
            landmarks = view.pop("landmarks", [[None, None] for _ in indices])
            if len(landmarks) != len(indices):
                raise ValueError("El número de puntos no coincide con la plantilla.")
            for point in landmarks:
                if point != [None, None] and (not isinstance(point, list) or len(point) != 2
                        or any(type(value) not in (int, float) or not np.isfinite(value) for value in point)):
                    raise ValueError("Cada punto debe tener dos coordenadas finitas o estar oculto.")
            suggested = set(view.pop("suggested", []))
            hidden = set(view.pop("hidden", []))
            view["points"] = {
                vertex: (None, False) if point == [None, None] else (point, vertex in suggested)
                for vertex, point in zip(indices, landmarks) if point != [None, None] or vertex in hidden
            }
            for asset in ("image", "mask", "body_mask"):
                if asset in view:
                    view[asset] = str((root / view[asset]).resolve())
            if "image" in view:
                with Image.open(view["image"]) as image:
                    images[name] = image.convert("RGBA")
            if "projection" in view:
                projection = np.asarray(view["projection"], dtype=float)
                if projection.shape != (2, 4) or not np.isfinite(projection).all():
                    raise ValueError("La proyección contiene valores inválidos.")
            if "calibration" in view:
                center, crown = view["calibration"]
                orthographic_projection(name, center, center[1], crown, template.bounds)
            views[name] = view
        pose_links = {int(vertex): joint for vertex, joint in data.get("pose_links", {}).items()}
        self.template_path, self.template = template_path, template
        self.indices, self.views, self.images = indices, views, images
        self.palette = data.get("palette", {})
        profile = data.get("profile")
        self.profile.set(profile if template_path == str(BUNDLED_BODY) and profile in BODY_PROFILES else "Personalizado")
        self.pose_links = {vertex: joint for vertex, joint in pose_links.items() if vertex in indices}
        self.side.set(data.get("side_opposite", views["side"].get("projection", [[0, 0, -1]])[0][2] > 0))
        self.side_manual = data.get("side_manual", "side_opposite" in data)
        self.infer_occlusion_depth.set(data.get("allow_inferred_depth") is True)
        self.export_rig.set(data.get("export_rig") is True)
        self.export_hybrid.set(data.get("export_hybrid") is True)
        for key, var in self.factors.items():
            var.set(data.get("proportions", {}).get(key, 1))
        self._invalidate_comparison()
        self._change_view()

    def _fit(self):
        if self.busy:
            return
        try:
            data = self._build_manifest(True)
            if len(data["vertex_indices"]) < 4:
                raise ValueError("Marca al menos cuatro vértices en la plantilla y sus puntos homólogos.")
        except ValueError as exc:
            self._error(exc)
            return
        if data["allow_inferred_depth"] and not messagebox.askyesno(
                "Reconstrucción parcial",
                "Las direcciones sin observación conservarán la profundidad de la plantilla. ¿Continuar con un ajuste parcial?",
                parent=self):
            return
        if self.export_rig.get():
            data["export_rig"] = True
        if self.visual_hull.get():
            categories = tuple(name for name in data.get("palette", {}) if name not in {"body", "face", "eyes"})
            if not categories:
                self._error(ValueError("Carga máscaras semánticas con una categoría de componente independiente."))
                return
            label = simpledialog.askstring("Volumen de componente", ", ".join(categories),
                                           initialvalue=categories[0], parent=self)
            if label is None:
                return
            if label not in categories:
                self._error(ValueError("Selecciona una categoría presente en las tres perspectivas."))
                return
            data["visual_hull_class"] = label
        if any(view["suggested"] for view in data["views"].values()) and not messagebox.askyesno(
                "Revisar puntos sugeridos", "¿Confirmas que revisaste los puntos amarillos de las tres vistas?",
                parent=self):
            return
        def solve():
            report = {}
            mesh, patches, errors = reconstruct_data(data, deformation_report=report)
            comparisons = {}
            for name, view in data["views"].items():
                mask = None
                if view.get("body_mask"):
                    with Image.open(view["body_mask"]) as source:
                        mask = source.convert("RGB")
                comparisons[name], iou = compare_silhouette(
                    mesh, self.images[name], view["projection"], mask,
                )
                report.setdefault("silhouette_iou", {})[name] = iou
                report.setdefault("silhouette_kind", {})[name] = "body" if mask is not None else "reference"
            return mesh, patches, report, errors, comparisons
        self._work("Ajustando y validando la geometría en CPU…", solve, lambda result: self._review(data, result))

    def _review(self, data, result):
        mesh, patches, report, errors, comparisons = result
        self.comparisons = comparisons
        self.comparison_mode.set(True)
        self._redraw()
        issues = blocking_geometry_issues(report)
        if issues:
            self.status.set("Exportación bloqueada. Revisa la silueta o corrige el ajuste.")
            messagebox.showerror("Geometría rechazada", "\n".join(issues), parent=self)
            return
        summary = "\n".join(f"{VIEW_LABELS[name]} {error:.2f} px" for name, error in errors.items())
        closed = "sí" if report["watertight"] else "no"
        winding = "sí" if report["winding_consistent"] else "no"
        self.status.set("Validación terminada. Puedes guardar o corregir el ajuste.")
        silhouette = ", ".join(
            f"{VIEW_LABELS[name]} ({'corporal' if report['silhouette_kind'][name] == 'body' else 'referencia'}) {value:.0%}"
            for name, value in report["silhouette_iou"].items()
            if value is not None
        ) or "sin primer plano fiable"
        evidence = report["reference_evidence"]
        if not messagebox.askyesno("Validación del cuerpo", (
            f'{report["vertices"]:,} vértices / {report["faces"]:,} caras · Cerrado {closed}\n'
            f"Orientación consistente {winding}\n"
            f'Autointersecciones detectadas {report["intersection_pairs_up_to_100"]} (máximo 100)\n'
            f'Caras con área extrema {report["extreme_area_faces"]}\n'
            f'Normales con giro mayor de 90° {report["reflected_normal_faces"]}\n'
            f'Simetría X, percentil 95 {report["symmetry_p95_height"]:.2%} de la altura\n\n'
            f"Reproyección antes de estilizar\n{summary}\n"
            f"IoU de siluetas · {silhouette}\n"
            f"Vistas disponibles {len(evidence['observed_views'])} de 3 · "
            f"Puntos con profundidad de plantilla {evidence['inferred_anchor_depths']}\n"
            "Una silueta de referencia puede incluir ropa y cabello.\n"
            "Verde coincide · rosa solo referencia · celeste solo modelo.\n\n¿Guardar el cuerpo completo?"
        ), parent=self):
            return
        folder = filedialog.askdirectory(parent=self, title="Carpeta de salida Genesis")
        if folder:
            self._work("Guardando GLB y validación…",
                       lambda: export_studio_reconstruction(folder, data, mesh, patches, report, errors, comparisons),
                       self._finish)

    def _finish(self, outputs):
        self.result = outputs
        self.destroy()

    def _work(self, status, operation, done):
        self.busy = True
        self.profile_combo.state(["disabled"])
        self.status.set(status)
        queue = Queue()
        def run():
            try:
                queue.put((True, operation()))
            except Exception as exc:
                queue.put((False, exc))
        def check():
            try:
                ok, result = queue.get_nowait()
            except Empty:
                self.after(60, check)
                return
            self.busy = False
            self.profile_combo.state(["!disabled"])
            if ok:
                done(result)
            else:
                self._error(result)
        Thread(target=run, daemon=True).start()
        self.after(60, check)

    def _error(self, exc):
        self.status.set(str(exc))
        messagebox.showerror("Reconstrucción anatómica", str(exc), parent=self)
