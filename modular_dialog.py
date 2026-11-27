"""Scene component editor for the Genesis modular avatar workflow."""

from pathlib import Path
from queue import Empty, Queue
import shutil
from tempfile import TemporaryDirectory
from threading import Thread
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
import tkinter as tk

import numpy as np
from PIL import Image

from aethryx_anatomy import blocking_geometry_issues, load_body_template, validate_deformation
from aethryx_anime import hair_locks, nikix_preset, sculpted_body, sculpted_face_parts
from aethryx_bridge import inspect_avatar_glb
from aethryx_materials import (CHANNELS, RESOLUTIONS, color_gradient, save_pbr_maps, transform_uv,
                              unwrap_uv, update_pbr_maps, uv_quality)
from aethryx_rig import export_rigged_avatar, preview_expression, preview_pose
from pose_dialog import PoseDialog
from aethryx_modular import (
    BODY, BODY_PREVIEW_BANDS, EYES, FACE_CONTROLS, ModularAvatar, anime_body, apply_material, avatar_eye,
    body_contacts, facial_vertices, fit_component_to_patch, fit_footwear, fit_garment_with_clearance,
    fit_hair, fitted_eyelids, head_frame,
    mouth_components, recolor_iris, regional_hair, write_new_glb,
)


class ComponentDialog(tk.Toplevel):
    def __init__(self, master, scene, hidden):
        super().__init__(master)
        self.title("Genesis · Componentes del avatar")
        self.geometry("1040x650")
        self.minsize(880, 540)
        self.transient(master)
        self.grab_set()
        self.result = None
        self.avatar = ModularAvatar(scene, hidden)
        self.edited = False
        self.visibility_changed = False
        self.busy = False
        self.buttons = []
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="Selecciona una pieza para editarla o crea un cuerpo anime.")
        actions = ttk.Frame(self, padding=10)
        actions.pack(fill="x")
        for row, buttons in enumerate((
            (("Añadir GLB/OBJ…", self.add), ("Sustituir…", self.replace),
             ("Quitar", self.remove), ("Ocultar / mostrar", self.visibility), ("Guardar pieza…", self.save_piece)),
            (("Crear cuerpo anime", self.create_body), ("Rostro…", self.face), ("Crear ojos", self.eyes),
             ("Cabello…", self.hair), ("Prenda superior…", self.clothing)),
            (("Color / iris…", self.material), ("UV y textura…", self.texture), ("Comprobar ajuste", self.contacts),
             ("Degradado por pieza…", self.gradient), ("Calzado…", self.footwear)),
            (("Perfil anime…", self.create_profile), ("Párpados", self.eyelids), ("Interior bucal", self.mouth),
             ("Cabello por regiones…", self.hair_regions), ("Ocultar región corporal…", self.toggle_body_band)),
            (("Cuerpo anime v2…", self.create_sculpted_body), ("Mechones independientes…", self.independent_hair)),
            (("Preset Nikix v3", lambda: self.create_nikix(3)), ("Preset Nikix v4", self.create_nikix),
             ("Ajustar malla a pieza…", self.fit_generated)),
            (("Mapas PBR…", self.pbr), ("Diagnóstico UV…", self.diagnose_uv), ("Ajustar UV…", self.edit_uv),
             ("Generar atlas UV", self.generate_uv)),
            (("Guardar conjunto…", self.save_outfit), ("Abrir conjunto…", self.load_outfit)),
            (("Exportar rig GLB…", self.export_rig), ("Guardar pose de prueba…", self.export_pose),
             ("Guardar expresión de prueba…", self.export_expression), ("Revisar rig GLB…", self.inspect_rig),
             ("Vista de pose…", self.pose_window)),
        )):
            for column, (label, command) in enumerate(buttons):
                button = ttk.Button(actions, text=label, command=command)
                button.grid(row=row, column=column, sticky="ew", padx=3, pady=3)
                self.buttons.append(button)
        self.tree = ttk.Treeview(self, columns=("visible", "faces"), show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Componente")
        self.tree.heading("visible", text="Visible")
        self.tree.heading("faces", text="Caras")
        self.tree.column("#0", width=320)
        self.tree.column("visible", width=110)
        self.tree.column("faces", width=100)
        self.tree.pack(fill="both", expand=True, padx=10)
        footer = ttk.Frame(self, padding=10)
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, wraplength=510).pack(side="left")
        apply_button = ttk.Button(footer, text="Aplicar a Studio…", command=self.apply)
        apply_button.pack(side="right")
        self.buttons.append(apply_button)
        self._refresh()

    def _refresh(self, selected=None):
        self.tree.delete(*self.tree.get_children())
        for node in self.avatar.nodes:
            geometry = self.avatar.scene.graph[node][1]
            mesh = self.avatar.scene.geometry[geometry]
            self.tree.insert("", "end", iid=node, text=node, values=(
                "No" if node in self.avatar.hidden else "Sí", len(mesh.faces),
            ))
        if selected in self.avatar.nodes:
            self.tree.selection_set(selected)

    def _selected(self):
        selected = self.tree.selection()
        if not selected:
            self.status.set("Selecciona un componente.")
        return selected[0] if selected else None

    def _edit(self, operation, node=None):
        try:
            result = operation()
        except Exception as exc:
            messagebox.showerror("Componentes", str(exc), parent=self)
            return
        self.edited = True
        self.status.set(result or "Cambio preparado.")
        self._refresh(node)

    def close(self):
        if not self.busy:
            self.destroy()

    def _edit_background(self, operation, node=None, *, updates_scene=True):
        self.busy = True
        for button in self.buttons:
            button.state(["disabled"])
        self.status.set("Validando geometría en CPU…")
        queue = Queue()

        def run():
            try:
                queue.put((True, operation()))
            except Exception as exc:
                queue.put((False, str(exc)))

        def poll():
            try:
                success, result = queue.get_nowait()
            except Empty:
                self.after(60, poll)
                return
            self.busy = False
            for button in self.buttons:
                button.state(["!disabled"])
            if success:
                self.edited |= updates_scene
                self.status.set(result or "Cambio validado.")
                self._refresh(node)
            else:
                self.status.set("Cambio descartado.")
                messagebox.showerror("Componentes", result, parent=self)

        Thread(target=run, daemon=True).start()
        self.after(60, poll)

    def add(self):
        path = self._piece_path()
        if path:
            self._edit(lambda: self.avatar.import_file(path), Path(path).stem)

    def replace(self):
        node = self._selected()
        if node:
            path = self._piece_path()
            if path:
                self._edit(lambda: self.avatar.import_file(path, node), node)

    def _piece_path(self):
        return filedialog.askopenfilename(parent=self, filetypes=[("Malla", "*.glb *.obj *.ply *.stl")])

    def fit_generated(self):
        target = self._selected()
        if not target or target == BODY:
            self.status.set("Selecciona el parche o componente que servirá de guía.")
            return
        path = self._piece_path()
        if not path:
            return
        node = f"{target}-ajuste"

        def align():
            candidate, report = fit_component_to_patch(self._body(), self.avatar.get(target, world=True),
                                                       load_body_template(Path(path)))
            self.avatar.add(node, candidate)
            return f"Ajuste preliminar. Contactos {report['contacts']}, vértices dentro {report['inside']}."

        self._edit_background(align, node)

    def remove(self):
        node = self._selected()
        if node:
            self._edit(lambda: self.avatar.remove(node))

    def visibility(self):
        node = self._selected()
        if node:
            self.avatar.hidden.symmetric_difference_update({node})
            self.visibility_changed = True
            self._refresh(node)

    def save_piece(self):
        node = self._selected()
        if not node:
            return
        path = self._save_path("Guardar pieza como GLB", node)
        if path:
            try:
                self.avatar.export_piece(node, path)
                self.status.set(f"Pieza guardada en {path}")
            except Exception as exc:
                messagebox.showerror("Guardar pieza", str(exc), parent=self)

    def _save_path(self, title, stem):
        return filedialog.asksaveasfilename(parent=self, title=title, initialfile=f"{stem}.glb",
                                            defaultextension=".glb", filetypes=[("GLB", "*.glb")])

    def _body(self, world=True):
        if BODY not in self.avatar.nodes:
            raise ValueError("Crea un cuerpo anime o abre un cuerpo reconstruido por Genesis.")
        return self.avatar.get(BODY, world=world)

    def create_body(self):
        self._edit(lambda: self.avatar.add(BODY, anime_body()), BODY)

    def create_profile(self):
        profile = simpledialog.askstring("Perfil corporal anime", "Elige neutro, femenino o masculino.",
                                         initialvalue="femenino", parent=self)
        if profile is not None:
            self._edit(lambda: self.avatar.add(BODY, anime_body("Anime " + profile.strip().lower())), BODY)

    def create_sculpted_body(self):
        profile = simpledialog.askstring("Cuerpo anime v2", "Elige neutro, femenino o masculino.",
                                         initialvalue="femenino", parent=self)
        if profile is None:
            return
        def generate():
            if BODY in self.avatar.nodes:
                raise ValueError("Abre una escena nueva para conservar el cuerpo base actual.")
            body = sculpted_body("Anime " + profile.strip().lower())
            issues = blocking_geometry_issues(validate_deformation(body, body))
            if issues:
                raise ValueError("\n".join(issues))
            self._group({BODY: body, **sculpted_face_parts(body)})
            return "Cuerpo con cabeza procedural, ojos, párpados, pestañas y cejas."
        self._edit_background(generate, BODY)

    def independent_hair(self):
        length = simpledialog.askfloat(
            "Mechones de cabello", "Longitud posterior de 0 a 1.", minvalue=0, maxvalue=1,
            initialvalue=.45, parent=self,
        )
        if length is None:
            return
        _, root = colorchooser.askcolor(color="#291d40", title="Raíces", parent=self)
        if not root:
            return
        _, tip = colorchooser.askcolor(color="#a45483", title="Puntas", parent=self)
        if tip:
            def generate():
                pieces = hair_locks(self._body(), length, root, tip)
                old_hair = {name for name in self.avatar.nodes if name.startswith("cabello")}
                self._group(pieces, old_hair - pieces.keys())
            self._edit(generate)

    def create_nikix(self, version=4):
        def generate():
            if self.avatar.nodes:
                raise ValueError("Abre una escena vacía para generar el preset completo.")
            parts = nikix_preset(version=version)
            issues = blocking_geometry_issues(validate_deformation(parts[BODY], parts[BODY]))
            if issues:
                raise ValueError("\n".join(issues))
            self._group(parts)
            return f"Preset procedural Nikix v{version} generado con prendas y accesorios independientes."

        self._edit_background(generate, BODY)

    def _group(self, pieces, hide=()):
        working = ModularAvatar(self.avatar.scene, self.avatar.hidden)
        for name, piece in pieces.items():
            if name in working.nodes:
                transform, _ = working.scene.graph[name]
                piece.apply_transform(np.linalg.inv(transform))
            working.upsert(name, piece)
        working.hidden.update(name for name in hide if name in working.nodes)
        self.avatar = working

    def eyelids(self):
        def generate():
            if any(node not in self.avatar.nodes for node in EYES):
                raise ValueError("Genera primero ambos ojos.")
            pieces = {f"{eye}-parpado-{name}": piece for eye in EYES
                      for name, piece in fitted_eyelids(self.avatar.get(eye, world=True)).items()}
            self._group(pieces)
        self._edit(generate)

    def mouth(self):
        self._edit(lambda: self._group(mouth_components(self._body())), "boca-cavidad")

    def hair_regions(self):
        values = self._numbers("Cabello por regiones", "Longitud 0–1 y volumen 0,02–0,15.", "0.35 0.055", 2)
        if values is None:
            return
        _, root = colorchooser.askcolor(color="#281a3a", title="Color de raíz", parent=self)
        if not root:
            return
        _, tip = colorchooser.askcolor(color="#99568c", title="Color de puntas", parent=self)
        if tip:
            self._edit(lambda: self._group(regional_hair(self._body(), *values, root, tip), {"cabello"}))

    def _numbers(self, title, description, initial, count):
        entry = simpledialog.askstring(title, description, initialvalue=initial, parent=self)
        if entry is None:
            return None
        try:
            values = [float(text.replace(",", ".")) for text in entry.split()]
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError
            return values
        except ValueError:
            messagebox.showerror(title, f"Introduce {count} números finitos separados por espacios.", parent=self)
            return None

    def face(self):
        factors = self._numbers(
            "Ajustes relativos del rostro",
            f"Factores 0,75–1,25 en este orden. {', '.join(FACE_CONTROLS.values())}",
            " ".join(["1"] * len(FACE_CONTROLS)), len(FACE_CONTROLS),
        )
        if factors is None:
            return
        factors = dict(zip(FACE_CONTROLS, factors))
        def update():
            working = ModularAvatar(self.avatar.scene, self.avatar.hidden)
            original = self._body(world=False)
            body = original.copy()
            body.vertices = facial_vertices(original, factors)
            issues = blocking_geometry_issues(validate_deformation(original, body))
            if issues:
                raise ValueError("\n".join(issues))
            body.metadata["aethryx_face_edits"] = [*original.metadata.get("aethryx_face_edits", []), factors]
            working.replace(BODY, body)
            frame = head_frame(self._body())
            center = frame[:, 0].mean()
            for node in EYES:
                if node not in working.nodes:
                    continue
                transform, _ = working.scene.graph[node]
                eye = working.get(node)
                previous = eye.bounds.mean(axis=0)
                world_center = transform[:3, :3] @ previous + transform[:3, 3]
                dx = (world_center[0] - center) * (factors["eye_spacing"] - 1)
                for part in (name for name in working.nodes if name == node or name.startswith(node + "-")):
                    matrix, _ = working.scene.graph[part]
                    component = working.get(part)
                    pivot = np.linalg.solve(matrix[:3, :3], world_center - matrix[:3, 3])
                    delta = np.linalg.solve(matrix[:3, :3], [dx, 0, 0])
                    component.vertices = pivot + (component.vertices - pivot) * factors["eye_size"] + delta
                    working.replace(part, component)
            self.avatar = working
        self._edit_background(update, BODY)

    def eyes(self):
        def generate():
            body = self._body()
            if any(node in self.avatar.nodes for node in EYES):
                raise ValueError("Ya existe un ojo generado. Quita ambos antes de recrearlos.")
            pair = [avatar_eye(body, index == 0, color)
                    for index, color in enumerate(("#68a9c2", "#739baf"))]
            for node, eye in zip(EYES, pair):
                self.avatar.add(node, eye)
        self._edit(generate, EYES[0])

    def hair(self):
        values = self._numbers("Cabello", "Longitud de nuca 0–1 y volumen 0,02–0,15.", "0.35 0.055", 2)
        if values is not None:
            self._edit(lambda: self._new_fitted("cabello", fit_hair(self._body(), *values)), "cabello")

    def clothing(self):
        style = simpledialog.askstring("Estilo de prenda", "Elige top, falda o vestido.", initialvalue="top", parent=self)
        if style is None:
            return
        style = style.strip().lower()
        values = self._numbers(
            "Prenda superior", "Holgura 0,003–0,06 y espesor 0,001–0,03 de la altura.", "0.012 0.006", 2,
        )
        if values is None:
            return
        def generate():
            garment, clearance = fit_garment_with_clearance(self._body(), *values, style)
            self._new_fitted(node, garment)
            return f"Prenda cerrada con holgura {clearance:.4f}, cero contactos y cero vértices dentro del cuerpo."
        node = "prenda-superior" if style == "top" else f"prenda-{style}"
        self._edit_background(generate, node)

    def footwear(self):
        values = self._numbers(
            "Calzado", "Holgura 0,003–0,03 y espesor 0,002–0,03 de la altura.", "0.007 0.006", 2,
        )
        if values is not None:
            def generate():
                shoes = fit_footwear(self._body(), *values)
                self._group(shoes)
                return "Calzado independiente validado con cero contactos y cero vértices dentro del cuerpo."
            self._edit_background(generate, "prenda-zapato-izquierdo")

    def toggle_body_band(self):
        chosen = simpledialog.askstring("Región corporal en el visor", "Escribe torso o cadera para alternar visibilidad.",
                                         initialvalue="torso", parent=self)
        if chosen is None:
            return
        key = chosen.strip().lower()

        def update():
            if key not in BODY_PREVIEW_BANDS:
                raise ValueError("Región corporal desconocida.")
            body = self._body(world=False)
            hidden = set(body.metadata.get("aethryx_hidden_body_bands", []))
            hidden.symmetric_difference_update({key})
            body.metadata["aethryx_hidden_body_bands"] = sorted(hidden)
            self.avatar.replace(BODY, body)
            return f"{key.capitalize()} {'oculto' if key in hidden else 'visible'} en el visor."
        self._edit(update, BODY)

    def save_outfit(self):
        path = self._save_path("Guardar conjunto de prendas", "conjunto")
        if path:
            try:
                self.avatar.export_outfit(path)
                self.status.set(f"Conjunto guardado en {path}")
            except Exception as exc:
                messagebox.showerror("Guardar conjunto", str(exc), parent=self)

    def export_rig(self):
        influences = simpledialog.askinteger("Rig humanoide", "Máximo de influencias por vértice.",
                                             initialvalue=4, minvalue=1, maxvalue=4, parent=self)
        if influences is None:
            return
        path = self._save_path("Guardar GLB con rig y expresiones", "avatar-articulado")
        if path:
            def run():
                destination = Path(path)
                if destination.suffix.lower() != ".glb" or destination.exists():
                    raise ValueError("Elige un archivo GLB nuevo.")
                with TemporaryDirectory(dir=destination.parent) as folder:
                    staged = Path(folder) / "avatar.glb"
                    report = export_rigged_avatar(self.avatar.scene, staged, influences)
                    checked = inspect_avatar_glb(staged)
                    if not checked["valid_native_glb"]:
                        raise ValueError("\n".join(checked["errors"]))
                    with staged.open("rb") as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output)
                return f"GLB validado con {report['joints']} huesos, {len(report['morphs'])} morphs y un clip de prueba."
            self._edit_background(run, updates_scene=False)

    def export_pose(self):
        path = self._save_path("Guardar una pose estática para inspeccionar en Studio", "pose-de-prueba")
        if path:
            try:
                write_new_glb(preview_pose(self.avatar.scene), path)
                self.status.set(f"Pose guardada en {path}.")
            except Exception as exc:
                messagebox.showerror("Pose de prueba", str(exc), parent=self)

    def pose_window(self):
        try:
            dialog = PoseDialog(self, self.avatar.scene, self.avatar.hidden)
            self.wait_window(dialog)
            self.grab_set()
            if dialog.result is not None:
                self.avatar = ModularAvatar(dialog.result, self.avatar.hidden)
                self.edited = True
                self.status.set("Ajustes del esqueleto preparados para guardar y exportar.")
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Vista de pose", str(exc), parent=self)

    def inspect_rig(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Avatar GLB", "*.glb")])
        if path:
            try:
                report = inspect_avatar_glb(path)
                result = "Validación local del GLB superada." if report["valid_native_glb"] else "\n".join(report["errors"])
                messagebox.showinfo("Revisión nativa del GLB", (
                    f"{result}\nHuesos {report['joints']} · Caras {report['triangles']}\n"
                    f"Materiales {report['materials']} · Visemas {report['visemes']}\n"
                    "La importación en Unity y VRChat requiere una prueba aparte."
                ), parent=self)
            except (ValueError, KeyError, IndexError, OSError, TypeError) as exc:
                messagebox.showerror("Revisar rig", str(exc), parent=self)

    def export_expression(self):
        name = simpledialog.askstring(
            "Expresión de prueba", "Escribe Smile, Sad, Anger, Surprise, JawOpen, BlinkLeft, BlinkRight o un visema.",
            initialvalue="Smile", parent=self,
        )
        if name:
            path = self._save_path("Guardar expresión estática para inspeccionar en Studio", "expresion-de-prueba")
            if path:
                try:
                    write_new_glb(preview_expression(self.avatar.scene, name), path)
                    self.status.set(f"Expresión guardada en {path}.")
                except Exception as exc:
                    messagebox.showerror("Expresión de prueba", str(exc), parent=self)

    def load_outfit(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Conjunto GLB", "*.glb")])
        if path:
            self._edit(lambda: self.avatar.apply_outfit(path))

    def diagnose_uv(self):
        node = self._selected()
        if not node:
            return
        try:
            quality = uv_quality(self.avatar.get(node))
            distortion = quality["area_distortion_p95_stops"]
            messagebox.showinfo("Diagnóstico UV", (
                f"Triángulos degenerados {quality['degenerate']}\n"
                f"Vértices fuera de 0–1 {quality['outside_vertices']}\n"
                f"Solapamiento muestreado {quality['overlap_fraction']:.1%}\n"
                f"Distorsión p95 en pasos de área {round(distortion, 2) if distortion is not None else 'sin datos'}"
            ), parent=self)
        except Exception as exc:
            messagebox.showerror("Diagnóstico UV", str(exc), parent=self)

    def edit_uv(self):
        node = self._selected()
        if not node:
            return
        values = self._numbers("Transformar UV", "Escala U V y desplazamiento U V.", "1 1 0 0", 4)
        if values is not None:
            self._edit(lambda: self.avatar.replace(node, transform_uv(
                self.avatar.get(node), values[:2], values[2:])), node)

    def generate_uv(self):
        node = self._selected()
        if node == BODY:
            self.status.set("Selecciona una pieza independiente para conservar la topología cerrada del cuerpo.")
        elif node:
            def generate():
                mesh = unwrap_uv(self.avatar.get(node))
                metrics = uv_quality(mesh)
                self.avatar.replace(node, mesh)
                return f"Atlas UV generado. Superposición muestreada {metrics['overlap_fraction']:.2%}."
            self._edit_background(generate, node)

    def pbr(self):
        node = self._selected()
        if not node:
            return
        dialog = tk.Toplevel(self)
        dialog.title(f"Mapas PBR · {node}")
        dialog.transient(self)
        dialog.grab_set()
        paths = {}
        mesh = self.avatar.get(node)
        current = getattr(mesh.visual, "material", None)
        mode = tk.StringVar(value=getattr(current, "alphaMode", None) or "OPAQUE")
        size = tk.StringVar(value="1024")
        cutoff = tk.StringVar(value=str(getattr(current, "alphaCutoff", None) or .5))
        frame = ttk.Frame(dialog, padding=12)
        frame.pack(fill="both")

        def choose(channel, label):
            path = filedialog.askopenfilename(parent=dialog, filetypes=[("Textura PNG", "*.png")])
            if path:
                paths[channel] = path
                label.set(Path(path).name)

        for index, (key, title) in enumerate(CHANNELS.items()):
            label = tk.StringVar(value="Conservar mapa actual")
            ttk.Button(frame, text=title, command=lambda k=key, v=label: choose(k, v)).grid(row=index, column=0, sticky="ew")
            ttk.Label(frame, textvariable=label).grid(row=index, column=1, sticky="w", padx=8)
        ttk.Combobox(frame, textvariable=size, values=RESOLUTIONS, state="readonly", width=10).grid(row=6, column=0)
        ttk.Combobox(frame, textvariable=mode, values=("OPAQUE", "MASK", "BLEND"),
                     state="readonly", width=10).grid(row=6, column=1)
        ttk.Label(frame, text="Corte alfa").grid(row=7, column=0)
        ttk.Entry(frame, textvariable=cutoff, width=10).grid(row=7, column=1)

        def accept():
            try:
                images = {}
                for channel, path in paths.items():
                    with Image.open(path) as image:
                        images[channel] = image.copy()
                update_pbr_maps(mesh, images, int(size.get()), mode.get(), float(cutoff.get().replace(",", ".")))
            except Exception as exc:
                messagebox.showerror("Mapas PBR", str(exc), parent=dialog)
                return
            dialog.destroy()
            self._edit(lambda: self.avatar.replace(node, mesh), node)

        def save():
            folder = filedialog.askdirectory(parent=dialog)
            if folder:
                try:
                    count = save_pbr_maps(mesh, folder)
                    messagebox.showinfo("Mapas PBR", f"{count} mapas guardados.", parent=dialog)
                except Exception as exc:
                    messagebox.showerror("Mapas PBR", str(exc), parent=dialog)

        ttk.Button(frame, text="Aplicar", command=accept).grid(row=8, column=0)
        ttk.Button(frame, text="Exportar PNG…", command=save).grid(row=8, column=1, padx=8)
        self.wait_window(dialog)
        self.grab_set()

    def _new_fitted(self, node, mesh):
        # The geometry was fitted in the world frame of the current body.
        if node in self.avatar.nodes:
            transform, _ = self.avatar.scene.graph[node]
            mesh.apply_transform(np.linalg.inv(transform))
        self.avatar.upsert(node, mesh)

    def contacts(self):
        node = self._selected()
        if not node or node == BODY:
            return
        try:
            report = body_contacts(self._body(), self.avatar.get(node, world=True))
            inside = report["inside"] if report["inside"] is not None else "indeterminado"
            self.status.set(f"Contactos con el cuerpo {report['contacts']} · Vértices dentro {inside}.")
        except Exception as exc:
            messagebox.showerror("Comprobar ajuste", str(exc), parent=self)

    def material(self):
        node = self._selected()
        if not node:
            return
        _, color = colorchooser.askcolor(parent=self, title="Color del iris" if node in EYES else "Color base PBR")
        if color:
            roughness = simpledialog.askfloat(
                "Rugosidad PBR", "0 para brillo liso y 1 para mate.",
                initialvalue=.23 if node in EYES else .57, minvalue=0, maxvalue=1, parent=self,
            )
            if roughness is None:
                return
            def update():
                mesh = self.avatar.get(node)
                if node in EYES:
                    recolor_iris(mesh, color, roughness)
                else:
                    apply_material(mesh, color, roughness=roughness)
                self.avatar.replace(node, mesh)
            self._edit(update, node)

    def texture(self):
        node = self._selected()
        if not node:
            return
        projection = simpledialog.askstring(
            "Coordenadas UV", "Escribe cylindrical para UV cilíndricas o planar para UV frontales.",
            initialvalue="cylindrical", parent=self,
        )
        if projection is None:
            return
        path = filedialog.askopenfilename(
            parent=self, title="Textura opcional · Cancelar para generar solamente UV",
            filetypes=[("Imagen", "*.png *.jpg *.jpeg")],
        )
        def update():
            mesh = self.avatar.get(node)
            image = None
            if path:
                with Image.open(path) as loaded:
                    image = loaded.convert("RGBA")
            material = getattr(mesh.visual, "material", None)
            factor = getattr(material, "baseColorFactor", None)
            if factor is None:
                factor = [255, 255, 255, 255]
            color = "#%02x%02x%02x" % tuple(map(int, factor[:3])) if image is None else "#ffffff"
            roughness = getattr(material, "roughnessFactor", None)
            apply_material(mesh, color, projection=projection.strip().lower(), image=image,
                           roughness=.57 if roughness is None else roughness)
            self.avatar.replace(node, mesh)
        self._edit(update, node)

    def gradient(self):
        node = self._selected()
        if not node:
            return
        _, top = colorchooser.askcolor(color="#ff268f", title="Color superior", parent=self)
        if not top:
            return
        _, bottom = colorchooser.askcolor(color="#00d9ff", title="Color inferior", parent=self)
        if not bottom:
            return
        def update():
            mesh = self.avatar.get(node)
            material = getattr(mesh.visual, "material", None)
            roughness = getattr(material, "roughnessFactor", None)
            projection = None if getattr(mesh.visual, "uv", None) is not None else "cylindrical"
            apply_material(mesh, "#ffffff", projection=projection, roughness=.57 if roughness is None else roughness,
                           image=color_gradient(top, bottom))
            self.avatar.replace(node, mesh)
        self._edit(update, node)

    def apply(self):
        if not self.edited and not self.visibility_changed:
            self.destroy()
            return
        path = None
        if self.edited:
            if not self.avatar.nodes:
                messagebox.showerror("Componentes", "La escena está vacía.", parent=self)
                return
            path = self._save_path("Guardar nueva escena modular estática", "avatar-modular")
            if not path:
                return
            try:
                write_new_glb(self.avatar.scene, path)
            except Exception as exc:
                messagebox.showerror("Guardar escena", str(exc), parent=self)
                return
        self.result = path, sorted(self.avatar.hidden & set(self.avatar.nodes))
        self.destroy()
