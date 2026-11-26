"""Scene component editor for the Genesis modular avatar workflow."""

from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
import tkinter as tk

import numpy as np
from PIL import Image

from aethryx_anatomy import blocking_geometry_issues, validate_deformation
from aethryx_modular import (
    BODY, EYES, FACE_CONTROLS, ModularAvatar, anime_body, apply_material, avatar_eye,
    body_contacts, facial_vertices, fit_garment, fit_hair, head_frame, recolor_iris, write_new_glb,
)


class ComponentDialog(tk.Toplevel):
    def __init__(self, master, scene, hidden):
        super().__init__(master)
        self.title("Genesis · Componentes del avatar")
        self.geometry("850x570")
        self.minsize(680, 500)
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
            (("Color / iris…", self.material), ("UV y textura…", self.texture), ("Comprobar ajuste", self.contacts)),
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

    def _edit_background(self, operation, node):
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
                self.edited = True
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
            "1 1 1 1 1 1", len(FACE_CONTROLS),
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
                eye.vertices = previous + (eye.vertices - previous) * factors["eye_size"]
                dx = (world_center[0] - center) * (factors["eye_spacing"] - 1)
                eye.apply_translation(np.linalg.solve(transform[:3, :3], [dx, 0, 0]))
                working.replace(node, eye)
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
        values = self._numbers(
            "Prenda superior", "Holgura 0,003–0,06 y espesor 0,001–0,03 de la altura.", "0.012 0.006", 2,
        )
        if values is None:
            return
        def generate():
            garment, report = fit_garment(self._body(), *values)
            if report["contacts"] or report["inside"] != 0:
                raise ValueError(f"Se rechazó el ajuste. Contactos {report['contacts']}, vértices dentro {report['inside']}.")
            self._new_fitted("prenda-superior", garment)
            return "Prenda cerrada, cero contactos y cero vértices dentro del cuerpo."
        self._edit_background(generate, "prenda-superior")

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
