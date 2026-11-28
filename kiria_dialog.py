"""Local KIRIA tool console. Commands are selected from a fixed registry."""

from dataclasses import dataclass
import json
import math
from queue import Empty, Queue
import re
from threading import Thread
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np

from aethryx_anatomy import blocking_geometry_issues, validate_deformation
from aethryx_anime import hair_locks
from aethryx_modular import (BODY, EYES, ModularAvatar, apply_material, clear_hair_penetrations, facial_vertices,
                            fit_garment_with_clearance, fit_sleeves, recolor_iris, write_new_glb)
from aethryx_ollama import LOCAL_ACTIONS, OllamaProvider
from aethryx_rig import PosePreview, hair_clearance_bodies
from aethryx_tools import execute_tool, tool_catalog


@dataclass(frozen=True)
class ScenePlan:
    action: str
    nodes: tuple[str, ...]
    color: str | None = None
    value: float | None = None

    @property
    def parameter(self):
        return self.color or (str(self.value) if self.value is not None else "")


def plan_scene_command(action, selection, nodes, parameter=""):
    if action not in LOCAL_ACTIONS:
        raise ValueError("Comando local desconocido.")
    selected = tuple(dict.fromkeys(selection))
    if not selected or not set(selected).issubset(nodes):
        raise ValueError("Selecciona objetos de la escena activa.")
    if action in ("color_base", "color_iris"):
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", parameter):
            raise ValueError("El color requiere el formato #RRGGBB.")
        if action == "color_iris" and not set(selected).issubset(EYES):
            raise ValueError("El cambio de iris requiere seleccionar ojos.")
        return ScenePlan(action, selected, parameter)
    ranges = {"altura": (.80, 1.20), "nariz": (.75, 1.25), "longitud_cabello": (0., 1.),
              "holgura_ropa": (.003, .06), "despejar_mechon": (.0001, .01)}
    if action in ranges:
        low, high = ranges[action]
        try:
            value = float(parameter.replace(",", "."))
        except ValueError:
            raise ValueError(f"Introduce un valor numérico entre {low} y {high}.") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Introduce un valor entre {low} y {high}.")
        if action not in {"holgura_ropa", "despejar_mechon"} and selected != (BODY,):
            raise ValueError("Selecciona el cuerpo base para preparar este cambio.")
        if action == "holgura_ropa" and (len(selected) != 1 or not selected[0].startswith("prenda-") or BODY not in nodes):
            raise ValueError("Selecciona una prenda ajustable en una escena con cuerpo base.")
        if action == "despejar_mechon" and (len(selected) != 1 or BODY not in nodes or (
                selected[0] != "cabello-fondo" and not selected[0].startswith(("cabello-largo-", "cabello-flequillo-")))):
            raise ValueError("Selecciona una cubierta o un mechón de Genesis en una escena con cuerpo base.")
        return ScenePlan(action, selected, value=value)
    if action == "mangas":
        if selected != (BODY,) or parameter:
            raise ValueError("Selecciona solo el cuerpo base y deja el parámetro vacío para proponer mangas.")
        if any(name.startswith("prenda-manga-") for name in nodes):
            raise ValueError("La escena ya contiene mangas.")
    return ScenePlan(action, selected)


def apply_scene_plan(plan, scene, hidden):
    avatar = ModularAvatar(scene, hidden)
    plan = plan_scene_command(plan.action, plan.nodes, avatar.nodes, plan.parameter)
    if plan.action == "ocultar":
        avatar.hidden.update(plan.nodes)
    elif plan.action == "mostrar":
        avatar.hidden.difference_update(plan.nodes)
    elif plan.action in ("color_base", "color_iris"):
        for node in plan.nodes:
            mesh = avatar.get(node)
            material = getattr(mesh.visual, "material", None)
            roughness = getattr(material, "roughnessFactor", None)
            if plan.action == "color_iris":
                recolor_iris(mesh, plan.color, .23 if roughness is None else roughness)
            else:
                apply_material(mesh, plan.color, roughness=.57 if roughness is None else roughness)
            avatar.replace(node, mesh)
    elif plan.action == "nariz":
        body = avatar.get(BODY)
        changed = body.copy()
        changed.vertices = facial_vertices(body, {"nose": plan.value})
        issues = blocking_geometry_issues(validate_deformation(body, changed))
        if issues:
            raise ValueError("\n".join(issues))
        avatar.replace(BODY, changed)
    elif plan.action == "longitud_cabello":
        pieces = hair_locks(avatar.get(BODY, world=True), plan.value)
        avatar.hidden.update(node for node in avatar.nodes if node.startswith("cabello") and node not in pieces)
        for node, mesh in pieces.items():
            if node in avatar.nodes:
                mesh.apply_transform(np.linalg.inv(avatar.scene.graph[node][0]))
            avatar.upsert(node, mesh)
            avatar.hidden.discard(node)
    elif plan.action == "holgura_ropa":
        node = plan.nodes[0]
        previous = avatar.get(node)
        garment = previous.metadata.get("aethryx_garment")
        if not isinstance(garment, dict):
            raise ValueError("La prenda necesita parámetros de ajuste de Genesis.")
        new, _ = fit_garment_with_clearance(avatar.get(BODY, world=True), plan.value, garment["thickness"], garment["style"])
        new.visual.material = previous.visual.material.copy()
        new.apply_transform(np.linalg.inv(avatar.scene.graph[node][0]))
        avatar.replace(node, new)
    elif plan.action == "despejar_mechon":
        node = plan.nodes[0]
        preview = PosePreview(avatar.scene)
        mesh, review = clear_hair_penetrations(preview.parts[BODY], preview.parts[node],
                                               clearance=plan.value, posed_bodies=hair_clearance_bodies(preview),
                                               repair_roots=True, paired_shell=node == "cabello-fondo")
        if review["changed"]:
            mesh.metadata["aethryx_pose_clearance"] = review
            mesh.apply_transform(np.linalg.inv(avatar.scene.graph[node][0]))
            avatar.replace(node, mesh)
    elif plan.action == "mangas":
        for node, sleeve in fit_sleeves(avatar.get(BODY, world=True)).items():
            avatar.add(node, sleeve)
    elif plan.action == "altura":
        floor = avatar.get(BODY, world=True).bounds[0, 1]
        scale = np.eye(4)
        scale[1, 1], scale[1, 3] = plan.value, floor * (1 - plan.value)
        for node in avatar.nodes:
            transform, _ = avatar.scene.graph[node]
            local = np.linalg.solve(transform, scale @ transform)
            mesh = avatar.get(node)
            mesh.apply_transform(local)
            if node == BODY:
                levels = mesh.metadata.get("aethryx_face_levels")
                guide = mesh.metadata.get("aethryx_eye_guide")
                if isinstance(levels, dict):
                    mesh.metadata["aethryx_face_levels"] = {
                        name: float(local[1, 1] * y + local[1, 3]) for name, y in levels.items()
                    }
                if isinstance(guide, list) and len(guide) == 4:
                    center = local[:3, :3] @ guide[:3] + local[:3, 3]
                    mesh.metadata["aethryx_eye_guide"] = [*center.tolist(), guide[3] * plan.value]
            avatar.replace(node, mesh)
    return avatar


class KiriaDialog(tk.Toplevel):
    def __init__(self, master, context, scene, selection, hidden):
        super().__init__(master)
        self.title("KIRIA · Herramientas locales")
        self.geometry("860x610")
        self.transient(master)
        self.grab_set()
        self.context = context
        self.scene = scene
        self.selection = tuple(selection)
        self.hidden = tuple(hidden)
        self.result = None
        self.delivery_requested = False
        self.plan = None
        self.busy = False
        self.provider = OllamaProvider()
        self.history = []
        self.protocol("WM_DELETE_WINDOW", self.close)
        names = [tool["name"] for tool in tool_catalog() if tool["available"]]
        self.tool = tk.StringVar(value="analyze_project_diagnostics")
        self.action = tk.StringVar(value="ocultar")
        self.color = tk.StringVar(value="#FF268F")
        self.model = tk.StringVar()
        info = ttk.Label(self, text=f"Selección · {', '.join(self.selection) or 'sin objetos'}", padding=8)
        info.pack(anchor="w", fill="x")
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        self.read_button = ttk.Button(bar, text="Ejecutar diagnóstico", command=self.run_read)
        self.read_button.pack(side="right")
        ttk.Button(bar, text="Preparar entrega…", command=self.request_delivery).pack(side="right", padx=6)
        ttk.Combobox(bar, textvariable=self.tool, values=names, state="readonly", width=34).pack(side="left")
        edits = ttk.Frame(self, padding=8)
        edits.pack(fill="x")
        ttk.Combobox(edits, textvariable=self.action, values=LOCAL_ACTIONS,
                     state="readonly", width=14).pack(side="left")
        ttk.Entry(edits, textvariable=self.color, width=10).pack(side="left", padx=6)
        ttk.Button(edits, text="Preparar cambio", command=self.prepare).pack(side="left")
        self.apply_button = ttk.Button(edits, text="Aplicar cambio preparado", command=self.apply)
        self.apply_button.pack(side="left", padx=6)
        self.apply_button.state(["disabled"])
        provider = ttk.Frame(self, padding=8)
        provider.pack(fill="x")
        ttk.Button(provider, text="Buscar modelos Ollama locales", command=self.discover_models).pack(side="left")
        self.models = ttk.Combobox(provider, textvariable=self.model, state="readonly", width=27)
        self.models.pack(side="left", padx=6)
        chat = ttk.Frame(self, padding=8)
        chat.pack(fill="x")
        self.prompt = tk.StringVar()
        entry = ttk.Entry(chat, textvariable=self.prompt)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        entry.bind("<Return>", self.send_chat)
        ttk.Button(chat, text="Enviar a Ollama local", command=self.send_chat).pack(side="right")
        log = ttk.Frame(self, padding=8)
        log.pack(fill="both", expand=True)
        self.output = tk.Text(log, wrap="word", bg="#11141b", fg="#eef1f7", state="disabled")
        scrollbar = ttk.Scrollbar(log, command=self.output.yview)
        self.output.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.output.pack(side="left", fill="both", expand=True)

    def close(self):
        if not self.busy:
            self.destroy()

    def _log(self, heading, content):
        self.output.configure(state="normal")
        self.output.insert("end", f"{heading}\n{content}\n\n")
        self.output.see("end")
        self.output.configure(state="disabled")

    def run_read(self):
        if self.busy:
            return
        selected = self.tool.get()
        self._log("Diagnóstico solicitado", selected)
        self._submit(
            lambda: execute_tool(selected, {"model_path": self.context.get("model") or ""}, self.context),
            lambda data: self._log("Resultado local", json.dumps(
                data, indent=2, ensure_ascii=False, default=lambda value: value.tolist(),
            )[:64000]),
        )

    def _submit(self, operation, completed):
        self.busy = True
        self.read_button.state(["disabled"])
        result = Queue()

        def run():
            try:
                result.put((True, operation()))
            except Exception as exc:
                result.put((False, str(exc)))

        def poll():
            try:
                ok, data = result.get_nowait()
            except Empty:
                self.after(60, poll)
                return
            self.busy = False
            self.read_button.state(["!disabled"])
            if ok:
                completed(data)
            else:
                self._log("Operación no disponible", data)

        Thread(target=run, daemon=True).start()
        self.after(60, poll)

    def discover_models(self):
        if self.busy:
            return
        self.model.set("")
        self.models.configure(values=())

        def show(names):
            self.models.configure(values=names)
            if names:
                self.model.set(names[0])
            self._log("Ollama local", "\n".join(names) if names else "No hay modelos GGUF locales instalados.")

        self._submit(self.provider.list_models, show)

    def send_chat(self, _event=None):
        if self.busy:
            return
        model, prompt = self.model.get(), self.prompt.get().strip()
        if not model or not prompt:
            self._log("Ollama local", "Selecciona un modelo local y escribe una consulta.")
            return
        self.prompt.set("")
        self.plan = None
        self.apply_button.state(["disabled"])
        self._log("Consulta a Ollama local", prompt)
        previous = tuple(self.history)

        def receive(response):
            if response.text:
                self._log(f"Respuesta de {model}", response.text)
                self.history.extend(({"role": "user", "content": prompt},
                                     {"role": "assistant", "content": response.text}))
                del self.history[:-10]
            if response.proposed_action:
                self.action.set(response.proposed_action)
                self.color.set(response.proposed_color or (
                    str(response.proposed_value) if response.proposed_value is not None else ""
                ))
                self.prepare()

        self._submit(lambda: self.provider.chat(model, prompt, self.selection, previous), receive)

    def request_delivery(self):
        if not self.busy:
            self.delivery_requested = True
            self.destroy()

    def prepare(self):
        self.plan = None
        self.apply_button.state(["disabled"])
        try:
            nodes = self.scene.graph.nodes_geometry if self.scene is not None else ()
            self.plan = plan_scene_command(self.action.get(), self.selection, nodes, self.color.get())
        except ValueError as exc:
            self._log("Cambio sin preparar", str(exc))
            return
        scope = "avatar completo" if self.plan.action == "altura" else ", ".join(self.plan.nodes)
        self._log("Cambio preparado", f"{self.plan.action} · {scope} {self.plan.parameter}")
        self.apply_button.state(["!disabled"])

    def apply(self):
        if self.busy or self.plan is None:
            return
        plan = self.plan
        scope = "avatar completo" if plan.action == "altura" else ", ".join(plan.nodes)
        if not messagebox.askyesno(
                "Confirmar cambio", f"¿Aplicar {plan.action} {plan.parameter} a {scope}?",
                parent=self):
            return
        path = None
        if plan.action not in ("ocultar", "mostrar"):
            path = filedialog.asksaveasfilename(
                parent=self, title="Guardar versión editada del avatar", defaultextension=".glb",
                filetypes=[("GLB", "*.glb")],
            )
            if not path:
                return

        def run():
            avatar = apply_scene_plan(plan, self.scene, self.hidden)
            if path:
                write_new_glb(avatar.scene, path)
            return path, sorted(avatar.hidden), plan.action

        def finish(result):
            self.result = result
            self.destroy()

        self._submit(run, finish)
