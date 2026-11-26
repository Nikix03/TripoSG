"""Local KIRIA tool console. Commands are selected from a fixed registry."""

from dataclasses import dataclass
import json
from queue import Empty, Queue
import re
from threading import Thread
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

from aethryx_modular import ModularAvatar, apply_material, write_new_glb
from aethryx_tools import execute_tool, tool_catalog


@dataclass(frozen=True)
class ScenePlan:
    action: str
    nodes: tuple[str, ...]
    color: str | None = None


def plan_scene_command(action, selection, nodes, color=""):
    if action not in ("ocultar", "mostrar", "color_base"):
        raise ValueError("Comando local desconocido.")
    selected = tuple(dict.fromkeys(selection))
    if not selected or not set(selected).issubset(nodes):
        raise ValueError("Selecciona objetos de la escena activa.")
    if action == "color_base" and not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError("El color requiere el formato #RRGGBB.")
    return ScenePlan(action, selected, color if action == "color_base" else None)


def apply_scene_plan(plan, scene, hidden):
    avatar = ModularAvatar(scene, hidden)
    plan = plan_scene_command(plan.action, plan.nodes, avatar.nodes, plan.color or "")
    if plan.action == "ocultar":
        avatar.hidden.update(plan.nodes)
    elif plan.action == "mostrar":
        avatar.hidden.difference_update(plan.nodes)
    else:
        for node in plan.nodes:
            mesh = avatar.get(node)
            material = getattr(mesh.visual, "material", None)
            roughness = getattr(material, "roughnessFactor", None)
            apply_material(mesh, plan.color, roughness=.57 if roughness is None else roughness)
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
        self.plan = None
        self.busy = False
        self.protocol("WM_DELETE_WINDOW", self.close)
        names = [tool["name"] for tool in tool_catalog() if tool["available"]]
        self.tool = tk.StringVar(value="analyze_project_diagnostics")
        self.action = tk.StringVar(value="ocultar")
        self.color = tk.StringVar(value="#FF268F")
        info = ttk.Label(self, text=f"Selección · {', '.join(self.selection) or 'sin objetos'}", padding=8)
        info.pack(anchor="w", fill="x")
        bar = ttk.Frame(self, padding=8)
        bar.pack(fill="x")
        self.read_button = ttk.Button(bar, text="Ejecutar diagnóstico", command=self.run_read)
        self.read_button.pack(side="right")
        ttk.Combobox(bar, textvariable=self.tool, values=names, state="readonly", width=34).pack(side="left")
        edits = ttk.Frame(self, padding=8)
        edits.pack(fill="x")
        ttk.Combobox(edits, textvariable=self.action, values=("ocultar", "mostrar", "color_base"),
                     state="readonly", width=14).pack(side="left")
        ttk.Entry(edits, textvariable=self.color, width=10).pack(side="left", padx=6)
        ttk.Button(edits, text="Preparar cambio", command=self.prepare).pack(side="left")
        self.apply_button = ttk.Button(edits, text="Aplicar cambio preparado", command=self.apply)
        self.apply_button.pack(side="left", padx=6)
        self.apply_button.state(["disabled"])
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
        self.busy = True
        self.read_button.state(["disabled"])
        self._log("Diagnóstico solicitado", selected)
        result = Queue()

        def run():
            try:
                data = execute_tool(selected, {"model_path": self.context.get("model") or ""}, self.context)
                result.put(data)
            except Exception as exc:
                result.put({"status": "error", "reason": str(exc)})

        def poll():
            try:
                data = result.get_nowait()
            except Empty:
                self.after(60, poll)
                return
            self.busy = False
            self.read_button.state(["!disabled"])
            self._log("Resultado local", json.dumps(data, indent=2, ensure_ascii=False,
                                                    default=lambda value: value.tolist()))

        Thread(target=run, daemon=True).start()
        self.after(60, poll)

    def prepare(self):
        self.plan = None
        self.apply_button.state(["disabled"])
        try:
            nodes = self.scene.graph.nodes_geometry if self.scene is not None else ()
            self.plan = plan_scene_command(self.action.get(), self.selection, nodes, self.color.get())
        except ValueError as exc:
            self._log("Cambio sin preparar", str(exc))
            return
        self._log("Cambio preparado", f"{self.plan.action} · {', '.join(self.plan.nodes)} {self.plan.color or ''}")
        self.apply_button.state(["!disabled"])

    def apply(self):
        if self.busy or self.plan is None:
            return
        if not messagebox.askyesno(
                "Confirmar cambio", f"¿Aplicar {self.plan.action} a {', '.join(self.plan.nodes)}?", parent=self):
            return
        try:
            avatar = apply_scene_plan(self.plan, self.scene, self.hidden)
            path = None
            if self.plan.color:
                path = filedialog.asksaveasfilename(
                    parent=self, title="Guardar versión con el nuevo material", defaultextension=".glb",
                    filetypes=[("GLB", "*.glb")],
                )
                if not path:
                    return
                write_new_glb(avatar.scene, path)
        except (OSError, ValueError, KeyError) as exc:
            self._log("Cambio rechazado", str(exc))
            return
        self.result = path, sorted(avatar.hidden), self.plan.action
        self.destroy()
