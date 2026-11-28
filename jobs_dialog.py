"""Review and run persistent TripoSG component jobs in Studio."""

from pathlib import Path
from contextlib import ExitStack
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from PIL import Image, ImageTk
from aethryx_generation import compare_candidates, rank_regional_candidates


STATUS_LABELS = {
    "pending": "Pendiente", "running": "En ejecución", "interrupted": "Interrumpido",
    "cancelled": "Cancelado", "failed": "Error", "complete": "Por revisar",
    "accepted": "Aceptado", "rejected": "Rechazado",
}


class JobsDialog(tk.Toplevel):
    def __init__(self, studio, jobs):
        super().__init__(studio)
        self.studio, self.jobs = studio, jobs
        self.title("Cola de componentes")
        self.transient(studio)
        self.geometry("900x380")
        self.rows = {}
        self.list = ttk.Treeview(self, columns=("status", "attempt"), selectmode="browse")
        self.list.heading("#0", text="Componente")
        self.list.heading("status", text="Estado")
        self.list.heading("attempt", text="Intentos")
        self.list.column("attempt", width=75, stretch=False)
        self.list.pack(fill="both", expand=True, padx=8, pady=8)
        self.list.bind("<<TreeviewSelect>>", self.show_details)
        self.list.bind("<Control-r>", lambda _event: self.perform(self.rank))
        self.details = tk.StringVar()
        ttk.Label(self, textvariable=self.details, wraplength=665).pack(fill="x", padx=8)
        actions = ttk.Frame(self, padding=8)
        actions.pack(fill="x")
        for label, action in (
            ("Registrar actual…", self.add), ("Ejecutar / reintentar", self.run),
            ("Cancelar", self.cancel), ("Ver malla", self.preview), ("Aceptar", lambda: self.review("accepted")),
            ("Rechazar", lambda: self.review("rejected")),
            ("Historial", self.history),
            ("Comparar…", self.compare),
            ("Ajustar a región…", self.fit_region),
            ("Priorizar…", self.rank),
        ):
            ttk.Button(actions, text=label, command=lambda operation=action: self.perform(operation)).pack(side="left")
        self.refresh()

    def perform(self, operation):
        try:
            operation()
            self.refresh()
        except (OSError, sqlite3.Error, ValueError, tk.TclError) as exc:
            messagebox.showerror("Cola de componentes", str(exc), parent=self)

    def refresh(self):
        selected = self.list.selection()
        self.rows = {row["id"]: row for row in self.jobs.entries()}
        self.list.delete(*self.list.get_children())
        for identifier, row in self.rows.items():
            self.list.insert("", "end", iid=identifier, text=row["component"],
                             values=(STATUS_LABELS[row["status"]], row["attempt"]))
        if selected and selected[0] in self.rows:
            self.list.selection_set(selected[0])
        self.show_details()

    def selected(self):
        selection = self.list.selection()
        return self.rows.get(selection[0]) if selection else None

    def show_details(self, _event=None):
        row = self.selected()
        if row is None:
            self.details.set("")
            return
        params = row["parameters"]
        engine = params.get("engine", "triposg")
        if engine == "delivery":
            summary = f"Entrega de avatar · {Path(params['source']).name}"
        elif engine == "blender":
            summary = f"Blender · {Path(params['source']).name} → {Path(params['output']).suffix.upper()}"
        else:
            summary = f"{engine} · {Path(params['image']).name} · semilla {params['settings']['seed']}"
        result = f"\n{Path(row['result']).name}" if row["result"] else ""
        error = f"\n{row['error'][-450:]}" if row["error"] else ""
        self.details.set(summary + result + error)

    def add(self):
        name = simpledialog.askstring("Registrar componente", "Nombre del componente", parent=self)
        if name and name.strip():
            parameters = self.studio.component_job_parameters(resolve=False)
            def register(engine):
                self.jobs.add(name.strip(), {**parameters, "engine": engine})
                self.refresh()
            self.studio.resolve_engine_async(parameters["engine"], parameters["engine_options"], register, self)

    def run(self):
        row = self.selected()
        if row and row["status"] in {"pending", "cancelled", "failed", "interrupted", "rejected"}:
            self.studio.run_component_job(row)

    def cancel(self):
        row = self.selected()
        if row and row["status"] == "pending":
            self.jobs.transition(row["id"], "cancelled", expected="pending")
        elif row and (row["id"], row["attempt"]) == self.studio.active_component_job:
            self.studio.cancel_generation()

    def history(self):
        row = self.selected()
        if row:
            lines = []
            for attempt in self.jobs.history(row["id"]):
                result = Path(attempt["result"]).name if attempt["result"] else ""
                engine = attempt["parameters"].get("engine", "triposg")
                lines.append(f"{attempt['number']} · {engine} · {STATUS_LABELS[attempt['status']]} · {result}")
            messagebox.showinfo("Historial de intentos", "\n".join(lines) or "Sin intentos registrados.", parent=self)

    def compare(self):
        row = self.selected()
        if not row or not row["result"] or Path(row["result"]).suffix.lower() != ".glb":
            return
        second = filedialog.askopenfilename(parent=self, title="Segundo candidato", filetypes=[("GLB", "*.glb")])
        if not second:
            return
        reference = filedialog.askopenfilename(parent=self, title="Referencia con alfa revisado",
                                               filetypes=[("PNG", "*.png")])
        if not reference:
            return
        view = simpledialog.askstring("Vista", "Escribe frontal, lateral o posterior.",
                                      parent=self, initialvalue="frontal")
        if view is None:
            return
        views = {"frontal": "front", "lateral": "side", "posterior": "back"}
        if view not in views:
            raise ValueError("Selecciona frontal, lateral o posterior.")
        with Image.open(reference) as image:
            results = compare_candidates((row["result"], second), image, views[view])
        window = tk.Toplevel(self)
        window.title("Comparación de candidatos")
        window.photos = []
        for index, result in enumerate(results):
            photo = ImageTk.PhotoImage(result["overlay"], master=window)
            window.photos.append(photo)
            ttk.Label(window, image=photo).grid(row=0, column=index)
            text = (f"{Path(result['path']).name}\nIoU {result['iou']:.2%} · "
                    f"{result['faces']:,} caras · {'volumen cerrado' if result['closed'] else 'volumen abierto'}")
            ttk.Label(window, text=text).grid(row=1, column=index)

    def preview(self):
        row = self.selected()
        if row and row["result"] and Path(row["result"]).suffix.lower() == ".glb" and self.studio._workspace_available():
            self.studio.load_model(row["result"], edit=True)

    def fit_region(self):
        row = self.selected()
        if (not row or row["parameters"].get("engine") == "delivery" or not row["result"] or Path(row["result"]).suffix.lower() != ".glb"
                or row["status"] not in {"complete", "accepted"}):
            return
        if self.studio.loaded_scene is None:
            raise ValueError("Abre antes un avatar modular con una guía regional.")
        self.studio.manage_components(candidate={
            "path": row["result"], "engine": row["parameters"].get("engine", "triposg"),
            "job": row["id"], "attempt": row["attempt"],
        })

    def rank(self):
        row = self.selected()
        if not row:
            return
        candidates = [{"path": other["result"], "engine": other["parameters"].get("engine", "triposg"),
                       "job": other["id"], "attempt": other["attempt"]}
                      for other in self.rows.values() if other["component"] == row["component"]
                      and other["result"] and Path(other["result"]).suffix.lower() == ".glb"
                      and other["status"] in {"complete", "accepted"}]
        if len(candidates) < 2:
            raise ValueError("Registra al menos dos resultados con el mismo nombre de componente.")
        directory = filedialog.askdirectory(parent=self, title="Carpeta con front.png, side.png o back.png con alfa")
        if not directory:
            return
        with ExitStack() as stack:
            masks = {name: stack.enter_context(Image.open(Path(directory) / f"{name}.png"))
                     for name in ("front", "side", "back") if (Path(directory) / f"{name}.png").is_file()}
            body, guide = None, None
            indices = self.studio._selected_part_indices()
            if self.studio.loaded_scene is not None and len(indices) == 1:
                from aethryx_modular import BODY, ModularAvatar
                avatar = ModularAvatar(self.studio.loaded_scene)
                node = self.studio.model_parts[indices[0]]["node"]
                if node != BODY and BODY in avatar.nodes:
                    body, guide = avatar.get(BODY, world=True), avatar.get(node, world=True)
            ranked = rank_regional_candidates(candidates, masks, body=body, guide=guide)
        self.studio.regional_review = [
            {key: candidate[key] for key in ("engine", "iou", "topology", "contacts", "position",
                                             "score", "eligible", "confidence")}
            for candidate in ranked
        ]
        lines = []
        for candidate in ranked:
            iou = " / ".join(f"{view} {value:.1%}" for view, value in candidate["iou"].items())
            edges = candidate["topology"]
            lines.append(f"{candidate['engine']} · {iou} · puntuación {candidate['score']:.1%}\n"
                         f"{edges['boundary_edges']} bordes abiertos · {edges['nonmanifold_edges']} no manifold")
            if candidate["contacts"] is not None:
                lines[-1] += (f" · {candidate['contacts']['contacts']} contactos · "
                              f"{candidate['contacts']['inside']} interiores · "
                              f"desplazamiento {candidate['position']:.1%} de la altura")
        best = ranked[0]
        if best["eligible"]:
            self.list.selection_set(best["job"])
            self.list.see(best["job"])
        summary = ("Se midió la posición sobre la guía activa." if guide is not None
                   else "Evidencia limitada a silueta y topología. Revisa el ajuste sobre el avatar.")
        messagebox.showinfo("Prioridad por región", "\n\n".join(lines) + "\n\n" + summary, parent=self)

    def review(self, status):
        row = self.selected()
        if row and row["status"] == "complete":
            self.jobs.transition(row["id"], status)
