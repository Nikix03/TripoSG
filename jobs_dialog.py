"""Review and run persistent TripoSG component jobs in Studio."""

from pathlib import Path
import sqlite3
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk


STATUS_LABELS = {
    "pending": "Pendiente", "running": "En ejecución", "interrupted": "Interrumpido",
    "cancelled": "Cancelado", "failed": "Error", "complete": "Por revisar",
    "accepted": "Aceptado", "rejected": "Rechazado",
}


class JobsDialog(tk.Toplevel):
    def __init__(self, studio, jobs):
        super().__init__(studio)
        self.studio, self.jobs = studio, jobs
        self.title("Cola de componentes TripoSG")
        self.transient(studio)
        self.geometry("690x380")
        self.rows = {}
        self.list = ttk.Treeview(self, columns=("status", "attempt"), selectmode="browse")
        self.list.heading("#0", text="Componente")
        self.list.heading("status", text="Estado")
        self.list.heading("attempt", text="Intentos")
        self.list.column("attempt", width=75, stretch=False)
        self.list.pack(fill="both", expand=True, padx=8, pady=8)
        self.list.bind("<<TreeviewSelect>>", self.show_details)
        self.details = tk.StringVar()
        ttk.Label(self, textvariable=self.details, wraplength=665).pack(fill="x", padx=8)
        actions = ttk.Frame(self, padding=8)
        actions.pack(fill="x")
        for label, action in (
            ("Registrar actual…", self.add), ("Ejecutar / reintentar", self.run),
            ("Cancelar", self.cancel), ("Ver malla", self.preview), ("Aceptar", lambda: self.review("accepted")),
            ("Rechazar", lambda: self.review("rejected")),
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
        summary = (f"{Path(params['image']).name} · {params['settings']['steps']} pasos · "
                   f"semilla {params['settings']['seed']}")
        result = f"\n{Path(row['result']).name}" if row["result"] else ""
        error = f"\n{row['error'][-450:]}" if row["error"] else ""
        self.details.set(summary + result + error)

    def add(self):
        name = simpledialog.askstring("Registrar componente", "Nombre del componente", parent=self)
        if name and name.strip():
            self.jobs.add(name.strip(), self.studio.component_job_parameters())

    def run(self):
        row = self.selected()
        if row and row["status"] in {"pending", "cancelled", "failed", "interrupted", "rejected"}:
            self.studio.run_component_job(row)

    def cancel(self):
        row = self.selected()
        if row and row["status"] == "pending":
            self.jobs.transition(row["id"], "cancelled", expected="pending")
        elif row and row["id"] == self.studio.active_component_job:
            self.studio.cancel_generation()

    def preview(self):
        row = self.selected()
        if row and row["result"] and self.studio._workspace_available():
            self.studio.load_model(row["result"], edit=True)

    def review(self, status):
        row = self.selected()
        if row and row["status"] == "complete":
            self.jobs.transition(row["id"], status)
