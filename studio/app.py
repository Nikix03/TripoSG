"""Nikix 3D Studio — local desktop interface for the existing TripoSG engine."""
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QSettings
from PySide6.QtGui import QPixmap, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFormLayout, QFrame, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QPlainTextEdit, QMessageBox, QSplitter, QScrollArea,
    QTabWidget, QLineEdit,
)
from studio.core import ROOT, generation_args, new_output

STYLE = """
QWidget { background:#151821; color:#e5e9f1; font-family:'Segoe UI'; font-size:13px; }
QFrame#panel { background:#1d2230; border:1px solid #343a4a; border-radius:10px; }
QLabel { background:transparent; }
QLabel#brand { font-size:25px; font-weight:700; color:#f5f7ff; }
QLabel#section { font-size:15px; font-weight:700; color:#65dbe9; }
QLabel#muted { color:#a9b3c7; }
QLabel#preview { background:#12151d; border:1px solid #343a4a; border-radius:8px; }
QPushButton { background:#30384a; border:1px solid #465067; padding:9px 12px; border-radius:7px; font-weight:600; }
QPushButton:hover { background:#424d64; }
QPushButton#primary { background:#ed78b9; color:#1b1420; border:1px solid #ed78b9; }
QPushButton#primary:hover { background:#f3a0d0; }
QPushButton:disabled { background:#252a36; color:#737d91; border-color:#343a4a; }
QLineEdit,QSpinBox,QDoubleSpinBox,QPlainTextEdit { background:#12151d; border:1px solid #3a4255; border-radius:6px; padding:6px; selection-background-color:#ae528b; }
QProgressBar { background:#11141c; border:1px solid #343a4a; border-radius:5px; min-height:8px; max-height:8px; }
QProgressBar::chunk { background:#69dce9; border-radius:4px; }
QTabWidget::pane { border:1px solid #343a4a; border-radius:7px; }
QTabBar::tab { background:#252b3a; padding:10px 17px; margin-right:3px; }
QTabBar::tab:selected { color:#70e4ee; background:#343b4c; }
QScrollArea { border:0; }
QSplitter::handle { background:#151821; width:7px; }
"""

def label(text, name=""):
    w = QLabel(text)
    w.setObjectName(name)
    w.setWordWrap(True)
    return w

def panel():
    w = QFrame()
    w.setObjectName("panel")
    layout = QVBoxLayout(w)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)
    return w, layout

class Studio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nikix 3D Studio | TripoSG")
        self.resize(1250, 820)
        self.setMinimumSize(880, 620)
        self.settings = QSettings("Nikix", "3DStudio")
        self.image = self.output = self.pending = None
        self.pixmap = None
        self.mesh_item = None
        self.cancelled = False
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.read_log)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)

        base = QWidget()
        self.setCentralWidget(base)
        outer = QVBoxLayout(base)
        outer.setContentsMargins(20, 16, 20, 16)
        header = QHBoxLayout()
        header.addWidget(label("NIKIX  /  3D STUDIO", "brand"))
        header.addStretch()
        header.addWidget(label("LOCAL  •  TRIPOSG  •  FASE 01", "muted"))
        outer.addLayout(header)
        outer.addWidget(label("Imagen a geometría 3D   /   Espacio de trabajo", "muted"))
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)

        left, controls = panel()
        controls.addWidget(label("01   Referencia", "section"))
        self.image_label = label("Sin imagen seleccionada", "preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(180)
        controls.addWidget(self.image_label)
        self.select_btn = QPushButton("Seleccionar imagen")
        self.select_btn.clicked.connect(self.select_image)
        controls.addWidget(self.select_btn)
        self.image_name = label("PNG, JPG o WEBP · un personaje por imagen", "muted")
        controls.addWidget(self.image_name)
        controls.addWidget(label("02   Configuración", "section"))
        form = QFormLayout()
        self.steps = QSpinBox(); self.steps.setRange(1, 200); self.steps.setValue(50)
        self.seed = QSpinBox(); self.seed.setRange(0, 2147483647); self.seed.setValue(42)
        self.guidance = QDoubleSpinBox(); self.guidance.setRange(0, 20); self.guidance.setValue(7); self.guidance.setSingleStep(.5)
        self.limit = QCheckBox("Reducir caras (no retopología)")
        self.faces = QSpinBox(); self.faces.setRange(100, 2000000); self.faces.setValue(50000); self.faces.setSingleStep(5000); self.faces.setEnabled(False)
        self.limit.toggled.connect(self.faces.setEnabled)
        form.addRow("Pasos", self.steps); form.addRow("Semilla", self.seed)
        form.addRow("Guidance", self.guidance); form.addRow(self.limit)
        form.addRow("Caras objetivo", self.faces)
        controls.addLayout(form)
        controls.addWidget(label("03   Destino", "section"))
        self.directory = QLineEdit(str(self.settings.value("directory", str(Path.home() / "Documents" / "Nikix3DStudio"))))
        self.directory.setReadOnly(True)
        self.dir_btn = QPushButton("Cambiar carpeta")
        self.dir_btn.clicked.connect(self.select_directory)
        controls.addWidget(self.directory); controls.addWidget(self.dir_btn)
        self.generate_btn = QPushButton("Generar modelo 3D")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.clicked.connect(self.generate)
        self.cancel_btn = QPushButton("Cancelar generación")
        self.cancel_btn.clicked.connect(self.cancel)
        controls.addWidget(self.generate_btn); controls.addWidget(self.cancel_btn)
        controls.addStretch()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left)
        split.addWidget(scroll)

        right, view = panel()
        view.addWidget(label("04   Inspección y exportación", "section"))
        self.tabs = QTabWidget()
        view.addWidget(self.tabs, 1)
        self.preview_page = QWidget()
        self.preview_layout = QVBoxLayout(self.preview_page)
        self.viewport_message = label("Genera o abre una malla GLB para inspeccionarla.", "muted")
        self.preview_layout.addWidget(self.viewport_message)
        self.gl = None
        try:
            import pyqtgraph.opengl as gl
            self.gl_module = gl
            self.gl = gl.GLViewWidget()
            self.gl.setBackgroundColor("#161b26")
            self.preview_layout.addWidget(self.gl, 1)
        except Exception as exc:
            self.viewport_message.setText("Visor OpenGL no disponible: " + str(exc))
        self.tabs.addTab(self.preview_page, "Visor 3D")
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2000)
        self.tabs.addTab(self.log, "Registro del motor")
        self.stats = label("Sin modelo cargado. TripoSG genera geometría, no un avatar riggeado.", "muted")
        view.addWidget(self.stats)
        actions = QHBoxLayout()
        self.open_btn = QPushButton("Abrir GLB")
        self.open_btn.clicked.connect(self.open_glb)
        self.reset_btn = QPushButton("Restablecer cámara")
        self.reset_btn.clicked.connect(self.reset_camera)
        self.glb_btn = QPushButton("Guardar GLB")
        self.glb_btn.clicked.connect(lambda: self.export("glb"))
        self.obj_btn = QPushButton("Exportar OBJ")
        self.obj_btn.clicked.connect(lambda: self.export("obj"))
        for btn in (self.open_btn, self.reset_btn, self.glb_btn, self.obj_btn):
            actions.addWidget(btn)
        view.addLayout(actions)
        split.addWidget(right); split.setSizes([340, 850]); split.setStretchFactor(1, 1)
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setRange(0, 100)
        self.status = label("Listo · Selecciona una imagen para comenzar.", "muted")
        outer.addWidget(self.progress); outer.addWidget(self.status)
        self.set_busy(False)

    def set_busy(self, busy):
        for w in (self.select_btn, self.dir_btn, self.steps, self.seed, self.guidance, self.limit, self.open_btn):
            w.setEnabled(not busy)
        self.faces.setEnabled(not busy and self.limit.isChecked())
        self.generate_btn.setEnabled(not busy and self.image is not None)
        self.cancel_btn.setEnabled(busy and not self.cancelled)
        self.glb_btn.setEnabled(not busy and self.output is not None)
        self.obj_btn.setEnabled(not busy and self.output is not None)
        self.reset_btn.setEnabled(self.gl is not None)
        self.progress.setRange(0, 0 if busy else 100)

    def select_image(self):
        name, _ = QFileDialog.getOpenFileName(self, "Seleccionar referencia", "", "Imágenes (*.png *.jpg *.jpeg *.webp)")
        if not name: return
        pix = QPixmap(name)
        if pix.isNull():
            QMessageBox.warning(self, "Imagen", "No se pudo leer la imagen."); return
        self.image = Path(name); self.pixmap = pix
        self.image_name.setText(self.image.name)
        self.refresh_image()
        self.set_busy(False)

    def refresh_image(self):
        if self.pixmap is not None:
            self.image_label.setPixmap(self.pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "image_label"): self.refresh_image()

    def select_directory(self):
        name = QFileDialog.getExistingDirectory(self, "Carpeta de salida", self.directory.text())
        if name:
            self.directory.setText(name); self.settings.setValue("directory", name)

    def generate(self):
        if self.process.state() != QProcess.ProcessState.NotRunning: return
        try:
            self.pending = new_output(self.directory.text())
            args = generation_args(self.image, self.pending, self.seed.value(), self.steps.value(), self.guidance.value(), self.faces.value() if self.limit.isChecked() else -1)
        except Exception as exc:
            QMessageBox.warning(self, "Configuración", str(exc)); return
        self.cancelled = False
        self.log.clear()
        self.log.appendPlainText("Motor: " + sys.executable + "\nSalida: " + str(self.pending))
        self.tabs.setCurrentWidget(self.log)
        self.set_busy(True)
        self.status.setText("Generando… La primera ejecución puede descargar los pesos del motor.")
        self.process.start(sys.executable, args)

    def read_log(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data: self.log.appendPlainText(data.rstrip())

    def process_error(self, error):
        self.log.appendPlainText(self.process.errorString())
        if error == QProcess.ProcessError.FailedToStart:
            self.status.setText("No se pudo iniciar Python.")
            self.set_busy(False)

    def cancel(self):
        self.cancelled = True; self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelando…")
        self.process.terminate()
        QTimer.singleShot(2500, self.kill_if_cancelled)

    def kill_if_cancelled(self):
        if self.cancelled and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def finished(self, code, exit_status):
        self.read_log()
        self.set_busy(False)
        if self.cancelled:
            self.status.setText("Generación cancelada."); return
        if code != 0 or exit_status != QProcess.ExitStatus.NormalExit or not self.pending or not self.pending.is_file():
            self.status.setText("La generación falló. Revisa el registro del motor.")
            self.tabs.setCurrentWidget(self.log); return
        if self.load_mesh(self.pending):
            self.status.setText("Modelo generado correctamente: " + self.pending.name)
            self.progress.setValue(100)
        else:
            self.status.setText("Archivo generado, pero no se pudo validar la malla.")

    def open_glb(self):
        name, _ = QFileDialog.getOpenFileName(self, "Abrir modelo", self.directory.text(), "Modelo GLB (*.glb)")
        if name and self.load_mesh(Path(name)):
            self.status.setText("Modelo cargado: " + Path(name).name)

    def load_mesh(self, path):
        try:
            import numpy as np
            import trimesh
            mesh = trimesh.load(str(path), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
                raise ValueError("Malla vacía o inválida.")
        except Exception as exc:
            QMessageBox.warning(self, "Modelo", str(exc)); return False
        self.output = Path(path)
        self.stats.setText(f"{len(mesh.vertices):,} vértices  |  {len(mesh.faces):,} triángulos  |  Cerrada: {'sí' if mesh.is_watertight else 'no'}  |  Geometría sin rig")
        if self.gl is not None:
            try:
                if self.mesh_item is not None:
                    self.gl.removeItem(self.mesh_item); self.mesh_item = None
                verts = np.asarray(mesh.vertices, dtype=np.float32)
                extent = float(np.max(mesh.extents))
                if extent <= 0: raise ValueError("Dimensiones inválidas.")
                verts = (verts - mesh.bounds.mean(axis=0)) / extent * 2
                # Display-only conversion from glTF Y-up to viewer Z-up.
                verts = verts[:, [0, 2, 1]] * np.array([1, -1, 1], dtype=np.float32)
                self.mesh_item = self.gl_module.GLMeshItem(vertexes=verts, faces=np.asarray(mesh.faces, dtype=np.uint32), color=(.72,.78,.88,1), smooth=True, shader="shaded", drawEdges=False)
                self.gl.addItem(self.mesh_item)
                self.reset_camera()
                self.viewport_message.setText("Arrastra para orbitar · rueda para zoom · vista de geometría")
            except Exception as exc:
                self.viewport_message.setText("Vista previa no disponible: " + str(exc))
        self.tabs.setCurrentWidget(self.preview_page)
        self.set_busy(False)
        return True

    def reset_camera(self):
        if self.gl is not None: self.gl.setCameraPosition(distance=4, elevation=10, azimuth=-90)

    def export(self, kind):
        if self.output is None: return
        name, _ = QFileDialog.getSaveFileName(self, "Exportar modelo", str(self.output.with_suffix("." + kind)), kind.upper() + " (*." + kind + ")")
        if not name: return
        target = Path(name)
        if target.suffix.lower() != "." + kind: target = target.with_suffix("." + kind)
        try:
            if target.resolve() != self.output.resolve():
                if kind == "glb": shutil.copyfile(self.output, target)
                else:
                    import trimesh
                    mesh = trimesh.load(str(self.output), force="mesh")
                    mesh.export(str(target), include_texture=False)
            self.status.setText("Exportado: " + str(target))
        except Exception as exc:
            QMessageBox.warning(self, "Exportación", str(exc))

    def closeEvent(self, event: QCloseEvent):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Generación activa", "Cancela la generación antes de cerrar.")
            event.ignore(); return
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = Studio(); window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
