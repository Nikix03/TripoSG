"""Paint visible semantic labels over a reference at native image resolution."""

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from aethryx_reference import remove_mask_islands, retain_mask_polygon


class MaskDialog(tk.Toplevel):
    def __init__(self, parent, source, mask, palette):
        super().__init__(parent)
        self.title("Editar máscara visible")
        self.transient(parent)
        self.grab_set()
        self.source = source.convert("RGB")
        self.mask = mask.convert("RGB").copy()
        self.palette = palette
        self.result = None
        self.previous = None
        self.last = None
        self.contour = []
        self.label = tk.StringVar(value=next(name for name in palette if name != "background"))
        self.radius = tk.IntVar(value=12)
        self.tool = tk.StringVar(value="Pincel")
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill="x")
        ttk.Combobox(toolbar, values=list(palette), textvariable=self.label, state="readonly", width=14).pack(side="left")
        for name in ("Pincel", "Borrador", "Contorno"):
            ttk.Radiobutton(toolbar, text=name, variable=self.tool, value=name).pack(side="left")
        ttk.Spinbox(toolbar, from_=1, to=150, textvariable=self.radius, width=5, state="readonly").pack(side="left")
        ttk.Button(toolbar, text="Quitar islas < 24 px", command=self._clean).pack(side="left")
        ttk.Button(toolbar, text="Deshacer", command=self._undo).pack(side="left")
        ttk.Button(toolbar, text="Conservar interior", command=self._retain).pack(side="left")
        ttk.Button(toolbar, text="Aplicar", command=self._apply).pack(side="right")
        self.size = self.source.copy()
        self.size.thumbnail((820, 650))
        self.canvas = tk.Canvas(self, width=self.size.width, height=self.size.height, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", lambda _: setattr(self, "last", None))
        self._redraw()

    def _redraw(self):
        overlay = Image.blend(self.source.resize(self.size.size), self.mask.resize(
            self.size.size, Image.Resampling.NEAREST), .42)
        self.photo = ImageTk.PhotoImage(overlay)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        points = [(x * self.size.width / self.mask.width, y * self.size.height / self.mask.height)
                  for x, y in self.contour]
        for x, y in points:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#fcdc65")
        if len(points) >= 2:
            self.canvas.create_line(*points, fill="#fcdc65", width=2)

    def _start(self, event):
        if self.tool.get() == "Contorno":
            self.contour.append(self._point(event))
            self._redraw()
            return
        self.previous = self.mask.copy()
        self._paint(event)

    def _point(self, event):
        x = min(max(int(event.x * self.mask.width / self.size.width), 0), self.mask.width - 1)
        y = min(max(int(event.y * self.mask.height / self.size.height), 0), self.mask.height - 1)
        return x, y

    def _retain(self):
        try:
            repaired = retain_mask_polygon(self.mask, self.contour)
        except ValueError as exc:
            messagebox.showerror("Contorno de máscara", str(exc), parent=self)
            return
        self.previous, self.mask = self.mask, repaired
        self.contour.clear()
        self._redraw()

    def _paint(self, event):
        if self.tool.get() == "Contorno":
            return
        x, y = self._point(event)
        color = self.palette["background" if self.tool.get() == "Borrador" else self.label.get()]
        radius = self.radius.get()
        draw = ImageDraw.Draw(self.mask)
        if self.last is not None:
            draw.line((self.last, (x, y)), fill=color, width=radius * 2)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        self.last = x, y
        self._redraw()

    def _clean(self):
        self.previous = self.mask.copy()
        self.mask = remove_mask_islands(self.mask, self.palette, 24)
        self._redraw()

    def _undo(self):
        if self.contour:
            self.contour.pop()
            self._redraw()
        elif self.previous is not None:
            self.mask, self.previous = self.previous, None
            self._redraw()

    def _apply(self):
        self.result = self.mask
        self.destroy()
