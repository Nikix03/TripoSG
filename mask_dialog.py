"""Paint visible semantic labels over a reference at native image resolution."""

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from aethryx_reference import remove_mask_islands


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
        self.label = tk.StringVar(value=next(name for name in palette if name != "background"))
        self.radius = tk.IntVar(value=12)
        self.tool = tk.StringVar(value="Pincel")
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill="x")
        ttk.Combobox(toolbar, values=list(palette), textvariable=self.label, state="readonly", width=14).pack(side="left")
        for name in ("Pincel", "Borrador"):
            ttk.Radiobutton(toolbar, text=name, variable=self.tool, value=name).pack(side="left")
        ttk.Spinbox(toolbar, from_=1, to=150, textvariable=self.radius, width=5, state="readonly").pack(side="left")
        ttk.Button(toolbar, text="Quitar islas < 24 px", command=self._clean).pack(side="left")
        ttk.Button(toolbar, text="Deshacer", command=self._undo).pack(side="left")
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

    def _start(self, event):
        self.previous = self.mask.copy()
        self._paint(event)

    def _paint(self, event):
        x = min(max(int(event.x * self.mask.width / self.size.width), 0), self.mask.width - 1)
        y = min(max(int(event.y * self.mask.height / self.size.height), 0), self.mask.height - 1)
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
        if self.previous is not None:
            self.mask, self.previous = self.previous, None
            self._redraw()

    def _apply(self):
        self.result = self.mask
        self.destroy()
