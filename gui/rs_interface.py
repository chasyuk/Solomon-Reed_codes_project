"""
QR + Reed–Solomon desktop UI.

**Error correction when reading QR codewords** (scan tab) uses only this
repository’s RS stack — no third-party RS libraries. The encode/damage/recover lab
runs Reed–Solomon on the logical module grid via ``decode_qr_from_modules_report``
(so CV warp/threshold jitter does not affect “Recover”).

**Scan tab** uses the detector + masking + Reed–Solomon pipeline:

``decode_qr_full_report`` → ``heal_qr_data`` → ``QR_Decoder`` /
``QR_ReedSolomon`` → ``core.decode.DecodeReedSolomon`` →
``core.reed_solomon.ReedSolomon`` (``core.galois_field``).

The PyPI packages ``reedsolo`` / ``unireedsolomon`` are **not** imported here.

**Generating** QR codes (encode tab) goes through ``qr.qr_generating`` which
substitutes ``qrcode.util.create_data`` with ``qr.qr_rs_codec.create_data_core_rs``:
data masking and matrix layout still come from the ``qrcode`` package, but every
ECC byte is produced from ``ReedSolomon(8, 255 - ec_per_block).code_poly`` and
``core.galois_field`` (same generator and remainder arithmetic as in decoding).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from qr.qr_generating import (
    build_qr_modules_and_image,
    damage_qr_modules,
    modules_to_pil,
)
from qr.qr_decode import decode_qr_from_modules_report, decode_qr_full_report

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_PX = 260
RENDER_BOX = 10
RENDER_BORDER = 4


def _pil_to_ctk(pil_img: Image.Image, max_side: int = PREVIEW_PX) -> ctk.CTkImage:
    img = pil_img.convert("RGB").copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)


class ReedSolomonGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reed–Solomon · QR lab")
        self.geometry("1180x820")
        self.minsize(980, 680)

        self._clean_modules: list[list[bool]] | None = None
        self._damaged_modules: list[list[bool]] | None = None
        self._last_clean_text: str = ""

        self._img_clean: ctk.CTkImage | None = None
        self._img_damaged: ctk.CTkImage | None = None
        self._img_recovered: ctk.CTkImage | None = None

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=16, pady=16)

        self.tabview.add("QR encode · damage · recover")
        self.tabview.add("Scan QR (our decoder)")
        self._build_lab_tab()
        self._build_scan_tab()
        self._bind_select_all()

    def _bind_select_all(self):
        def sel_entry(e):
            e.widget.select_range(0, "end")
            e.widget.icursor("end")
            return "break"

        for w in (
            getattr(self, "ent_message", None),
            getattr(self, "ent_seed", None),
        ):
            if w is not None:
                w.bind("<Control-a>", sel_entry)
                w.bind("<Control-A>", sel_entry)

        def sel_txt(e):
            e.widget.tag_add("sel", "1.0", "end-1c")
            return "break"

        if getattr(self, "txt_scan_log", None):
            self.txt_scan_log.bind("<Control-a>", sel_txt)
            self.txt_scan_log.bind("<Control-A>", sel_txt)

    # --- Lab tab ---
    def _build_lab_tab(self):
        t = self.tabview.tab("QR encode · damage · recover")

        top = ctk.CTkFrame(t, fg_color="transparent")
        top.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(top, text="Message", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w"
        )
        row = ctk.CTkFrame(top, fg_color="transparent")
        row.pack(fill="x", pady=(4, 0))
        self.ent_message = ctk.CTkEntry(row, placeholder_text="Type text to encode…")
        self.ent_message.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.ent_message.insert(0, "Error correction demo")
        ctk.CTkButton(row, text="Encode QR", width=120, command=self._lab_encode).pack(
            side="left"
        )

        mid = ctk.CTkFrame(t, fg_color="transparent")
        mid.pack(fill="both", expand=True)

        def col(parent, title):
            f = ctk.CTkFrame(parent, fg_color=("gray90", "gray17"), corner_radius=10)
            f.pack(side="left", fill="both", expand=True, padx=6)
            ctk.CTkLabel(
                f, text=title, font=ctk.CTkFont(size=12, weight="bold")
            ).pack(pady=(10, 6))
            lbl = ctk.CTkLabel(f, text="—", width=PREVIEW_PX, height=PREVIEW_PX, fg_color=("gray85", "gray20"))
            lbl.pack(pady=(0, 10))
            return lbl

        self.lbl_lab_clean = col(mid, "1 · Clean")
        self.lbl_lab_damaged = col(mid, "2 · Damaged")
        self.lbl_lab_recovered = col(mid, "3 · After decode")

        ctrl = ctk.CTkFrame(t, fg_color="transparent")
        ctrl.pack(fill="x", pady=12)

        ctk.CTkLabel(ctrl, text="Module flips (data area only)").grid(row=0, column=0, sticky="w")
        self.slider_flips = ctk.CTkSlider(ctrl, from_=0, to=80, number_of_steps=80, command=self._noop_slider)
        self.slider_flips.set(20)
        self.slider_flips.grid(row=0, column=1, sticky="ew", padx=12)
        self.lbl_flip_val = ctk.CTkLabel(ctrl, text="20", width=36)
        self.lbl_flip_val.grid(row=0, column=2)
        ctrl.grid_columnconfigure(1, weight=1)

        seed_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        seed_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ctk.CTkLabel(seed_row, text="Random seed (optional)").pack(side="left", padx=(0, 8))
        self.ent_seed = ctk.CTkEntry(seed_row, width=100, placeholder_text="e.g. 42")
        self.ent_seed.pack(side="left", padx=(0, 8))
        ctk.CTkButton(seed_row, text="Apply damage", command=self._lab_damage).pack(side="left", padx=4)
        ctk.CTkButton(
            seed_row,
            text="Recover (mask + Reed–Solomon)",
            fg_color=("#1f538d", "#1f538d"),
            command=self._lab_recover,
        ).pack(side="left", padx=4)

        self.lbl_lab_status = ctk.CTkLabel(
            t,
            text="Encode a QR, then damage it and run the custom decoder.",
            font=ctk.CTkFont(size=13),
            text_color=("gray30", "gray75"),
            anchor="w",
            justify="left",
        )
        self.lbl_lab_status.pack(fill="x", pady=(4, 0))

        self.slider_flips.configure(command=self._on_flip_slider)

    def _noop_slider(self, _v):
        pass

    def _on_flip_slider(self, v):
        self.lbl_flip_val.configure(text=str(int(float(v))))

    def _set_lab_image(self, label: ctk.CTkLabel, pil: Image.Image | None):
        if pil is None:
            label.configure(image=None, text="—")
            return
        ctk_img = _pil_to_ctk(pil)
        label.configure(image=ctk_img, text="")
        if label is self.lbl_lab_clean:
            self._img_clean = ctk_img
        elif label is self.lbl_lab_damaged:
            self._img_damaged = ctk_img
        else:
            self._img_recovered = ctk_img
        label.image = ctk_img

    def _lab_encode(self):
        text = self.ent_message.get().strip()
        if not text:
            messagebox.showwarning("Encode", "Enter a non-empty message.")
            return
        try:
            self._clean_modules, clean_pil = build_qr_modules_and_image(text)
            self._damaged_modules = None
            self._last_clean_text = text
        except Exception as e:
            messagebox.showerror("Encode", str(e))
            return

        self._set_lab_image(self.lbl_lab_clean, clean_pil)
        self._set_lab_image(self.lbl_lab_damaged, None)
        self._set_lab_image(self.lbl_lab_recovered, None)
        self.lbl_lab_status.configure(
            text=f"Encoded: {len(text)} characters · {len(self._clean_modules)}×{len(self._clean_modules)} modules. "
            "Adjust flips and press “Apply damage”.",
            text_color=("gray20", "gray80"),
        )

    def _lab_damage(self):
        if not self._clean_modules:
            messagebox.showinfo("Damage", "Encode a QR first.")
            return
        import random

        n = int(self.slider_flips.get())
        seed_txt = self.ent_seed.get().strip()
        rng = random.Random(int(seed_txt)) if seed_txt.isdigit() else random.Random()

        self._damaged_modules = damage_qr_modules(self._clean_modules, n, rng=rng)
        pil = modules_to_pil(self._damaged_modules, box_size=RENDER_BOX, border=RENDER_BORDER)
        self._set_lab_image(self.lbl_lab_damaged, pil)
        self._set_lab_image(self.lbl_lab_recovered, None)
        self.lbl_lab_status.configure(
            text=f"Damaged: flipped {min(n, self._count_flips())} data modules (finder & timing preserved). "
            "Run “Recover” to run our masking + Reed–Solomon pipeline on a rendered image.",
            text_color=("gray20", "gray80"),
        )

    def _count_flips(self) -> int:
        if not self._clean_modules or not self._damaged_modules:
            return 0
        return sum(
            1
            for r in range(len(self._clean_modules))
            for c in range(len(self._clean_modules[r]))
            if self._clean_modules[r][c] != self._damaged_modules[r][c]
        )

    def _lab_recover(self):
        if not self._damaged_modules:
            messagebox.showinfo("Recover", "Apply damage first (or encode then damage).")
            return

        report = decode_qr_from_modules_report(self._damaged_modules)

        if report["ok"]:
            recovered = report["text"] or ""
            ok_match = recovered == self._last_clean_text
            self.lbl_lab_status.configure(
                text=(
                    f"Decoder: success · mask {report['mask']} · {report['detail']}\n"
                    f"Exact match: {'yes' if ok_match else 'no (decoder or framing limits)'} · Payload: {recovered!r}"
                ),
                text_color=("#2ecc71", "#58d68d") if ok_match else ("#d35400", "#f39c12"),
            )
            try:
                _, pil_r = build_qr_modules_and_image(recovered)
                self._set_lab_image(self.lbl_lab_recovered, pil_r)
            except Exception:
                self._set_lab_image(self.lbl_lab_recovered, None)
        else:
            self.lbl_lab_status.configure(
                text=f"Decoder failed · {report['detail']}\n"
                f"({report.get('detector_note', '')})\n"
                "Tip: reduce module flips or set a fixed seed and try again — capacity depends on QR version.",
                text_color=("#c0392b", "#e74c3c"),
            )
            self._set_lab_image(self.lbl_lab_recovered, None)

    # --- Scan tab ---
    def _build_scan_tab(self):
        t = self.tabview.tab("Scan QR (our decoder)")

        bar = ctk.CTkFrame(t, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 10))

        self.lbl_scan_path = ctk.CTkLabel(
            bar, text="No file loaded", anchor="w", font=ctk.CTkFont(size=12)
        )
        self.lbl_scan_path.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(bar, text="Open image…", width=120, command=self._scan_pick).pack(
            side="right", padx=(8, 0)
        )
        ctk.CTkButton(
            bar,
            text="Run our pipeline",
            width=140,
            fg_color=("#1f538d", "#1f538d"),
            command=self._scan_run,
        ).pack(side="right")

        body = ctk.CTkFrame(t, fg_color="transparent")
        body.pack(fill="both", expand=True)

        left = ctk.CTkFrame(body, fg_color=("gray90", "gray17"), corner_radius=10, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="Preview", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 6))
        self.lbl_scan_img = ctk.CTkLabel(left, text="—", width=280, height=280, fg_color=("gray85", "gray20"))
        self.lbl_scan_img.pack(pady=(0, 12))

        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="Report", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
        self.txt_scan_log = ctk.CTkTextbox(right, font=ctk.CTkFont(family="Consolas", size=13))
        self.txt_scan_log.pack(fill="both", expand=True, pady=(6, 0))
        self._scan_file: Path | None = None

    def _scan_pick(self):
        path = filedialog.askopenfilename(
            title="QR image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        self._scan_file = Path(path)
        self.lbl_scan_path.configure(text=str(self._scan_file))

        try:
            img = Image.open(self._scan_file).convert("RGB")
            ctk_img = _pil_to_ctk(img, max_side=280)
            self.lbl_scan_img.configure(image=ctk_img, text="")
            self.lbl_scan_img.image = ctk_img
        except Exception as e:
            messagebox.showerror("Preview", str(e))

        self.txt_scan_log.delete("1.0", "end")
        self.txt_scan_log.insert("1.0", "Press “Run our pipeline” to decode with masking + Reed–Solomon.")

    def _scan_run(self):
        if not self._scan_file or not self._scan_file.is_file():
            messagebox.showinfo("Scan", "Choose an image first.")
            return

        report = decode_qr_full_report(self._scan_file)
        self.txt_scan_log.delete("1.0", "end")

        lines = [
            f"Detector path: {report.get('detector_note', '')}",
            f"Success: {report['ok']}",
            f"Detail: {report['detail']}",
        ]
        if report.get("mask") is not None:
            lines.append(f"Winning mask index: {report['mask']}")
        if report.get("text"):
            lines.append("")
            lines.append("--- Payload ---")
            lines.append(report["text"])

        self.txt_scan_log.insert("1.0", "\n".join(lines))

        if report["ok"]:
            messagebox.showinfo("Scan", "Decoded successfully — see report.")
        else:
            messagebox.showwarning("Scan", "Decoder did not recover a payload — see report.")


if __name__ == "__main__":
    app = ReedSolomonGUI()
    app.mainloop()
