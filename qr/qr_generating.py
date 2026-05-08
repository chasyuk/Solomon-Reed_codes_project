"""QR symbol generation: layout/masking via ``qrcode``; RS parity via ``qr_rs_codec`` (``core.reed_solomon``)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import qrcode

from qr.qr_rs_codec import create_data_core_rs

IMAGES_DIR = Path("images")


def _make_qr(message, version=None, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4):
    import qrcode.util as qutil

    saved = qutil.create_data
    qutil.create_data = create_data_core_rs
    try:
        qr = qrcode.QRCode(
            version=version,
            error_correction=error_correction,
            box_size=box_size,
            border=border,
        )
        qr.add_data(message)
        qr.make(fit=True)
    finally:
        qutil.create_data = saved
    return qr


def build_qr_modules(message, version=None, error_correction=qrcode.constants.ERROR_CORRECT_H):
    """Return a deep copy of the QR module matrix (list of list of bool); True = dark module."""
    qr = _make_qr(message, version=version, error_correction=error_correction)
    return [list(row) for row in qr.modules]


def build_qr_modules_and_image(message, version=None, error_correction=qrcode.constants.ERROR_CORRECT_H):
    """
    Same QR as ``build_qr_modules``, plus a PIL image from the official rasteriser
    (matches typical printed QR pixel layout better than ad-hoc drawing).
    """
    qr = _make_qr(message, version=version, error_correction=error_correction)
    modules = [list(row) for row in qr.modules]
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return modules, pil


def modules_to_pil(modules, box_size=10, border=4):
    """Rasterise a QR module matrix to an RGB PIL image (white background, black modules)."""
    from PIL import Image, ImageDraw

    n = len(modules)
    px = (n + 2 * border) * box_size
    img = Image.new("RGB", (px, px), "white")
    draw = ImageDraw.Draw(img)
    for r in range(n):
        for c in range(n):
            if not modules[r][c]:
                continue
            y0 = (border + r) * box_size
            x0 = (border + c) * box_size
            draw.rectangle([x0, y0, x0 + box_size - 1, y0 + box_size - 1], fill="black")
    return img


def damage_qr_modules(modules, num_flips, rng=None):
    """
    Flip ``num_flips`` data modules (skipping finder patterns and timing lines)
    for a visible damage demo. Returns a new matrix; ``modules`` is not modified.
    """
    rng = rng or random.Random()
    h, w = len(modules), len(modules[0])
    candidates = []
    for r in range(h):
        for c in range(w):
            if r < 9 and c < 9:
                continue
            if r < 9 and c >= w - 8:
                continue
            if r >= h - 8 and c < 9:
                continue
            if r == 6 or c == 6:
                continue
            candidates.append((r, c))
    out = [list(row) for row in modules]
    n = min(int(num_flips), len(candidates))
    rng.shuffle(candidates)
    for i in range(n):
        r, c = candidates[i]
        out[r][c] = not out[r][c]
    return out


def generate_qr_code(message, filename="my_qrcode.png"):
    qr = _make_qr(message)

    img = qr.make_image(fill_color="black", back_color="white")

    path = Path(filename)
    if not path.is_absolute() and path.parent == Path(".") and path.name == "generated_qr.png":
        IMAGES_DIR.mkdir(exist_ok=True)
        path = IMAGES_DIR / path.name

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"QR-код збережено як '{path}'!")
    return str(path)

if __name__ == "__main__":
    data = "Never gonna give you up..."
    generate_qr_code(data, "gen.jpg")
