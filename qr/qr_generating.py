"""QR symbol generation: layout/masking via ``qrcode``; RS parity via ``qr_rs_codec`` (``core.reed_solomon``)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import qrcode
import qrcode.base

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


# ---------------------------------------------------------------------------
# Codeword-level damage (correct implementation)
# ---------------------------------------------------------------------------

def _qr_data_module_positions(size: int) -> list[tuple[int, int]]:
    """
    Return (row, col) for every data/parity module in the true QR read order
    (right-to-left zigzag column pairs, bottom-to-top / top-to-bottom alternating).
    Skips all function-pattern modules: finder corners, separators, timing strips,
    format info cells.
    """
    def is_function(r, c, n):
        if r == 6 or c == 6:          # timing strips
            return True
        if r < 9 and c < 9:           # top-left finder + separator + format
            return True
        if r < 9 and c >= n - 8:      # top-right finder + separator + format
            return True
        if r >= n - 8 and c < 9:      # bottom-left finder + separator + format
            return True
        return False

    positions = []
    col = size - 1
    going_up = True
    while col >= 1:
        if col == 6:          # skip vertical timing strip column pair
            col -= 1
            continue
        for delta in range(size):
            row = (size - 1 - delta) if going_up else delta
            for dc in (0, 1):
                c = col - dc
                if 0 <= c < size and not is_function(row, c, size):
                    positions.append((row, c))
        going_up = not going_up
        col -= 2
    return positions


def qr_correction_capacity(version: int, error_correction=qrcode.constants.ERROR_CORRECT_H) -> int:
    """
    Total RS codeword errors correctable across **all** RS blocks for the given
    QR version and EC level.  Each block independently corrects ⌊ec_per_block/2⌋
    errors, so the total can be much larger than 16 for higher versions.
    """
    blocks = qrcode.base.rs_blocks(version, error_correction)
    return sum((block.total_count - block.data_count) // 2 for block in blocks)


def damage_qr_modules(modules, num_errors, rng=None):
    """
    Damage exactly ``num_errors`` real RS codewords in the QR module matrix.

    Modules are read in the true QR zigzag order and grouped into 8-bit
    codewords — exactly as a real decoder reads them.  Each chosen codeword
    receives a random non-zero XOR mask, so every damaged codeword is a
    guaranteed distinct RS error symbol.

    Returns a new matrix; ``modules`` is not modified.
    Stores bookkeeping on the function object:
      ``damage_qr_modules.last_total_codewords``  – total codewords in this QR
    """
    rng = rng or random.Random()
    size = len(modules)
    positions = _qr_data_module_positions(size)

    # group into 8-bit codewords in true read order
    codewords = [positions[i:i + 8] for i in range(0, len(positions) - 7, 8)]
    damage_qr_modules.last_total_codewords = len(codewords)

    n = min(int(num_errors), len(codewords))
    chosen_indices = rng.sample(range(len(codewords)), n)

    out = [list(row) for row in modules]
    for idx in chosen_indices:
        mask = rng.randint(1, 255)   # non-zero → at least 1 module flips per codeword
        for bit_pos, (r, c) in enumerate(codewords[idx]):
            if (mask >> bit_pos) & 1:
                out[r][c] = not out[r][c]

    return out

damage_qr_modules.last_total_codewords = 0


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
