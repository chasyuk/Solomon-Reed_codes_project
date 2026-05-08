import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from qrcode import base as qr_base
from qrcode import constants as qr_constants

from core.reed_solomon import ReedSolomon
from core.decode import DecodeReedSolomon
from core.galois_field import GaloisField

# Construct paths to model files relative to repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"

try:
    QR_DETECTOR = cv2.wechat_qrcode_WeChatQRCode(
        str(MODEL_DIR / "detect.prototxt"),
        str(MODEL_DIR / "detect.caffemodel"),
        str(MODEL_DIR / "sr.prototxt"),
        str(MODEL_DIR / "sr.caffemodel")
    )
except (AttributeError, cv2.error):
    QR_DETECTOR = None

class QR_ReedSolomon(ReedSolomon):
    def __init__(self, ec_count):
        super().__init__(m=8, k=255 - ec_count)
        self.ec_count = ec_count

class QR_Decoder(DecodeReedSolomon):
    def _calc_syndromes(self, received_poly):
        syndromes_poly = ReedSolomon(self.rs.m, self.rs.k)
        for i in range(self.rs.ec_count):
            alpha_val = int(self.rs.primitive_table[i], 2)
            x_val = GaloisField(self.rs.m, self.rs.gen_poly, alpha_val)
            syndromes_poly[i] = received_poly.evaluate_poly(x_val)
        return syndromes_poly

    def _euclidean_algorithm(self, _t_dummy, syndrome):
        r_prev, a_prev = ReedSolomon(), ReedSolomon()
        r_curr, a_curr = syndrome, ReedSolomon()
        r_prev[self.rs.ec_count] = 1
        a_prev[0], a_curr[0] = 0, 1

        while r_curr.degree >= self.rs.t:
            q = r_prev / r_curr
            r_prev, r_curr = r_curr, r_prev % r_curr
            a_prev, a_curr = a_curr, a_prev + (q * a_curr)

        if a_curr[0].coeffs == 0:
            raise ZeroDivisionError()
        return r_curr / a_curr[0], a_curr / a_curr[0]

    def decode(self):
        received_poly = self.rs
        syndromes = self._calc_syndromes(received_poly)

        all_zero = True
        for i in range(syndromes.degree + 1):
            if syndromes[i].coeffs != 0:
                all_zero = False; break
        if all_zero: return received_poly

        try:
            gamma, lamda = self._euclidean_algorithm(None, syndromes)
        except ZeroDivisionError as exc:
            raise ValueError("Math Error") from exc

        positions = self._error_location(lamda)

        if not positions or len(positions) != lamda.degree:
            raise ValueError("Uncorrectable")

        try:
            err_vals = self._forney_algorithm(positions, gamma, lamda)
        except ZeroDivisionError as exc:
            raise ValueError("Math Error") from exc

        if not err_vals: raise ValueError("Too many errors")
        return self._correct_errors(received_poly, err_vals)


def get_grid_size(aligned_image):
    width, height = aligned_image.shape[1], aligned_image.shape[0]
    max_black = 0
    for scan_y in range(min(15, height)):
        current_black, started = 0, False
        for x in range(width):
            if aligned_image[scan_y, x] == 0:
                started, current_black = True, current_black + 1
            elif started: break
        if 0 < current_black < (width * 0.5):
            max_black = max(max_black, current_black)

    if max_black == 0: return 25
    version = round(((width / (max_black / 7.0)) - 17) / 4)
    return max(1, min(40, version)) * 4 + 17


def qr_xor_mask_bitmap(grid_size, mask_type):
    """QR data mask XOR pattern (masked modules only; finder/timing masked out)."""
    mask = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            if mask_type == 0:
                cond = (r + c) % 2 == 0
            elif mask_type == 1:
                cond = r % 2 == 0
            elif mask_type == 2:
                cond = c % 3 == 0
            elif mask_type == 3:
                cond = (r + c) % 3 == 0
            elif mask_type == 4:
                cond = ((r // 2) + (c // 3)) % 2 == 0
            elif mask_type == 5:
                cond = ((r * c) % 2) + ((r * c) % 3) == 0
            elif mask_type == 6:
                cond = (((r * c) % 2) + ((r * c) % 3)) % 2 == 0
            elif mask_type == 7:
                cond = (((r + c) % 2) + ((r * c) % 3)) % 2 == 0
            else:
                cond = False
            if cond:
                mask[r, c] = 1

    mask[0:9, 0:9] = 0
    mask[0:9, grid_size - 8 : grid_size] = 0
    mask[grid_size - 8 : grid_size, 0:9] = 0
    return mask


def sample_aligned_image_to_bitmap(aligned_image, grid_size):
    """Map a binarised, deskewed QR (0=black) to a grid_size×grid_size bit matrix (1=dark)."""
    h, w = aligned_image.shape[:2]
    step_y = h / grid_size
    step_x = w / grid_size
    bit_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            y = min(int((r + 0.5) * step_y), h - 1)
            x = min(int((c + 0.5) * step_x), w - 1)
            if aligned_image[y, x] == 0:
                bit_matrix[r, c] = 1
    return bit_matrix


def ordered_qr_grid_sizes(aligned_image):
    """
    Try the heuristic version first, then nearby versions, then all Model 2 sizes.
    Reduces failures when finder-based width guess is off by a few modules.
    """
    guess = get_grid_size(aligned_image)
    v_cent = max(1, min(40, round((guess - 17) / 4)))
    ordered = []
    seen = set()
    for d in sorted(range(-10, 11), key=abs):
        vv = v_cent + d
        if vv < 1 or vv > 40:
            continue
        g = vv * 4 + 17
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    for vv in range(1, 41):
        g = vv * 4 + 17
        if g not in seen:
            ordered.append(g)
    return ordered


def _warp_corners_to_square(image_bgr, pts):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape == (1, 4, 2):
        pts = pts.reshape(4, 2)
    elif pts.shape != (4, 2):
        return None
    side = int(
        max(
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
        )
    )
    if side < 10:
        return None
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image_bgr, m, (side, side))
    return cv2.adaptiveThreshold(
        cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        5,
    )


def extract_aligned_qr_opencv(image_bgr):
    """Straighten QR using built-in OpenCV detector (fallback when WeChat model is missing)."""
    if image_bgr is None:
        return None
    det = cv2.QRCodeDetector()
    ok, corners = det.detect(image_bgr)
    if not ok or corners is None:
        return None
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return _warp_corners_to_square(image_bgr, pts)


def extract_aligned_qr(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    if QR_DETECTOR is not None:
        _, points = QR_DETECTOR.detectAndDecode(image)
        if points:
            out = _warp_corners_to_square(image, points[0])
            if out is not None:
                return out

    return extract_aligned_qr_opencv(image)

def unmask_qr_matrix(aligned_image, mask_type):
    grid_size = get_grid_size(aligned_image)
    bit_matrix = sample_aligned_image_to_bitmap(aligned_image, grid_size)
    return np.bitwise_xor(bit_matrix, qr_xor_mask_bitmap(grid_size, mask_type))

def extract_data_bits(unmasked_matrix):
    size = unmasked_matrix.shape[0]
    reserved = np.zeros((size, size), dtype=int)
    reserved[0:9, 0:9] = reserved[0:9, size-8:size] = reserved[size-8:size, 0:9] = 1
    reserved[6, :] = reserved[:, 6] = 1

    if size >= 45:
        reserved[0:6, size-11:size-8] = 1
        reserved[size-11:size-8, 0:6] = 1

    centers = []
    if size == 25: centers = [6, 18]
    elif size == 29: centers = [6, 22]
    elif size == 33: centers = [6, 26]
    elif size == 37: centers = [6, 30]
    elif size == 41: centers = [6, 34]
    elif size == 45: centers = [6, 22, 38]
    elif size == 49: centers = [6, 24, 42]
    elif size == 53: centers = [6, 26, 46]

    for r in centers:
        for c in centers:
            if (r < 9 and c < 9) or (r < 9 and c > size-9) or (r > size-9 and c < 9): continue
            reserved[r-2:r+3, c-2:c+3] = 1

    data_bits = []
    row, col, upward = size-1, size-1, True
    while col > 0:
        if col == 6: col -= 1
        for c in [col, col-1]:
            if not reserved[row, c]: data_bits.append(int(unmasked_matrix[row, c]))
        if upward:
            row -= 1
            if row < 0: row, col, upward = 0, col-2, False
        else:
            row += 1
            if row >= size: row, col, upward = size-1, col-2, True
    return data_bits


def bitmap_to_codewords(unmasked_matrix):
    bits = extract_data_bits(unmasked_matrix)
    cw = []
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        if len(chunk) == 8:
            cw.append(int("".join(map(str, chunk)), 2))
    return cw


def _deinterleave_qr_codewords(codewords, rs_blocks_tuple):
    """
    Reverse ``qr_rs_codec.create_rs_codeword_sequence``: interleaved QR stream
    → per-block data and parity lists matching ``qrcode.base.rs_blocks``.
    """
    blocks = tuple(rs_blocks_tuple)
    dc_counts = [b.data_count for b in blocks]
    ec_counts = [b.total_count - b.data_count for b in blocks]
    max_dc = max(dc_counts)
    max_ec = max(ec_counts)

    dcdata = [[] for _ in blocks]
    ecdata = [[] for _ in blocks]

    idx = 0
    total_needed = sum(b.total_count for b in blocks)
    if len(codewords) != total_needed:
        return None

    try:
        for row in range(max_dc):
            for bi, dc_n in enumerate(dc_counts):
                if row < dc_n:
                    dcdata[bi].append(codewords[idx])
                    idx += 1
        for row in range(max_ec):
            for bi, ec_n in enumerate(ec_counts):
                if row < ec_n:
                    ecdata[bi].append(codewords[idx])
                    idx += 1
    except IndexError:
        return None

    if idx != len(codewords):
        return None
    return dcdata, ec_counts, ecdata


def _decode_one_rs_block(dc_bytes, ec_bytes, ec_per_block):
    rs_poly = QR_ReedSolomon(ec_per_block)
    cw_block = dc_bytes + ec_bytes
    for deg, val in enumerate(reversed(cw_block)):
        rs_poly[deg] = val
    try:
        result = QR_Decoder(rs_poly).decode()
    except ValueError:
        return None
    data_out = []
    for deg in range(len(cw_block) - 1, ec_per_block - 1, -1):
        data_out.append(result[deg].coeffs)
    return data_out


def heal_qr_codewords_for_layout(codewords, rs_blocks_tuple):
    """
    De-interleave and RS-correct QR blocks using this repo's ``QR_Decoder``.
    Returns corrected data codewords in block order (one flat list).
    """
    split = _deinterleave_qr_codewords(codewords, rs_blocks_tuple)
    if split is None:
        return None
    dcdata, ec_counts, ecdata = split

    healed = []
    for bi in range(len(rs_blocks_tuple)):
        row = _decode_one_rs_block(dcdata[bi], ecdata[bi], ec_counts[bi])
        if row is None:
            return None
        healed.extend(row)
    return healed


def heal_qr_data(codewords, *, qr_version=None, error_correction=None):
    """
    De-interleave and RS-decode QR codewords.

    Passing ``qr_version`` uses ISO layouts from ``qrcode.base.rs_blocks`` (recommended).
    Omit ``error_correction`` to try H → Q → M → L until one layout decodes cleanly.
    """
    ecc_try_order = (
        qr_constants.ERROR_CORRECT_H,
        qr_constants.ERROR_CORRECT_Q,
        qr_constants.ERROR_CORRECT_M,
        qr_constants.ERROR_CORRECT_L,
    )

    if qr_version is not None:
        ecc_list = [error_correction] if error_correction is not None else list(ecc_try_order)
        for ec in ecc_list:
            try:
                rs_bl = qr_base.rs_blocks(qr_version, ec)
            except ValueError:
                continue
            out = heal_qr_codewords_for_layout(codewords, rs_bl)
            if out:
                return out
        return None

    # Legacy: codeword count only (ambiguous; leftover API use without version).

    total = len(codewords)
    blocks_map = {
        # Put likely multi-block ECC patterns before wrong (num_blocks, ec) pairs.
        26:  [(1, 7), (1, 10), (1, 13), (1, 17)],
        44:  [(1, 10), (1, 16), (1, 22), (1, 28)],
        70:  [(2, 22), (1, 15), (1, 26), (2, 18)],
        100: [(1, 20), (2, 18), (2, 26), (4, 16)],
        134: [(1, 26), (2, 24), (4, 18), (4, 22)],
        172: [(2, 18), (4, 16), (4, 24), (4, 28)],
        196: [(2, 20), (4, 18), (6, 18), (5, 26)],
    }
    configs = blocks_map.get(total, [(1, 7)])

    for num_blocks, ec_per_block in configs:
        total_ec = num_blocks * ec_per_block
        total_data = total - total_ec
        if total_data <= 0:
            continue

        base_len = total_data // num_blocks
        remainder = total_data % num_blocks
        data_lengths = [base_len] * (num_blocks - remainder) + [base_len + 1] * remainder

        blocks_data = [[] for _ in range(num_blocks)]
        blocks_ec = [[] for _ in range(num_blocks)]

        idx = 0
        try:
            max_len = base_len + 1 if remainder > 0 else base_len
            for r in range(max_len):
                for b in range(num_blocks):
                    if r < data_lengths[b]:
                        blocks_data[b].append(codewords[idx]); idx += 1
            for r in range(ec_per_block):
                for b in range(num_blocks):
                    blocks_ec[b].append(codewords[idx]); idx += 1
        except IndexError:
            continue

        all_healed = True
        healed_chunks = []
        for b in range(num_blocks):
            rs_poly = QR_ReedSolomon(ec_per_block)
            cw_block = blocks_data[b] + blocks_ec[b]
            for deg, val in enumerate(reversed(cw_block)):
                rs_poly[deg] = val

            try:
                result = QR_Decoder(rs_poly).decode()
                if result is None:
                    all_healed = False
                    break
                for deg in range(len(cw_block) - 1, ec_per_block - 1, -1):
                    healed_chunks.append(result[deg].coeffs)
            except ValueError:
                all_healed = False
                break

        if all_healed:
            return healed_chunks
    return None


def extract_and_decode_text(clean_data):
    bit_string = "".join(f"{byte:08b}" for byte in clean_data)
    idx, final_text = 0, ""
    alpha = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    MAX_SAFE_LENGTH = 500

    while idx + 4 <= len(bit_string):
        mode = int(bit_string[idx:idx+4], 2); idx += 4
        if mode == 0: break
        elif mode == 7: # ECI
            if idx < len(bit_string):
                if bit_string[idx] == '0': idx += 8
                elif bit_string[idx:idx+2] == '10': idx += 16
                else: idx += 24
            continue
        elif mode in [3, 5, 9]: idx += (16 if mode == 3 else 8); continue

        len_bits = {1: 10, 2: 9, 4: 8, 8: 8}.get(mode, 0)
        if not len_bits or idx + len_bits > len(bit_string): break
        length = int(bit_string[idx:idx+len_bits], 2); idx += len_bits

        if mode == 1:
            if length > MAX_SAFE_LENGTH: raise ValueError()
            for _ in range(length // 3):
                final_text += f"{int(bit_string[idx:idx+10], 2):03d}"; idx += 10
            rem = length % 3
            if rem:
                bits = 7 if rem == 2 else 4
                final_text += str(int(bit_string[idx:idx+bits], 2)); idx += bits
        elif mode == 2:
            for _ in range(length // 2):
                if idx + 11 > len(bit_string):
                    raise ValueError()
                chunk = int(bit_string[idx:idx+11], 2)
                hi, lo = divmod(chunk, 45)
                if hi >= 45 or lo >= 45:
                    raise ValueError()
                final_text += alpha[hi] + alpha[lo]
                idx += 11
            if length % 2:
                if idx + 6 > len(bit_string):
                    raise ValueError()
                tail = int(bit_string[idx:idx+6], 2)
                if tail >= 45:
                    raise ValueError()
                final_text += alpha[tail]
                idx += 6
        elif mode == 4:
            if length > MAX_SAFE_LENGTH: raise ValueError()
            res = bytearray()
            for _ in range(length):
                b = int(bit_string[idx:idx+8], 2)
                res.append(b); idx += 8
            try: final_text += res.decode('utf-8')
            except: final_text += res.decode('latin-1')
        elif mode == 8:
            kanji_bytes = bytearray()
            for _ in range(length):
                chunk = int(bit_string[idx:idx+13], 2); idx += 13
                high, low = chunk // 0xC0, chunk % 0xC0
                high += 0x81 if high <= 0x1E else 0xC1
                low += 0x40 + (1 if low >= 0x7F else 0)
                kanji_bytes.extend([high, low])
            try: final_text += kanji_bytes.decode('shift_jis')
            except: pass

    if not final_text.strip() or len(final_text) < 2: raise ValueError()
    return final_text


def decode_qr_from_modules_report(modules):
    """
    Decode from the logical QR module matrix (True = dark), skipping OpenCV.
    Same RS + zigzag extraction as ``decode_qr_full_report``.
    Requires side N = 21 + 4·k with version k in 1..40 (Model 2).
    """
    base_fail = {
        "ok": False,
        "text": None,
        "mask": None,
        "detector_note": "logical module grid (CV skipped)",
    }

    if not modules:
        return {**base_fail, "detail": "Empty module matrix."}

    h = len(modules)
    if any(len(row) != h for row in modules):
        return {**base_fail, "detail": "Module matrix must be square."}

    if h < 21 or (h - 17) % 4 != 0:
        return {
            **base_fail,
            "detail": "Invalid QR side length (expect 21 + 4·k, Model 2).",
        }

    v = (h - 17) // 4
    if not (1 <= v <= 40):
        return {
            **base_fail,
            "detail": f"QR version inferred as {v}; supported range is 1–40.",
        }

    note_joined = base_fail["detector_note"]
    bit_matrix = np.array([[1 if cell else 0 for cell in row] for row in modules], dtype=int)

    for mask in range(8):
        unmasked = np.bitwise_xor(bit_matrix, qr_xor_mask_bitmap(h, mask))
        cw = bitmap_to_codewords(unmasked)
        data = heal_qr_data(cw, qr_version=v)
        if not data:
            continue
        try:
            text = extract_and_decode_text(data)
            return {
                "ok": True,
                "text": text,
                "detail": f"Decoded from module grid; mask {mask}; RS OK (QR v{v}).",
                "mask": mask,
                "detector_note": note_joined,
            }
        except (ValueError, IndexError):
            continue

    return {
        **base_fail,
        "detail": (
            "Could not recover payload (too many errors for this ECC level, "
            "or extractor mismatch)."
        ),
    }


def process_qr_pipeline(image_path):
    img = extract_aligned_qr(image_path)
    if img is None: return

    for grid_size in ordered_qr_grid_sizes(img):
        qr_ver = max(1, min(40, (grid_size - 17) // 4))
        if qr_ver * 4 + 17 != grid_size:
            continue
        bit_matrix = sample_aligned_image_to_bitmap(img, grid_size)
        for mask in range(8):
            unmasked = np.bitwise_xor(bit_matrix, qr_xor_mask_bitmap(grid_size, mask))
            cw = bitmap_to_codewords(unmasked)
            data = heal_qr_data(cw, qr_version=qr_ver)
            if data:
                try:
                    text = extract_and_decode_text(data)
                    return text
                except (ValueError, IndexError): pass

    return "Не вдалося розкодувати."


DECODE_FAIL_UK = "Не вдалося розкодувати."


def decode_qr_full_report(image_path):
    """
    Run the full custom masking + Reed–Solomon + payload parse pipeline.
    Returns a dict with keys: ok (bool), text (str|None), detail (human message),
    mask (int|None), detector_note (str).
    """
    path = str(image_path)
    note = []
    image = cv2.imread(path)
    if image is None:
        return {"ok": False, "text": None, "detail": "Could not read image file.", "mask": None, "detector_note": ""}

    if QR_DETECTOR is not None:
        _, wpoints = QR_DETECTOR.detectAndDecode(image)
        if wpoints:
            note.append("geometry: WeChat QR detector")
        else:
            note.append("WeChat found no corners; tried OpenCV fallback")
    else:
        note.append("WeChat models unavailable; using OpenCV QR detector")

    aligned = extract_aligned_qr(path)
    if aligned is None:
        return {
            "ok": False,
            "text": None,
            "detail": "QR not found or perspective warp failed.",
            "mask": None,
            "detector_note": "; ".join(note),
        }

    for grid_size in ordered_qr_grid_sizes(aligned):
        qr_ver = max(1, min(40, (grid_size - 17) // 4))
        if qr_ver * 4 + 17 != grid_size:
            continue
        bit_matrix = sample_aligned_image_to_bitmap(aligned, grid_size)
        for mask in range(8):
            unmasked = np.bitwise_xor(bit_matrix, qr_xor_mask_bitmap(grid_size, mask))
            cw = bitmap_to_codewords(unmasked)

            data = heal_qr_data(cw, qr_version=qr_ver)
            if not data:
                continue
            try:
                text = extract_and_decode_text(data)
                return {
                    "ok": True,
                    "text": text,
                    "detail": (
                        f"Decoded ({grid_size}×{grid_size} grid, mask {mask}); "
                        "Reed–Solomon blocks recovered."
                    ),
                    "mask": mask,
                    "detector_note": "; ".join(note),
                }
            except (ValueError, IndexError):
                continue

    return {
        "ok": False,
        "text": None,
        "detail": "Could not recover payload after RS (too much damage or wrong QR).",
        "mask": None,
        "detector_note": "; ".join(note),
    }


if __name__ == "__main__":
    for i in ["2.jpg"]:
        process_qr_pipeline(f"test_data/{i}")
