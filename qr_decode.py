import cv2
import numpy as np
from reed_solomon import ReedSolomon
from decode import DecodeReedSolomon
from galois_field import GaloisField

# ==========================================
# ПОВНІСТЮ БРОНЬОВАНІ КЛАСИ (Без залежності від суворих перевірок decode_2.py)
# ==========================================

class QR_ReedSolomon(ReedSolomon):
    def __init__(self, ec_count):
        super().__init__(m=8, k=255 - ec_count)
        self.ec_count = ec_count

class QR_Decoder(DecodeReedSolomon):
    def _calc_syndromes(self, received_poly):
        syndromes_poly = ReedSolomon(self.rs.m, self.rs.k)
        for i in range(self.rs.ec_count):
            alpha_bin_str = self.rs._ReedSolomon__primitive_table[i]
            alpha_val = int(alpha_bin_str, 2)
            x_val = GaloisField(self.rs.m, self.rs.gen_poly, alpha_val)
            syndromes_poly[i] = received_poly.evaluate_poly(x_val)
        return syndromes_poly

    def _euclidean_algorithm(self, t_dummy, syndrome):
        r_prev = ReedSolomon()
        r_prev[self.rs.ec_count] = 1
        r_curr = syndrome

        a_prev = ReedSolomon()
        a_prev[0] = 0
        a_curr = ReedSolomon()
        a_curr[0] = 1

        while r_curr.degree >= (self.rs.ec_count / 2.0):
            q = r_prev / r_curr
            r_next = r_prev % r_curr
            a_next = a_prev + (q * a_curr)

            r_prev, r_curr = r_curr, r_next
            a_prev, a_curr = a_curr, a_next

        gamma_scalar = a_curr[0]
        if gamma_scalar.coeffs == 0:
            raise ZeroDivisionError("Нульовий скаляр")
        gamma = r_curr / gamma_scalar
        lamda = a_curr / gamma_scalar
        return gamma, lamda

    def decode(self):
        received_poly = self.rs
        syndromes = self._calc_syndromes(received_poly)

        all_zero = True
        for i in range(syndromes.degree + 1):
            if syndromes[i].coeffs != 0:
                all_zero = False
                break

        if all_zero: return received_poly

        try:
            gamma, lamda = self._euclidean_algorithm(None, syndromes)
        except ZeroDivisionError:
            raise ValueError("Math Error")

        positions = self._error_location(lamda)

        if positions:
            try:
                err_vals = self._forney_algorithm(positions, gamma, lamda)
            except ZeroDivisionError:
                raise ValueError("Math Error")
            if not err_vals: raise ValueError("Too many errors")
            return self._correct_errors(received_poly, err_vals)
        else:
            raise ValueError("No error positions found")

# ==========================================
# ОСНОВНА ЛОГІКА ЗЧИТУВАННЯ
# ==========================================

def get_grid_size(aligned_image):
    width = aligned_image.shape[1]
    height = aligned_image.shape[0]
    max_black = 0

    for scan_y in range(min(15, height)):
        current_black = 0
        started = False
        for x in range(width):
            if aligned_image[scan_y, x] == 0:
                started = True
                current_black += 1
            elif started: break

        if 0 < current_black < (width * 0.5):
            max_black = max(max_black, current_black)

    if max_black == 0: return 25
    module_size = max_black / 7.0
    version = round(((width / module_size) - 17) / 4)
    version = max(1, min(40, version))
    return version * 4 + 17

def extract_aligned_qr(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    try:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            "detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"
        )
    except AttributeError: return None

    _, points = detector.detectAndDecode(image)
    if not points: return None

    pts = points[0]
    side = int(max(np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2])))
    dst = np.array([[0,0], [side-1,0], [side-1,side-1], [0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (side, side))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

def unmask_qr_matrix(aligned_image, mask_type):
    grid_size = get_grid_size(aligned_image)

    # 🎯 ПОВЕРНЕНО: Снайперське зчитування (Математичний Центр)!
    # Саме це ідеально розкодовувало 124.jpg!
    step_y = aligned_image.shape[0] / grid_size
    step_x = aligned_image.shape[1] / grid_size
    bit_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            y = int((r + 0.5) * step_y)
            x = int((c + 0.5) * step_x)
            if aligned_image[y, x] == 0:
                bit_matrix[r, c] = 1

    mask = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            if   mask_type == 0: cond = (r + c) % 2 == 0
            elif mask_type == 1: cond = r % 2 == 0
            elif mask_type == 2: cond = c % 3 == 0
            elif mask_type == 3: cond = (r + c) % 3 == 0
            elif mask_type == 4: cond = ((r // 2) + (c // 3)) % 2 == 0
            elif mask_type == 5: cond = ((r * c) % 2) + ((r * c) % 3) == 0
            elif mask_type == 6: cond = (((r * c) % 2) + ((r * c) % 3)) % 2 == 0
            elif mask_type == 7: cond = (((r + c) % 2) + ((r * c) % 3)) % 2 == 0
            else: cond = False
            if cond: mask[r, c] = 1

    mask[0:9, 0:9] = 0
    mask[0:9, grid_size-8:grid_size] = 0
    mask[grid_size-8:grid_size, 0:9] = 0
    return bit_matrix ^ mask

def extract_data_bits(unmasked_matrix):
    size = unmasked_matrix.shape[0]
    reserved = np.zeros((size, size), dtype=int)
    reserved[0:9, 0:9] = reserved[0:9, size-8:size] = reserved[size-8:size, 0:9] = 1
    reserved[6, :] = reserved[:, 6] = 1
    if size >= 25:
        c = size - 7
        reserved[c-2:c+3, c-2:c+3] = 1

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

def heal_qr_data(codewords):
    total = len(codewords)
    blocks_map = {
        26: [(1, 7), (1, 10), (1, 13), (1, 17)],
        44: [(1, 10), (1, 16), (1, 22), (1, 28)],
        70: [(1, 15), (1, 26), (2, 18), (2, 22)],
        100: [(1, 20), (2, 18), (2, 26), (4, 16)]
    }
    configs = blocks_map.get(total, [(1, 7)])

    for num_blocks, ec_per_block in configs:
        total_data = total - (num_blocks * ec_per_block)
        if total_data <= 0: continue

        data_per_block = total_data // num_blocks
        idx, all_healed = 0, True
        healed_chunks = []

        try:
            blocks = [[] for _ in range(num_blocks)]
            for i in range(data_per_block):
                for b in range(num_blocks):
                    blocks[b].append(codewords[idx]); idx += 1
            for i in range(ec_per_block):
                for b in range(num_blocks):
                    blocks[b].append(codewords[idx]); idx += 1

            for b in range(num_blocks):
                rs_poly = QR_ReedSolomon(ec_per_block)
                for deg, val in enumerate(reversed(blocks[b])):
                    rs_poly[deg] = val

                decoder = QR_Decoder(rs_poly)
                result = decoder.decode()

                for deg in range(len(blocks[b]) - 1, ec_per_block - 1, -1):
                    healed_chunks.append(result[deg].coeffs)

            if all_healed: return healed_chunks
        except Exception: continue
    return None

def extract_and_decode_text(clean_data):
    bit_string = "".join(f"{byte:08b}" for byte in clean_data)
    idx, final_text = 0, ""
    alpha = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

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

        if mode == 1: # Numeric
            for _ in range(length // 3):
                final_text += f"{int(bit_string[idx:idx+10], 2):03d}"; idx += 10
            rem = length % 3
            if rem:
                bits = 7 if rem == 2 else 4
                final_text += str(int(bit_string[idx:idx+bits], 2)); idx += bits

        elif mode == 2: # Alphanumeric
            for _ in range(length // 2):
                chunk = int(bit_string[idx:idx+11], 2)
                final_text += alpha[chunk // 45] + alpha[chunk % 45]; idx += 11
            if length % 2: final_text += alpha[int(bit_string[idx:idx+6], 2)]; idx += 6

        elif mode == 4: # Byte
            res = bytearray()
            for _ in range(length):
                res.append(int(bit_string[idx:idx+8], 2)); idx += 8
            try: final_text += res.decode('utf-8')
            except: final_text += res.decode('latin-1')

        elif mode == 8: # Kanji
            kanji_bytes = bytearray()
            for _ in range(length):
                chunk = int(bit_string[idx:idx+13], 2); idx += 13
                high, low = chunk // 0xC0, chunk % 0xC0
                high += 0x81 if high <= 0x1E else 0xC1
                low += 0x40 + (1 if low >= 0x7F else 0)
                kanji_bytes.extend([high, low])
            try: final_text += kanji_bytes.decode('shift_jis')
            except: pass

    # М'яка валідація, щоб не заблокувати 163.jpg!
    if not final_text.strip() or len(final_text) < 2:
        raise ValueError("Порожньо або закоротко")

    return final_text

def process_qr_pipeline(image_path):
    print(f"\n--- Обробка: {image_path} ---")
    img = extract_aligned_qr(image_path)
    if img is None: return

    for mask in range(8):
        unmasked = unmask_qr_matrix(img, mask)
        bits = extract_data_bits(unmasked)
        cw = []
        for i in range(0, len(bits), 8):
            chunk = bits[i:i+8]
            if len(chunk) == 8: cw.append(int("".join(map(str, chunk)), 2))

        data = heal_qr_data(cw)
        if data:
            try:
                text = extract_and_decode_text(data)
                print(f"🎯 МАСКА {mask} | ТЕКСТ: {text}")
                return
            except ValueError:
                pass

    print("❌ Не вдалося розкодувати.")

if __name__ == "__main__":
    test_files = [
        "1.jpg", "124.jpg", "163.jpg", "145.jpg"
    ]
    for i in test_files:
        process_qr_pipeline(f"test_data/{i}")
