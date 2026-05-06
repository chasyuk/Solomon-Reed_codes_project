import cv2
import numpy as np
from reed_solomon import ReedSolomon
from decode import DecodeReedSolomon
from galois_field import GaloisField

# ==========================================
# КЛАСИ-ОБГОРТКИ (З бронею від ділення на нуль)
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

            r_prev = r_curr
            r_curr = r_next
            a_prev = a_curr
            a_curr = a_next

        gamma_scalar = a_curr[0]
        # 🛡️ БРОНЯ: Захист від KeyError: '0b0'
        if gamma_scalar.coeffs == 0:
            raise ZeroDivisionError("Нульовий скаляр в алгоритмі Евкліда.")

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

        if all_zero:
            return received_poly

        try:
            gamma, lamda = self._euclidean_algorithm(None, syndromes)
        except ZeroDivisionError:
            raise ValueError("Математичний збій Евкліда. Хибна маска.")

        positions = self._error_location(lamda)

        if positions:
            try:
                err_vals = self._forney_algorithm(positions, gamma, lamda)
            except ZeroDivisionError:
                raise ValueError("Математичний збій Форні. Хибна маска.")
            if not err_vals:
                raise ValueError("Забагато помилок для виправлення.")
            return self._correct_errors(received_poly, err_vals)
        else:
            raise ValueError("Неможливо знайти помилки. Хибна маска.")

# ==========================================
# ОСНОВНА ЛОГІКА QR
# ==========================================

def get_grid_size(aligned_image):
    """
    Сканує перші 15 рядків і шукає найдовшу безперервну чорну лінію.
    Це 100% гарантує правильне вимірювання верхньої рамки "ока".
    """
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
            elif started:
                break
        if current_black > max_black:
            max_black = current_black

    if max_black == 0:
        return 25

    module_size_pixels = max_black / 7.0
    estimated_size = round(width / module_size_pixels)
    version = round((estimated_size - 17) / 4)
    version = max(1, min(40, version))

    return version * 4 + 17

def extract_aligned_qr(image_path):
    image = cv2.imread(image_path)
    if image is None: return None

    try:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            "detect.prototxt", "detect.caffemodel",
            "sr.prototxt", "sr.caffemodel"
        )
    except AttributeError:
        return None

    _, points = detector.detectAndDecode(image)
    if not points: return None

    pts = points[0]
    widthA = np.linalg.norm(pts[2] - pts[3])
    widthB = np.linalg.norm(pts[1] - pts[0])
    heightA = np.linalg.norm(pts[1] - pts[2])
    heightB = np.linalg.norm(pts[0] - pts[3])
    side_length = max(int(widthA), int(widthB), int(heightA), int(heightB))

    dst = np.array([[0, 0], [side_length - 1, 0], [side_length - 1, side_length - 1], [0, side_length - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (side_length, side_length))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)

    return binary_image

def unmask_qr_matrix(aligned_image, mask_type):
    if aligned_image is None: return None

    grid_size = get_grid_size(aligned_image)
    small_grid = cv2.resize(aligned_image, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    bit_matrix = (small_grid == 0).astype(int)

    mask = np.zeros((grid_size, grid_size), dtype=int)
    for r in range(grid_size):
        for c in range(grid_size):
            if mask_type == 0: condition = (r + c) % 2 == 0
            elif mask_type == 1: condition = r % 2 == 0
            elif mask_type == 2: condition = c % 3 == 0
            elif mask_type == 3: condition = (r + c) % 3 == 0
            elif mask_type == 4: condition = ((r // 2) + (c // 3)) % 2 == 0
            elif mask_type == 5: condition = ((r * c) % 2) + ((r * c) % 3) == 0
            elif mask_type == 6: condition = (((r * c) % 2) + ((r * c) % 3)) % 2 == 0
            elif mask_type == 7: condition = (((r + c) % 2) + ((r * c) % 3)) % 2 == 0
            else: condition = False

            if condition: mask[r, c] = 1

    mask[0:9, 0:9] = 0
    mask[0:9, grid_size-8:grid_size] = 0
    mask[grid_size-8:grid_size, 0:9] = 0

    return bit_matrix ^ mask

def extract_data_bits(unmasked_matrix):
    grid_size = unmasked_matrix.shape[0]
    reserved = np.zeros((grid_size, grid_size), dtype=int)

    reserved[0:9, 0:9] = 1
    reserved[0:9, grid_size-8:grid_size] = 1
    reserved[grid_size-8:grid_size, 0:9] = 1

    reserved[6, :] = 1
    reserved[:, 6] = 1

    if grid_size >= 25:
        center = grid_size - 7
        reserved[center-2 : center+3, center-2 : center+3] = 1

    data_bits = []
    row = grid_size - 1
    col = grid_size - 1
    upward = True

    while col > 0:
        if col == 6: col -= 1

        for c in [col, col - 1]:
            if not reserved[row, c]:
                data_bits.append(int(unmasked_matrix[row, c]))

        if upward:
            row -= 1
            if row < 0:
                upward = False
                row = 0
                col -= 2
        else:
            row += 1
            if row >= grid_size:
                upward = True
                row = grid_size - 1
                col -= 2

    return data_bits

def parse_bits_to_codewords(data_bits):
    codewords = []
    for i in range(0, len(data_bits), 8):
        byte_chunk = data_bits[i : i + 8]
        if len(byte_chunk) == 8:
            byte_val = 0
            for bit in byte_chunk:
                byte_val = (byte_val << 1) | bit
            codewords.append(byte_val)
    return codewords

def heal_qr_data(codewords):
    total = len(codewords)
    blocks_map = {
        26: [(1, 7), (1, 10), (1, 13), (1, 17)],       # Версія 1
        44: [(1, 10), (1, 16), (1, 22), (1, 28)],      # Версія 2
        70: [(1, 15), (1, 26), (2, 18), (2, 22)],      # Версія 3
        100: [(1, 20), (2, 18), (2, 26), (4, 16)]      # Версія 4
    }

    configs = blocks_map.get(total, [(1, 7)])

    for num_blocks, ec_per_block in configs:
        total_ec = ec_per_block * num_blocks
        total_data = total - total_ec
        if total_data <= 0: continue

        data_per_block = total_data // num_blocks
        blocks_data = [[] for _ in range(num_blocks)]
        blocks_ec = [[] for _ in range(num_blocks)]

        idx = 0
        try:
            for i in range(data_per_block):
                for b in range(num_blocks):
                    blocks_data[b].append(codewords[idx])
                    idx += 1

            for i in range(ec_per_block):
                for b in range(num_blocks):
                    blocks_ec[b].append(codewords[idx])
                    idx += 1
        except IndexError:
            continue

        all_healed = True
        healed_data_chunks = [[] for _ in range(num_blocks)]

        for b in range(num_blocks):
            block_poly = QR_ReedSolomon(ec_count=ec_per_block)
            block_cw = blocks_data[b] + blocks_ec[b]

            degree = len(block_cw) - 1
            for byte_val in block_cw:
                block_poly[degree] = byte_val
                degree -= 1

            decoder = QR_Decoder(block_poly)
            try:
                final_poly = decoder.decode()
                for deg in range(len(block_cw) - 1, ec_per_block - 1, -1):
                    healed_data_chunks[b].append(final_poly[deg].coeffs)
            except ValueError:
                all_healed = False
                break

        if all_healed:
            final_clean_data = []
            for b in range(num_blocks):
                final_clean_data.extend(healed_data_chunks[b])
            return final_clean_data

    return None

def extract_and_decode_text(clean_data):
    bit_string = "".join(f"{byte:08b}" for byte in clean_data)
    idx = 0
    final_text = ""
    alpha_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

    while idx + 4 <= len(bit_string):
        mode = int(bit_string[idx:idx+4], 2)
        idx += 4

        if mode == 0: break
        elif mode == 7: # ECI
            if idx < len(bit_string):
                if bit_string[idx] == '0': idx += 8
                elif bit_string[idx:idx+2] == '10': idx += 16
                elif bit_string[idx:idx+3] == '110': idx += 24
                else: idx += 8
            continue
        elif mode in [3, 5, 9]: # Інші службові (FNC1)
            idx += (16 if mode == 3 else 8)
            continue

        elif mode == 1: # NUMERIC
            length = int(bit_string[idx:idx+10], 2)
            idx += 10
            for _ in range(length // 3):
                final_text += f"{int(bit_string[idx:idx+10], 2):03d}"
                idx += 10
            remainder = length % 3
            if remainder == 2:
                final_text += f"{int(bit_string[idx:idx+7], 2):02d}"
                idx += 7
            elif remainder == 1:
                final_text += f"{int(bit_string[idx:idx+4], 2):d}"
                idx += 4

        elif mode == 2: # ALPHANUMERIC
            length = int(bit_string[idx:idx+9], 2)
            idx += 9
            for _ in range(length // 2):
                chunk = int(bit_string[idx:idx+11], 2)
                final_text += alpha_chars[chunk // 45] + alpha_chars[chunk % 45]
                idx += 11
            if length % 2 == 1:
                final_text += alpha_chars[int(bit_string[idx:idx+6], 2)]
                idx += 6

        elif mode == 4: # BYTE
            length = int(bit_string[idx:idx+8], 2)
            idx += 8
            byte_array = bytearray()
            for _ in range(length):
                byte_array.append(int(bit_string[idx:idx+8], 2))
                idx += 8
            try:
                final_text += byte_array.decode('utf-8')
            except UnicodeDecodeError:
                final_text += byte_array.decode('latin-1')

        elif mode == 8: # KANJI
            length = int(bit_string[idx:idx+8], 2)
            idx += 8
            kanji_bytes = bytearray()
            for _ in range(length):
                chunk = int(bit_string[idx:idx+13], 2)
                idx += 13
                high = chunk // 0xC0
                low = chunk % 0xC0
                high += 0x81 if high <= 0x1E else 0xC1
                low += 0x40 + (1 if low >= 0x7F else 0)
                kanji_bytes.extend([high, low])
            try:
                final_text += kanji_bytes.decode('shift_jis')
            except Exception:
                pass
        else:
            raise ValueError(f"Невідомий режим: {mode}")

    if not final_text:
        raise ValueError("Розшифрований текст порожній!")

    return final_text

def process_qr_pipeline(image_path):
    print(f"\n--- Початок обробки: {image_path} ---")
    aligned_img = extract_aligned_qr(image_path)
    if aligned_img is None:
        return

    for test_mask in range(8):
        unmasked = unmask_qr_matrix(aligned_img, mask_type=test_mask)
        raw_bits = extract_data_bits(unmasked)
        codewords = parse_bits_to_codewords(raw_bits)
        clean_data = heal_qr_data(codewords)

        if clean_data is not None:
            try:
                text = extract_and_decode_text(clean_data)
                print(f"🎯 УСПІХ! Справжня маска: {test_mask}")
                print("\n==================================")
                print(f"🎉 ТАЄМНИЙ ТЕКСТ: {text}")
                print("==================================")
                return
            except (ValueError, IndexError):
                pass

    print("❌ Жодна маска не пройшла перевірку. Код занадто сильно пошкоджений.")

# Запуск
if __name__ == "__main__":
    test_files = [
        "test_data/image063.jpg",
        "test_data/image064.jpg",
        "test_data/1.jpg",
        "test_data/124.jpg",
        "test_data/163.jpg"
    ]
    for file in test_files:
        process_qr_pipeline(file)
