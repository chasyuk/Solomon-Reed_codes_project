import cv2
import numpy as np

def get_grid_size(aligned_image):
    """
    Рахує пікселі верхнього 'ока' і математично вираховує розмір сітки.
    (Надійна версія, що ігнорує білі рамки).
    """
    width = aligned_image.shape[1]
    scan_y = 3 if aligned_image.shape[0] > 10 else 0

    black_pixels = 0
    started_counting = False

    for x in range(width):
        if aligned_image[scan_y, x] == 0:
            started_counting = True
            black_pixels += 1
        elif started_counting:
            break

    if black_pixels == 0:
        print("⚠️ Увага: Не вдалося виміряти око. Використовуємо розмір за замовчуванням (25x25).")
        return 25

    module_size_pixels = black_pixels / 7.0
    estimated_size = round(width / module_size_pixels)

    version = round((estimated_size - 17) / 4)
    version = max(1, version)

    return version * 4 + 17

def detect_mask_type(bit_matrix):
    """
    Читає службові пікселі навколо лівого верхнього ока і дізнається номер маски.
    """
    b2 = bit_matrix[2, 8] ^ 1
    b1 = bit_matrix[3, 8] ^ 0
    b0 = bit_matrix[4, 8] ^ 1

    mask_type = (b2 << 2) | (b1 << 1) | b0
    return mask_type

def extract_aligned_qr(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Помилка завантаження зображення.")
        return None

    try:
        detector = cv2.wechat_qrcode_WeChatQRCode(
            "detect.prototxt", "detect.caffemodel",
            "sr.prototxt", "sr.caffemodel"
        )
    except AttributeError:
        print("❌ Помилка: WeChatQRCode не знайдено.")
        return None

    res, points = detector.detectAndDecode(image)
    if not points:
        print("❌ QR-код не знайдено.")
        return None

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
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return binary_image

def unmask_qr_matrix(aligned_image):
    if aligned_image is None:
        return None

    grid_size = get_grid_size(aligned_image)
    print(f"✅ Визначено розмір сітки: {grid_size}x{grid_size}")

    small_grid = cv2.resize(aligned_image, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
    bit_matrix = (small_grid == 0).astype(int)

    mask_type = detect_mask_type(bit_matrix)
    print(f"✅ Визначено маску: Тип {mask_type}")

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

    unmasked_matrix = bit_matrix ^ mask
    print("✅ Маску успішно знято! Дані готові до зчитування.")

    return unmasked_matrix

def extract_data_bits(unmasked_matrix):
    """
    Проходить матрицю 'змійкою', оминаючи системні зони, і збирає біти.
    """
    grid_size = unmasked_matrix.shape[0]
    reserved = np.zeros((grid_size, grid_size), dtype=int)

    reserved[0:9, 0:9] = 1
    reserved[0:9, grid_size-8:grid_size] = 1
    reserved[grid_size-8:grid_size, 0:9] = 1

    reserved[6, :] = 1
    reserved[:, 6] = 1


    data_bits = []

    row = grid_size - 1
    col = grid_size - 1
    upward = True

    while col > 0:
        if col == 6:
            col -= 1

        for c in [col, col - 1]:
            if not reserved[row, c]:
                data_bits.append(unmasked_matrix[row, c])

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

    print(f"✅ Змійка завершена! Зібрано бітів: {len(data_bits)}")
    return data_bits

def process_qr_pipeline(image_path):
    print(f"\n--- Початок обробки: {image_path} ---")

    aligned_img = extract_aligned_qr(image_path)

    if aligned_img is not None:
        final_bits = unmask_qr_matrix(aligned_img)

        if final_bits is not None:
            r = extract_data_bits(final_bits)
            print(r)

# Запуск
if __name__ == "__main__":
    process_qr_pipeline("test_data/1.jpg")
