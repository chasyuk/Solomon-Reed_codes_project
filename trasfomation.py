import random
from reed_solomon import ReedSolomon
from decode import DecodeReedSolomon

def pack_to_symbols(bit_string, m):
    remainder = len(bit_string) % m
    if remainder:
        bit_string += '0' * (m - remainder)

    return [int(bit_string[i:i+m], 2) for i in range(0, len(bit_string), m)]


def unpack_from_symbols(symbols, m, original_bit_length):
    bit_string = ''.join(format(s, f'0{m}b') for s in symbols)
    return bit_string[:original_bit_length]

def serialize(data_list):
    """[(значення, кількість_бітів), ...] → бітовий рядок"""
    return ''.join(format(val, f'0{bits}b') for val, bits in data_list)


def deserialize(bit_string, data_list):
    """Бітовий рядок → список значень за схемою data_list"""
    results, pos = [], 0
    for _, bits in data_list:
        results.append(int(bit_string[pos:pos + bits], 2))
        pos += bits
    return results

def rs_encode(symbols, m, k):
    rs = ReedSolomon(m, k)
    for i, s in enumerate(symbols):
        rs[i] = s
    rs.encode()
    return rs


def rs_inject_errors(rs, count=2):
    indices = random.sample(range(rs.degree + 1), count)
    for idx in indices:
        rs.poly[idx].coeffs ^= 1
    return indices


def rs_decode(rs, num_symbols):
    fixed = DecodeReedSolomon(rs).decode()
    offset = 2 * rs.t
    return [fixed.poly[offset + i].coeffs for i in range(num_symbols)]


def automatic_rs_handler(data_list, m=8, k=223):
    """
    Приймає [(значення, біти), ...], кодує через Reed-Solomon,
    симулює помилки та повертає відновлені дані.
    """
    bit_stream = serialize(data_list)
    symbols = pack_to_symbols(bit_stream, m)
    rs = rs_encode(symbols, m, k)
    bad_indices = rs_inject_errors(rs, count=2)

    fixed_symbols = rs_decode(rs, len(symbols))
    recovered_bits = unpack_from_symbols(fixed_symbols, m, len(bit_stream))

    return deserialize(recovered_bits, data_list)

def transfrom_to_reed_solomon_code(msg, m=8, k=223):
    if not isinstance(msg, str):
        raise ValueError("Можна трансформувати лише рядки!")

    rs = ReedSolomon(m, k)
    for i, char in zip(range(rs.gf_size), reversed([ord(c) for c in msg])):
        rs[i] = char
    return rs


def transfrom_to_string(code):
    return ''.join(
        chr(sym.coeffs) for sym in reversed(code.poly) if sym.coeffs != 0
    )


if __name__ == '__main__':
    print("=== ТЕСТ 1: ТЕКСТ ===")
    text = "hello"
    encoded = transfrom_to_reed_solomon_code(text, m=4, k=11)
    decoded = transfrom_to_string(encoded)
    print(f"'{text}' → '{decoded}'")
    assert decoded == text, f"Текст не збігається: {decoded!r}"
    print("OK")

    print("\n=== ТЕСТ 2: ДОВІЛЬНІ БІТОВІ СТРУКТУРИ ===")
    custom_data = [
        (18446744073709551615, 64), 
        (0b11,              2),
        (0b101010101010,   12),
    ]

    try:
        results = automatic_rs_handler(custom_data, m=8, k=223)

        print("\nВідновлені дані:")
        print(f"  64-bit : {results[0]}")
        print(f"  Flag   : {bin(results[1])}")
        print(f"  Struct : {bin(results[2])}")

        assert results == [d[0] for d in custom_data], "Дані не збіглися!"
        print("\nВсі біти відновлено успішно!")

    except Exception as e:
        print(f"\nПомилка: {e}")
