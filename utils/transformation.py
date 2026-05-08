import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reed_solomon import ReedSolomon

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
