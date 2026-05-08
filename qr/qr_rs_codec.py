"""
Reed–Solomon parity for QR data blocks using only ``core.reed_solomon`` /
``core.galois_field`` (same generator as ``ReedSolomon(8, 255 - ec_count).code_poly``).

Polynomial layout matches ``qrcode.base.Polynomial`` (coefficients low degree → high).
"""

from __future__ import annotations

from core.reed_solomon import ReedSolomon


def _strip_leading_zeros(coeffs: list[int]) -> list[int]:
    i = 0
    while i < len(coeffs) and coeffs[i] == 0:
        i += 1
    return coeffs[i:]


def _gexp(primitive_table, k: int) -> int:
    return int(primitive_table[k % 255], 2)


def _glog(reversed_table, a: int) -> int:
    if a == 0:
        raise ValueError("glog(0)")
    return reversed_table[bin(a)]


def _poly_divmod_qr(dividend: list[int], gen: list[int], primitive_table, reversed_table) -> list[int]:
    """Remainder of ``dividend`` ÷ ``gen`` (qrcode ``Polynomial.__mod__`` semantics)."""
    nums = list(dividend)

    def mod_step(a: list[int]) -> list[int]:
        a = _strip_leading_zeros(a)
        if not a:
            a = [0]
        difference = len(a) - len(gen)
        if difference < 0:
            return a
        if a[0] == 0:
            return mod_step(a[1:])
        ratio = _glog(reversed_table, a[0]) - _glog(reversed_table, gen[0])
        num = [u ^ _gexp(primitive_table, _glog(reversed_table, gv) + ratio) for u, gv in zip(a, gen)]
        if difference:
            num.extend(a[len(gen) :])
        return mod_step(num)

    return mod_step(nums)


def qr_rs_ec_codewords(data_codewords: list[int], ec_count: int) -> list[int]:
    """
    Return ``ec_count`` error-correction bytes for one QR RS block (same as
    ``qrcode.util.create_bytes`` for that block).
    """
    if ec_count <= 0:
        return []
    rs = ReedSolomon(8, 255 - ec_count)
    gen_poly = rs.code_poly
    coeffs = [gen_poly[i].coeffs for i in range(gen_poly.degree + 1)]
    gen = list(reversed(coeffs))

    body = _strip_leading_zeros(list(data_codewords)) + [0] * ec_count
    rem = _poly_divmod_qr(body, gen, rs.primitive_table, rs.reversed_primitive_table)

    mod_offset = len(rem) - ec_count
    out = []
    for i in range(ec_count):
        idx = i + mod_offset
        out.append(rem[idx] if idx >= 0 else 0)
    return out


def create_rs_codeword_sequence(buffer_bytes: list[int], rs_blocks) -> list[int]:
    offset = 0
    dcdata: list[list[int]] = []
    ecdata: list[list[int]] = []

    max_dc = 0
    max_ec = 0

    for rs_block in rs_blocks:
        dc_count = rs_block.data_count
        ec_count = rs_block.total_count - rs_block.data_count
        max_dc = max(max_dc, dc_count)
        max_ec = max(max_ec, ec_count)

        chunk = [0xFF & buffer_bytes[i + offset] for i in range(dc_count)]
        offset += dc_count
        dcdata.append(chunk)
        ecdata.append(qr_rs_ec_codewords(chunk, ec_count))

    out: list[int] = []
    for i in range(max_dc):
        for dc in dcdata:
            if i < len(dc):
                out.append(dc[i])
    for i in range(max_ec):
        for ec in ecdata:
            if i < len(ec):
                out.append(ec[i])
    return out


def create_data_core_rs(version, error_correction, data_list):
    """
    Same as ``qrcode.util.create_data`` but RS parity bytes are produced by
    ``create_rs_codeword_sequence`` (this module, backed by ``core.reed_solomon``).
    """
    from qrcode import base, exceptions, util

    buffer = util.BitBuffer()
    for data in data_list:
        buffer.put(data.mode, 4)
        buffer.put(len(data), util.length_in_bits(data.mode, version))
        data.write(buffer)

    rs_blocks = base.rs_blocks(version, error_correction)
    bit_limit = sum(block.data_count * 8 for block in rs_blocks)
    if len(buffer) > bit_limit:
        raise exceptions.DataOverflowError(
            "Data length overflow. Data size (%s) > size available (%s)"
            % (len(buffer), bit_limit)
        )

    for _ in range(min(bit_limit - len(buffer), 4)):
        buffer.put_bit(False)

    delimit = len(buffer) % 8
    if delimit:
        for _ in range(8 - delimit):
            buffer.put_bit(False)

    bytes_to_fill = (bit_limit - len(buffer)) // 8
    for i in range(bytes_to_fill):
        if i % 2 == 0:
            buffer.put(util.PAD0, 8)
        else:
            buffer.put(util.PAD1, 8)

    return create_rs_codeword_sequence(buffer.buffer, rs_blocks)
