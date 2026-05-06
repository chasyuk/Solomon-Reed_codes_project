def build_gf_antilog_table(m, primitive_poly=None):
    """
    Build and display the GF(2^m) antilog table in the canonical format:
    | Power | Polynomial Form | Binary | Decimal |
    """

    DEFAULT_POLYS = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        8: 0b100011101
    }

    if primitive_poly is None:
        if m not in DEFAULT_POLYS:
            raise ValueError
        primitive_poly = DEFAULT_POLYS[m]

    field_size   = 1 << m
    cycle_length = field_size
    overflow_bit = field_size

    antilog = [0] * field_size
    element = 1
    for i in range(cycle_length):
        antilog[i] = element
        element <<= 1
        if element & overflow_bit:
            element ^= primitive_poly

    antilog_table = {0: 0}
    for i in range(cycle_length):
        antilog_table[i] = bin(antilog[i])


    return antilog_table

if __name__ == "__main__":
    a = build_gf_antilog_table(4)
    print(a)
    # print(int(a[1] + a[2]), 2)
