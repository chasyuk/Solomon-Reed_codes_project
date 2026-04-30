from galois_field import *
from primitive_table import build_gf_antilog_table

DEFAULT_POLYS = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        8: 0b100011101
    }

class ReedSolomon:

    __primitive_table = build_gf_antilog_table(4)
    __reversed_primitive_table = {value: key for key, value in __primitive_table.items()}


    def __init__(self, m=4, k=11):
        self.m = m
        self.k = k
        self.gf_size = 2**m
        self.n = self.gf_size - 1
        self.t = (self.gf_size-1 - self.k) // 2

        self.gen_ploy = DEFAULT_POLYS[m]
        self.poly = [
            GaloisField(self.m, self.gen_ploy, 0b0) for _ in range(self.gf_size)
                     ]
        self._code_poly = None

    def __setitem__(self, degree, value):
        if isinstance(value, int):
            value = GaloisField(self.m, self.gen_poly, value)
        self.poly[degree] = value

    def __getitem__(self, degree):
        return self.poly[degree]

    @property
    def degree(self):
        for i in range(len(self.poly) - 1, -1, -1):
            if self.poly[i].coeffs != 0:
                return i
        return 0

    @property
    def code_poly(self):
        if self._code_poly:
            return self._code_poly

        g = ReedSolomon(self.m, self.k)
        g[0] = GaloisField(self.m, self.gen_ploy, 1)

        for i in range(2 * self.t):
            factor = ReedSolomon(self.m, self.k)
            alpha_i = int(self.__primitive_table[i % self.n], 2)

            factor[0] = GaloisField(self.m, self.gen_ploy, alpha_i)
            factor[1] = GaloisField(self.m, self.gen_ploy, 1)

            g = g * factor

        self._code_poly = g
        return g

    def __mul__(self, other):
        if isinstance(other, int):
            new_code = ReedSolomon(self.m + other)
            for idx, poly in enumerate(self.poly):
                new_code[idx] = poly
            return new_code
        ...
        #Треба реалізувати множення на інший поліном



    def __truediv__(self, other):
        ...

    def __add__(self, other):
        new_code = ReedSolomon(self.m, self.k)
        for i in range(self.gf_size):
            new_code[i] = self.poly[i] + other.poly[i]
        return new_code

    def encoding(self, message_list):
        msg_poly = ReedSolomon(self.m, self.k)
        for i, val in enumerate(message_list):
            msg_poly[i] = val
        shifted_msg = msg_poly * (2 * self.t)
        divided_poly = shifted_msg % self.code_poly
        codeword_poly = shifted_msg + divided_poly

        max_deg = codeword_poly.degree
        return [codeword_poly[i].coeffs for i in range(max_deg + 1)]


if __name__ == '__main__':
    g = ReedSolomon(4, 11)
    print(g.code_poly)
