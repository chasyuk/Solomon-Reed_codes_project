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


    def __init__(self, m=8, k=6):
        self.m = m
        self.k = k
        self.gf_size = 2**m
        self.gen_ploy = DEFAULT_POLYS[m]
        self.poly = [
            GaloisField(self.gf_size, self.gen_ploy, 0b0) for _ in range(m)
                     ]
        self._code_poly = None

    def __setitem__(self, key, value):
        self.poly[self.m - key].coeffs = bin(value)

    @property
    def code_poly(self):
        degrees = [2^i for i in range(self.m)]
        for deg in degrees:

    def __mul__(self, other):
        new_code = ReedSolomon(self.m + (self.m-self.k))
        for idx, poly in enumerate(self.poly):
            new_code[idx] = poly

    def __truediv__(self, other):
        

    def __add__(self, other):
        for i in range(self.m):
            self[i] += other[i]

    def encoding(self):
        pass
