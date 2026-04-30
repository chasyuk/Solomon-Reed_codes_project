# CODE_GEN_POLY_255 = x16 + 59x15 + 13x14 + 104x13 + 189x12 + 68x11 + 209x10 + 30x9
# + 8x8 + 163x7 + 65x6 + 41x5 + 229x4 + 98x3 + 50x2 + 36x + 59

from primitive_table import build_gf_antilog_table

class GaloisField:

    __primitive_table = build_gf_antilog_table(4)
    __reversed_primitive_table = {value: key for key, value in __primitive_table.items()}


    def __init__(self, degree, generator_poly, coeffs):
        self.degree = 2 ** degree
        self._generator_poly = generator_poly
        self.coeffs = coeffs

    def __add__(self, other):
        if not isinstance(other, GaloisField):
            return self

        return self.coeffs ^ other.coeffs

    def __mul__(self, other):
        new_degree = self.__reversed_primitive_table[f"{bin(self.coeffs)}"] + self.__reversed_primitive_table[f"{bin(other.coeffs)}"]
        new_degree %= (self.degree - 1)

        return self.__primitive_table[new_degree]

    def inverse(self):
        p = self.__reversed_primitive_table[f"{bin(self.coeffs)}"]
        inv_p = (self.degree - 1 - p) % (self.degree - 1)
        return GaloisField(self.degree, self._generator_poly, int(self.__primitive_table[inv_p], 2))

    def __truediv__(self, other):
        return self * other.inverse()

    def __len__(self):
        return len(self.coeffs)

if __name__ == '__main__':
    a = GaloisField(4, 0b10011, 0b0101)
    b = GaloisField(4, 0b10011, 0b1110)
    print(a + b)
    print(a * b)
    print(a / b)
