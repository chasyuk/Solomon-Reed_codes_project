import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.primitive_table import build_gf_antilog_table

class GaloisField:

    def __init__(self, degree, generator_poly, coeffs, primitive_table=None, reversed_primitive_table=None):
        self.__primitive_table = build_gf_antilog_table(degree) if primitive_table is None else primitive_table
        self.__reversed_primitive_table = {value: key for key, value in self.__primitive_table.items()} if reversed_primitive_table is None else reversed_primitive_table
        self._m = degree
        self.degree = 2 ** degree
        self._generator_poly = generator_poly
        self.coeffs = coeffs

    def __add__(self, other):
        if not isinstance(other, GaloisField):
            return self

        res_coeffs = self.coeffs ^ other.coeffs
        return GaloisField(self._m, self._generator_poly, res_coeffs, self.__primitive_table, self.__reversed_primitive_table)

    def __mul__(self, other):
        if self.coeffs == 0 or other.coeffs == 0:
            return GaloisField(self._m, self._generator_poly, 0)
        new_degree = self.__reversed_primitive_table[bin(self.coeffs)] + self.__reversed_primitive_table[bin(other.coeffs)]
        new_degree %= (self.degree - 1)

        return GaloisField(self._m, self._generator_poly, int(self.__primitive_table[new_degree], 2), self.__primitive_table, self.__reversed_primitive_table)

    def inverse(self):
        p = self.__reversed_primitive_table[f"{bin(self.coeffs)}"]
        inv_p = (self.degree - 1 - p) % (self.degree - 1)
        return GaloisField(self._m, self._generator_poly, int(self.__primitive_table[inv_p], 2), self.__primitive_table, self.__reversed_primitive_table)

    def __truediv__(self, other):
        if other.coeffs == 0:
            raise ZeroDivisionError("Ділення на нуль в полі Галуа неможливе!")
        if self.coeffs == 0:
            return GaloisField(self._m, self._generator_poly, 0, self.__primitive_table, self.__reversed_primitive_table)
        return self * other.inverse()

    def __len__(self):
        return len(self.coeffs)

    def __str__(self):
        polynomial = ""

        for idx, _ in enumerate(f"{self.coeffs:b}"):
            polynomial += f"x**{self.degree - idx - 1} +"


        return polynomial[:-2]


    def __repr__(self):
        return str(self.coeffs)

if __name__ == '__main__':
    a = GaloisField(4, 0b10011, 0b0101)
    b = GaloisField(4, 0b10011, 0b1110)
    print(a + b)
    print(a * b)
    print(a / b)
