import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.galois_field import GaloisField
from core.primitive_table import build_gf_antilog_table

DEFAULT_POLYS = {
        2: 0b111,
        3: 0b1011,
        4: 0b10011,
        8: 0b100011101
    }

class ReedSolomon:


    def __init__(self, m=8, k=223, primitive_table=None, reversed_primitive_table=None):
        self.__primitive_table = build_gf_antilog_table(m) if primitive_table is None else primitive_table
        self.__reversed_primitive_table = {value: key for key, value in self.__primitive_table.items()} if reversed_primitive_table is None else reversed_primitive_table
        self.m = m
        self.k = k
        self.gf_size = 2**m
        self.n = self.gf_size - 1
        self.n_parity = self.n - self.k
        self.t = self.n_parity // 2

        self.gen_poly = DEFAULT_POLYS[m]
        self.poly = [
            GaloisField(self.m, self.gen_poly, 0b0, self.__primitive_table, self.__reversed_primitive_table) for _ in range(self.gf_size)
                     ]
        self._code_poly = None

    @property
    def primitive_table(self):
        return self.__primitive_table

    @property
    def reversed_primitive_table(self):
        return self.__reversed_primitive_table

    def __setitem__(self, degree, value):
        if isinstance(value, int):
            value = GaloisField(self.m, self.gen_poly, value, self.__primitive_table, self.__reversed_primitive_table)

        while degree >= len(self.poly):
            self.poly.append(GaloisField(self.m, self.gen_poly, 0, self.__primitive_table, self.__reversed_primitive_table))

        self.poly[degree] = value

    def __getitem__(self, degree):
        if degree >= len(self.poly):
            return GaloisField(self.m, self.gen_poly, 0, self.__primitive_table, self.__reversed_primitive_table)
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

        generator_poly = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
        generator_poly[0] = GaloisField(self.m, self.gen_poly, 1, self.__primitive_table, self.__reversed_primitive_table)

        for i in range(self.n_parity):
            factor = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
            alpha_i = int(self.__primitive_table[i % self.n], 2)

            factor[0] = GaloisField(self.m, self.gen_poly, alpha_i, self.__primitive_table, self.__reversed_primitive_table)
            factor[1] = GaloisField(self.m, self.gen_poly, 1, self.__primitive_table, self.__reversed_primitive_table)

            generator_poly = generator_poly * factor

        self._code_poly = generator_poly
        return generator_poly

    def __mul__(self, other):
        new_code = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
        if isinstance(other, int):
            for i in range(self.degree + 1):
                if self.poly[i].coeffs != 0:
                    new_code[i + other] = self.poly[i]
            return new_code

        if isinstance(other, GaloisField):
            for i in range(self.degree + 1):
                new_code[i] = self.poly[i] * other
            return new_code

        if isinstance(other, ReedSolomon):
            for i in range(self.degree + 1):
                if self.poly[i].coeffs == 0:
                    continue
                for j in range(other.degree + 1):
                    if other.poly[j].coeffs == 0:
                        continue
                    new_code[i + j] = new_code[i + j] + (self.poly[i] * other.poly[j])
            return new_code


    def _divmod(self, divisor):
        remainder = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
        for i in range(self.degree + 1):
            remainder[i] = self.poly[i]
        quotient = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)

        divisor_deg = divisor.degree
        lead_divisor = divisor[divisor_deg]
        if lead_divisor.coeffs == 0:
            raise ZeroDivisionError()

        while remainder.degree >= divisor_deg:
            rem_deg = remainder.degree
            lead_rem = remainder[rem_deg]
            if lead_rem.coeffs == 0:
                break

            ratio = lead_rem / lead_divisor
            shift = rem_deg - divisor_deg
            quotient[shift] = quotient[shift] + ratio

            for i in range(divisor_deg + 1):
                if divisor.poly[i].coeffs != 0:
                    prod = divisor.poly[i] * ratio
                    remainder[shift + i] = remainder[shift + i] + prod

        return quotient, remainder

    def __mod__(self, other):
        _, remainder = self._divmod(other)
        return remainder

    def __truediv__(self, other):
        if isinstance(other, GaloisField):
            return self * other.inverse()
        quotient, _ = self._divmod(other)
        return quotient

    def __add__(self, other):
        new_code = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
        max_deg = max(self.degree, other.degree)

        for i in range(max_deg + 1):
            if i < len(self.poly):
                a = self.poly[i]
            else:
                a = GaloisField(self.m, self.gen_poly, 0, self.__primitive_table, self.__reversed_primitive_table)
            if i < len(other.poly):
                b = other.poly[i]
            else:
                b = GaloisField(self.m, self.gen_poly, 0, self.__primitive_table, self.__reversed_primitive_table)
            new_code[i] = a + b

        return new_code

    def form_derivative(self):
        derivative = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)

        for i in range(1, self.degree + 1):
            if i % 2 != 0:
                derivative[i - 1] = self.poly[i]

        return derivative

    def encode(self):
        message_list = self.poly
        msg_poly = ReedSolomon(self.m, self.k, self.__primitive_table, self.__reversed_primitive_table)
        for i, val in enumerate(message_list):
            msg_poly[i] = val
        shifted_msg = msg_poly * self.n_parity
        divided_poly = shifted_msg % self.code_poly
        codeword_poly = shifted_msg + divided_poly

        max_deg = codeword_poly.degree
        # return divided_poly
        self.poly = [GaloisField(self.m, self.gen_poly, int(codeword_poly[i].coeffs), self.__primitive_table, self.__reversed_primitive_table) for i in range(max_deg + 1)]

    def get_original(self):
        self.poly = self.poly[self.n_parity:]


    def evaluate_poly(self, x_val):
        if isinstance(x_val, int):
            x_val = GaloisField(self.m, self.gen_poly, x_val, self.__primitive_table, self.__reversed_primitive_table)

        result = GaloisField(self.m, self.gen_poly, 0, self.__primitive_table, self.__reversed_primitive_table)

        for i in range(self.degree, -1, -1):
            result = (result * x_val) + self.poly[i]

        return result

    def __str__(self):
        terms = []
        poly_size = len(self.poly)
        for idx, field in enumerate(reversed(self.poly)):
            coeff = repr(field)
            if coeff == '0':
                continue
            power = poly_size - idx - 1
            if power > 0:
                terms.append(f"{coeff}x^{power}")
            else:
                terms.append(coeff)

        return " + ".join(terms)

if __name__ == '__main__':
    g = ReedSolomon(4, 11)
    print(g.encode())
    print(g.code_poly)
