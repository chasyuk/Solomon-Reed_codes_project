from galois_field import GaloisField
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

        self.gen_poly = DEFAULT_POLYS[m]
        self.poly = [
            GaloisField(self.m, self.gen_poly, 0b0) for _ in range(self.gf_size)
                     ]
        self._code_poly = None

    def __setitem__(self, degree, value):
        if isinstance(value, int):
            value = GaloisField(self.m, self.gen_poly, value)

        while degree >= len(self.poly):
            self.poly.append(GaloisField(self.m, self.gen_poly, 0))

        self.poly[degree] = value
            
    def __getitem__(self, degree):
        if degree >= len(self.poly):
            return GaloisField(self.m, self.gen_poly, 0)
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
        g[0] = GaloisField(self.m, self.gen_poly, 1)

        for i in range(2 * self.t):
            factor = ReedSolomon(self.m, self.k)
            alpha_i = int(self.__primitive_table[i % self.n], 2)

            factor[0] = GaloisField(self.m, self.gen_poly, alpha_i)
            factor[1] = GaloisField(self.m, self.gen_poly, 1)

            g = g * factor

        self._code_poly = g
        return g

    def __mul__(self, other):
        new_code = ReedSolomon(self.m, self.k)
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
        remainder = ReedSolomon(self.m, self.k)
        for i in range(self.degree + 1):
            remainder[i] = self.poly[i]
        quotient = ReedSolomon(self.m, self.k)

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
        new_code = ReedSolomon(self.m, self.k)
        max_deg = max(self.degree, other.degree)

        for i in range(max_deg + 1):
            if i < len(self.poly):
                a = self.poly[i]
            else:
                a = GaloisField(self.m, self.gen_poly, 0)
            if i < len(other.poly):
                b = other.poly[i]
            else:
                b = GaloisField(self.m, self.gen_poly, 0)
            new_code[i] = a + b

        return new_code

    def form_derivative(self):
        derivative = ReedSolomon(self.m, self.k)

        for i in range(1, self.degree + 1):
            if i % 2 != 0:
                derivative[i - 1] = self.poly[i]

        return derivative

    def encoding(self, message_list):
        msg_poly = ReedSolomon(self.m, self.k)
        for i, val in enumerate(message_list):
            msg_poly[i] = val
        shifted_msg = msg_poly * (2 * self.t)
        divided_poly = shifted_msg % self.code_poly
        codeword_poly = shifted_msg + divided_poly

        max_deg = codeword_poly.degree
        return [codeword_poly[i].coeffs for i in range(max_deg + 1)]

    def evaluate_poly(self, x_val):
        if isinstance(x_val, int):
            x_val = GaloisField(self.m, self.gen_poly, x_val)

        result = GaloisField(self.m, self.gen_poly, 0)

        for i in range(self.degree, -1, -1):
            result = (result * x_val) + self.poly[i]

        return result

    def __str__(self):
        polynomial = ""

        for idx, field in enumerate(self.poly):
            polynomial += repr(field) + f"x**{self.gf_size - idx - 1}+"  if repr(field) != '0' else ""

        return polynomial[:-1]

class DecodeReedSolomon:
    def __init__(self, message_to_decode):
        self.rs = message_to_decode

    def calc_syndromes(self, received_poly):
        syndromes_poly = ReedSolomon(self.rs.m, self.rs.k)

        for i in range(2 * self.rs.t):
            alpha_bin_str = self.rs._ReedSolomon__primitive_table[i]
            alpha_val = int(alpha_bin_str, 2)
            x_val = GaloisField(self.rs.m, self.rs.gen_poly, alpha_val)

            syndrome_value = received_poly.evaluate_poly(x_val)
            syndromes_poly[i] = syndrome_value

        return syndromes_poly

    def error_location(self, error_locator_poly):
        error_positions = []

        for i in range(self.rs.n):
            alpha_bin_str = self.rs._ReedSolomon__primitive_table[i]
            alpha_val = int(alpha_bin_str, 2)
            alpha_i = GaloisField(self.rs.m, self.rs.gen_poly, alpha_val)
            alpha_inverse = alpha_i.inverse()

            result = error_locator_poly.evaluate_poly(alpha_inverse)

            if result.coeffs == 0:
                error_positions.append(i)

        return error_positions

    def forney_algorithm(self, error_positions, error_evaluator_poly, error_locator_poly):
        error_values = {}
        locator_derivative = error_locator_poly.form_derivative()

        for pos in error_positions:
            alpha_bin_str = self.rs._ReedSolomon__primitive_table[pos]
            x_j = GaloisField(self.rs.m, self.rs.gen_poly, int(alpha_bin_str, 2))
            x_j_inv = x_j.inverse()

            omega_val = error_evaluator_poly.evaluate_poly(x_j_inv)
            lambda_deriv_val = locator_derivative.evaluate_poly(x_j_inv)
            y_j = x_j * (omega_val / lambda_deriv_val)

            error_values[pos] = y_j

        return error_values

    def correct_errors(self, received_poly, error_values):
        corrected_poly = ReedSolomon(self.rs.m, self.rs.k)
        for i in range(received_poly.degree + 1):
            corrected_poly[i] = received_poly[i]

        for pos, error_val in error_values.items():
            corrected_poly[pos] = corrected_poly[pos] + error_val

        return corrected_poly


if __name__ == '__main__':
    g = ReedSolomon(4, 11)
    print(g.code_poly)
