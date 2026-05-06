from reed_solomon import ReedSolomon
from galois_field import GaloisField

class DecodeReedSolomon:
    def __init__(self, message_to_decode):
        self.rs = message_to_decode

    def _calc_syndromes(self, received_poly):
        syndromes_poly = ReedSolomon(self.rs.m, self.rs.k)

        for i in range(2 * self.rs.t):
            alpha_bin_str = self.rs._ReedSolomon__primitive_table[i]
            alpha_val = int(alpha_bin_str, 2)
            x_val = GaloisField(self.rs.m, self.rs.gen_poly, alpha_val)

            syndrome_value = received_poly.evaluate_poly(x_val)
            syndromes_poly[i] = syndrome_value

        return syndromes_poly

    def _error_location(self, error_locator_poly):
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

    def _euclidean_algorithm(self, t, syndrome):
        # Euclidean Algorithm implementation
        r_prev = ReedSolomon()
        r_prev[2*t] = 1
        r_curr = syndrome

        a_prev = ReedSolomon()
        a_prev[0] = 0
        a_curr = ReedSolomon()
        a_curr[0] = 1

        while r_curr.degree >= t:
            q = r_prev / r_curr
            r_next = r_prev % r_curr
            
            a_next = a_prev + (q * a_curr)

            r_prev = r_curr
            r_curr = r_next
            a_prev = a_curr
            a_curr = a_next

        gamma_scalar = a_curr[0]
        gamma = r_curr / gamma_scalar
        lamda = a_curr / gamma_scalar

        return gamma, lamda

    def _forney_algorithm(self, error_positions, error_evaluator_poly, error_locator_poly):
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

        if len(error_values) != error_locator_poly.degree:
            return {}

        return error_values

    def _correct_errors(self, received_poly, error_values):
        corrected_poly = ReedSolomon(self.rs.m, self.rs.k)
        for i in range(received_poly.degree + 1):
            corrected_poly[i] = received_poly[i]

        for pos, error_val in error_values.items():
            corrected_poly[pos] = corrected_poly[pos] + error_val

        return corrected_poly

    def decode(self, received_poly):
        syndromes = self._calc_syndromes(received_poly)
        
        # Check if all syndromes are 0 (no errors)
        all_zero = True
        for i in range(syndromes.degree + 1):
            if syndromes[i].coeffs != 0:
                all_zero = False
                break
        if all_zero:
            return received_poly

        gamma, lamda = self._euclidean_algorithm(self.rs.t, syndromes)
        positions = self._error_location(lamda)
        if positions:
            err_vals = self._forney_algorithm(positions, gamma, lamda)
            if not err_vals:
                raise ValueError("Code has too many errors to correct")
            corrected = self._correct_errors(received_poly, err_vals)
            return corrected
        else:
            return received_poly
