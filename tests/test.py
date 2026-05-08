import importlib
import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from core.primitive_table import build_gf_antilog_table
from core.galois_field import GaloisField

rs_mod = importlib.import_module("core.reed_solomon")
ReedSolomon = rs_mod.ReedSolomon
decode_mod = importlib.import_module("core.decode")
DecodeReedSolomon = decode_mod.DecodeReedSolomon

class TestPrimitiveTable:
    @pytest.mark.parametrize("m, expected", [(2, 4), (3, 8), (4, 16), (8, 256)])
    def test_table_size(self, m, expected):
        tbl = build_gf_antilog_table(m)
        assert len(tbl) == expected
        assert int(tbl[0], 2) == 1

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            build_gf_antilog_table(5)

    def test_custom_poly(self):
        tbl = build_gf_antilog_table(5, primitive_poly=0b100101)
        assert len(tbl) == 32

    def test_main_execution(self, capsys):
        runpy.run_path(str(ROOT_DIR / "core" / "primitive_table.py"), run_name="__main__")
        assert "{" in capsys.readouterr().out

class TestGaloisField:
    @pytest.fixture
    def gf(self):
        return lambda c: GaloisField(4, 0b10011, c)

    def test_init(self, gf):
        a = gf(5)
        assert a._m == 4
        assert a.degree == 16
        assert a.coeffs == 5

    def test_add(self, gf):
        assert (gf(5) + gf(14)).coeffs == 5 ^ 14
        assert (gf(11) + gf(11)).coeffs == 0
        a = gf(5)
        assert (a + 42) is a  # Invalid type fallback

    def test_mul_and_div(self, gf):
        a, b = gf(2), gf(3)
        res = a * b
        assert isinstance(res, GaloisField)
        assert (a / a).coeffs == 1
        assert (gf(7) / gf(1)).coeffs == 7

    def test_mul_by_zero(self, gf):
        assert (gf(0) * gf(5)).coeffs == 0
        assert (gf(5) * gf(0)).coeffs == 0

    def test_div_by_zero(self, gf):
        with pytest.raises(ZeroDivisionError, match="Ділення на нуль в полі Галуа неможливе!"):
            gf(5) / gf(0)

    def test_zero_div(self, gf):
        assert (gf(0) / gf(5)).coeffs == 0

    def test_inverse(self, gf):
        for i in range(1, 16):
            a = gf(i)
            inv = a.inverse()
            assert (a * inv).coeffs == 1
            assert inv.inverse().coeffs == a.coeffs

    def test_repr_and_str(self, gf):
        assert repr(gf(10)) == "10"
        assert "x**" in str(gf(5))

    def test_len_error(self, gf):
        with pytest.raises(TypeError):
            len(gf(5))

    def test_main_execution(self, capsys):
        runpy.run_path(str(ROOT_DIR / "core" / "galois_field.py"), run_name="__main__")
        out = capsys.readouterr().out
        assert len(out) > 0

class TestReedSolomon:
    def test_init_and_props(self):
        rs = ReedSolomon(4, 11)
        assert rs.m == 4 and rs.k == 11 and rs.t == 2
        assert len(rs.poly) == 16
        assert rs.degree == 0

    def test_setitem_getitem(self):
        rs = ReedSolomon()
        rs[3] = 5
        assert rs[3].coeffs == 5
        rs[5] = GaloisField(4, 0b10011, 10)
        assert rs[5].coeffs == 10

    def test_degree(self):
        rs = ReedSolomon()
        rs[10] = 5
        assert rs.degree == 10
        rs[10] = 0
        assert rs.degree == 0

    def test_add(self):
        a, b = ReedSolomon(), ReedSolomon()
        a[0], b[0] = 5, 3
        assert (a + b)[0].coeffs == 5 ^ 3
        assert all(c.coeffs == 0 for c in (a + a).poly)

    def test_add_length_mismatch(self):
        a = ReedSolomon()
        a[20] = 1
        b = ReedSolomon()
        b[2] = 1
        res = a + b
        assert res[20].coeffs == 1
        assert res[2].coeffs == 1
        res2 = b + a
        assert res2[20].coeffs == 1
        assert res2[2].coeffs == 1

    def test_mul_int(self):
        rs = ReedSolomon()
        rs[0], rs[2] = 5, 3
        res = rs * 4
        assert res[4].coeffs == 5 and res[6].coeffs == 3

    def test_mul_galois_field(self):
        rs = ReedSolomon()
        rs[0], rs[1] = 5, 3
        gf = GaloisField(4, 0b10011, 2)
        res = rs * gf
        assert res[0].coeffs == (GaloisField(4, 0b10011, 5) * gf).coeffs
        assert res[1].coeffs == (GaloisField(4, 0b10011, 3) * gf).coeffs

    def test_mul_poly(self):
        a, b = ReedSolomon(), ReedSolomon()
        a[0], a[1] = 1, 1
        b[0], b[1] = 2, 1
        res = a * b
        assert res.degree == 2

        # Test boundary resize
        c, d = ReedSolomon(), ReedSolomon()
        c[15], d[15] = 1, 1
        assert (c * d).degree == 30

    def test_mod(self):
        dividend = ReedSolomon()
        dividend[5], dividend[3] = 1, 1

        divisor = ReedSolomon()
        divisor[4], divisor[0] = 1, 1

        res = dividend % divisor
        assert res.degree < 4

        # Exact division
        a, b = ReedSolomon(), ReedSolomon()
        a[1], b[0] = 1, 1
        assert all(c.coeffs == 0 for c in ((a * b) % b).poly)

    def test_truediv(self):
        with pytest.raises(ZeroDivisionError):
            ReedSolomon() / ReedSolomon()

    def test_truediv_galois_field(self):
        rs = ReedSolomon()
        rs[0], rs[1] = 5, 3
        gf = GaloisField(8, 0b100011101, 2)
        res = rs / gf
        assert res[0].coeffs == (GaloisField(8, 0b100011101, 5) / gf).coeffs
        assert res[1].coeffs == (GaloisField(8, 0b100011101, 3) / gf).coeffs

    def test_truediv_poly(self):
        dividend = ReedSolomon()
        dividend[5], dividend[3] = 1, 1
        divisor = ReedSolomon()
        divisor[4], divisor[0] = 1, 1
        res = dividend / divisor
        assert res.degree <= 1

    def test_code_poly(self):
        rs = ReedSolomon(4, 11)
        g = rs.code_poly
        assert g.degree == 4
        assert rs.code_poly is g  # Caching

    def test_form_derivative(self):
        rs = ReedSolomon()
        rs[0] = 5
        rs[1] = 3
        rs[2] = 2
        rs[3] = 4
        deriv = rs.form_derivative()
        assert deriv[0].coeffs == 3
        assert deriv[1].coeffs == 0
        assert deriv[2].coeffs == 4

    def test_evaluate_poly(self):
        rs = ReedSolomon()
        rs[0] = 2
        rs[1] = 3
        res = rs.evaluate_poly(2)
        expected = (GaloisField(4, 0b10011, 3) * GaloisField(4, 0b10011, 2)) + GaloisField(4, 0b10011, 2)
        assert res.coeffs == expected.coeffs

        gf_val = GaloisField(4, 0b10011, 5)
        res2 = rs.evaluate_poly(gf_val)
        expected2 = (GaloisField(4, 0b10011, 3) * gf_val) + GaloisField(4, 0b10011, 2)
        assert res2.coeffs == expected2.coeffs

    def test_encoding(self):
        rs = ReedSolomon(4, 11)
        # Set message in polynomial
        msg = [1, 2, 3]
        for i, val in enumerate(msg):
            rs[i] = val
        rs.encode()
        # After encoding, check that data is preserved with parity bits at the start
        # Expected format: [parity0, parity1, parity2, parity3, msg0, msg1, msg2]
        assert rs.degree + 1 >= len(msg) + 2 * rs.t
        for i, val in enumerate(msg):
            # Message should be at position 2*t + i
            assert rs[2 * rs.t + i].coeffs == val

    def test_str(self):
        rs = ReedSolomon()
        assert str(rs) == ""
        rs[3], rs[5] = 5, 3
        assert str(rs).count("x^") == 2

    def test_main_execution(self, capsys):
        runpy.run_path(str(ROOT_DIR / "core" / "reed_solomon.py"), run_name="__main__")
        assert "x^" in capsys.readouterr().out

class TestDecodeReedSolomon:
    def test_decode_flow(self):
        rs = ReedSolomon(4, 11)
        decoder = DecodeReedSolomon(rs)

        # 1. calculate syndromes
        received = ReedSolomon(4, 11)
        received[0] = 5
        received[1] = 2
        syndromes = decoder._calc_syndromes(received)
        assert syndromes.degree >= 0

        # 2. error_location
        err_loc = ReedSolomon(4, 11)
        err_loc[0] = 1
        alpha_1 = int(rs.primitive_table[1], 2)
        err_loc[1] = alpha_1
        positions = decoder._error_location(err_loc)
        assert positions == [1]

        # 3. forney_algorithm
        err_eval = ReedSolomon(4, 11)
        err_eval[0] = 5
        err_values = decoder._forney_algorithm([1], err_eval, err_loc)
        assert 1 in err_values
        assert isinstance(err_values[1], GaloisField)

        # 4. correct_errors
        received[1] = 0
        corrected = decoder._correct_errors(received, err_values)
        assert corrected[1].coeffs == (received[1] + err_values[1]).coeffs

class TestExactNumericValues:
    def test_exact_polynomial_multiplication(self):
        p1 = ReedSolomon(4, 11)
        p1[0] = 5
        p1[1] = 3
        p2 = ReedSolomon(4, 11)
        p2[0] = 2
        p2[1] = 4
        p3 = p1 * p2
        assert [p3[i].coeffs for i in range(p3.degree + 1)] == [10, 1, 12]

    def test_exact_encoding(self):
        rs = ReedSolomon(4, 11)
        msg = [1, 2, 3]
        for i, val in enumerate(msg):
            rs[i] = val
        rs.encode()
        # Check that the encoded codeword has parity bits first, then message
        cw = [rs[i].coeffs for i in range(rs.degree + 1)]
        assert cw == [5, 4, 11, 10, 1, 2, 3]

    def test_exact_euclid_and_decoding(self):
        rs = ReedSolomon(4, 11)
        cw = [5, 4, 11, 10, 1, 2, 3]
        received = ReedSolomon(4, 11)
        for i, val in enumerate(cw):
            received[i] = val

        # Introduce exact error at pos 2 with magnitude 14
        received[2] = received[2] + GaloisField(4, rs.gen_poly, 14)

        decoder = DecodeReedSolomon(rs)
        syndromes = decoder._calc_syndromes(received)
        assert [syndromes[i].coeffs for i in range(2 * rs.t)] == [14, 13, 1, 4]

        gamma, lamda = decoder._euclidean_algorithm(rs.t, syndromes)
        assert [gamma[i].coeffs for i in range(gamma.degree + 1)] == [14]
        assert [lamda[i].coeffs for i in range(lamda.degree + 1)] == [1, 4]

        positions = decoder._error_location(lamda)
        assert positions == [2]

        err_vals = decoder._forney_algorithm(positions, gamma, lamda)
        assert err_vals[2].coeffs == 14

        corrected = decoder._correct_errors(received, err_vals)
        assert corrected[2].coeffs == 11
        for i in range(len(cw)):
            assert corrected[i].coeffs == cw[i]

    def test_decode_public_method(self):
        rs = ReedSolomon(4, 11)
        cw = [5, 4, 11, 10, 1, 2, 3]
        received = ReedSolomon(4, 11)
        for i, val in enumerate(cw):
            received[i] = val

        # Error at pos 2
        received[2] = received[2] + GaloisField(4, rs.gen_poly, 14)
        decoder = DecodeReedSolomon(received)
        corrected = decoder.decode()
        for i in range(len(cw)):
            assert corrected[i].coeffs == cw[i]

if __name__ == "__main__":  # pragma: no cover
    # Execute pytest
    pytest.main([__file__, "-v"])
