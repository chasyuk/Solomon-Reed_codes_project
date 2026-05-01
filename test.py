import importlib
import runpy
import pytest
from primitive_table import build_gf_antilog_table
from galois_field import GaloisField

rs_mod = importlib.import_module("reed_solomon")
ReedSolomon = rs_mod.ReedSolomon

class TestPrimitiveTable:
    @pytest.mark.parametrize("m, expected", [(2, 3), (3, 7), (4, 15), (8, 255)])
    def test_table_size(self, m, expected):
        tbl = build_gf_antilog_table(m)
        assert len(tbl) == expected
        assert int(tbl[0], 2) == 1

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            build_gf_antilog_table(5)

    def test_custom_poly(self):
        tbl = build_gf_antilog_table(5, primitive_poly=0b100101)
        assert len(tbl) == 31

    def test_main_execution(self, capsys):
        runpy.run_path("primitive_table.py", run_name="__main__")
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
        assert (gf(5) + gf(14)) == 5 ^ 14
        assert (gf(11) + gf(11)) == 0
        a = gf(5)
        assert (a + 42) is a  # Invalid type fallback

    def test_mul_and_div(self, gf):
        a, b = gf(2), gf(3)
        res = a * b
        assert isinstance(res, str)
        assert int(a / a, 2) == 1
        assert int(gf(7) / gf(1), 2) == 7

    def test_inverse(self, gf):
        for i in range(1, 16):
            a = gf(i)
            inv = a.inverse()
            assert int(a * inv, 2) == 1
            assert inv.inverse().coeffs == a.coeffs

    def test_repr_and_str(self, gf):
        assert repr(gf(10)) == "10"
        assert "x**" in str(gf(5))

    def test_len_error(self, gf):
        with pytest.raises(TypeError):
            len(gf(5))

    def test_main_execution(self, capsys):
        runpy.run_path("galois_field.py", run_name="__main__")
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

    def test_mul_int(self):
        rs = ReedSolomon()
        rs[0], rs[2] = 5, 3
        res = rs * 4
        assert res[4].coeffs == 5 and res[6].coeffs == 3

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
        assert (ReedSolomon() / ReedSolomon()) in (None, ...)

    def test_code_poly(self):
        rs = ReedSolomon(4, 11)
        g = rs.code_poly
        assert g.degree == 4
        assert rs.code_poly is g  # Caching

    def test_encoding(self):
        rs = ReedSolomon(4, 11)
        msg = [1, 2, 3]
        cw = rs.encoding(msg)
        assert len(cw) >= len(msg) + 4
        for i, val in enumerate(msg):
            assert cw[4 + i] == val

    def test_str(self):
        rs = ReedSolomon()
        assert str(rs) == ""
        rs[3], rs[5] = 5, 3
        assert str(rs).count("x**") == 2

    def test_main_execution(self, capsys):
        runpy.run_path("reed-solomon.py", run_name="__main__")
        assert "16" in capsys.readouterr().out

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
