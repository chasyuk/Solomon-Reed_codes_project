from reed_solomon import ReedSolomon
 DecodeReedSolomon
from galois_field import GaloisField

rs = ReedSolomon(4, 11)
msg = [1, 2, 3]
cw = rs.encoding(msg)
received = ReedSolomon(4, 11)
for i, val in enumerate(cw):
    received[i] = val

# Introduce an error at position 2
received[2] = received[2] + GaloisField(4, rs.gen_poly, 14)

decoder = DecodeReedSolomon(rs)
syndromes = decoder.calc_syndromes(received)
print("Syndromes:", [syndromes[i].coeffs for i in range(2 * rs.t)])

gamma, lamda = euclidean_algorithm(rs.t, syndromes)
print("Euclid gamma:", [gamma[i].coeffs for i in range(gamma.degree + 1)])
print("Euclid lamda:", [lamda[i].coeffs for i in range(lamda.degree + 1)])

positions = decoder.error_location(lamda)
print("Error locations:", positions)

if positions:
    err_vals = decoder.forney_algorithm(positions, gamma, lamda)
    print("Error values:", {pos: val.coeffs for pos, val in err_vals.items()})

corrected = decoder.correct_errors(received, err_vals)
print("Corrected pos 2:", corrected[2].coeffs, "original:", cw[2])
