from reed_solomon import ReedSolomon

def euclidean_algorithm(t, syndrome):
    r0 = ReedSolomon(m=8, k=syndrome.k)
    r0[2*t] = 1

    r1 = syndrome

    a0 = ReedSolomon(m=8, k=syndrome.k)
    a1 = ReedSolomon(m=8, k=syndrome.k)
    a1[0] = 1

    while r1.degree >= t:
        q = r0 / r1
        r_next = r0 % r1
        a_next = a0 + (q * a1)
        r0, r1 = r1, r_next
        a0, a1 = a1, a_next

    return r1, a1
