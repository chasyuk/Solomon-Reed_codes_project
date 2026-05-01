from reed_solomon import ReedSolomon

def euclidean_algorithm(t, syndrome):
    a1 = 0
    a2 = 1

    x = ReedSolomon()
    x[0] = 1
    quotinent = x / syndrome
    remainder = x % syndrome
    while quotinent.degree >= t:
        temp = quotinent % remainder
        quotinent = remainder
        remainder = temp

        temp = a1 + quotinent * a2
        a1 = a2
        a2 = temp

    gamma = remainder / 9
    lamda = a2 / 9

    return gamma, lamda
