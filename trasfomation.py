from reed_solomon import ReedSolomon

def transfrom_to_reed_solomon_code(msg, m=8, k=223):
    if not isinstance(msg, str):
        raise ValueError("You can only transform string!")
    code = ReedSolomon(m, k)
    msg_ascii  = [ord(char) for char in msg]
    size = len(msg)
    for i, char in zip(range(code.gf_size), reversed(msg_ascii)):
        code[i] = char
    return code

def transfrom_to_string(code, m=8, k=223):
    if not isinstance(code, ReedSolomon):
        raise ValueError("You can only transform back Reed-Solomon code!")

    msg = ''

    for char in reversed(code.poly):
        msg += chr(char.coeffs)

    return msg



if __name__ == '__main__':
    m = "hello"
    rd_code = transfrom_to_reed_solomon_code(m, 4, 11)
    print(rd_code)
    message = transfrom_to_string(rd_code)
    print(message)
