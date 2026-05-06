from reed_solomon import  ReedSolomon
from decode import DecodeReedSolomon

# Too much errors example

received = ReedSolomon(4, 11)
received[0] = 12
received[1] = 4
received[2] = 5
received[3] = 3
received[4] = 11
received[5] = 0
received[6] = 14
received[7] = 13
received[8] = 2
received[9] = 12
received[10] = 11

print("Initial code:")
print(received)
print()

received.encode()

print("Code with extra bits:")
print(received)
print()


received[3] = 1
received[10] = 9
# received[14] = 13
# received[1] = 14
# received[4] = 3
print("Altered Code:")
print(received)
print()

a = DecodeReedSolomon(received)
fixed_word = a.decode()
print("Fixed Code:")
print(f"{fixed_word}")
print()

fixed_word.get_original()

print("Original Code:")
print(f"{fixed_word}")
