import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.reed_solomon import ReedSolomon
from core.decode import DecodeReedSolomon
import random

orig = ReedSolomon(4, 11)
orig[0] = 12; orig[1] = 4; orig[2] = 5; orig[3] = 3; orig[4] = 11
orig[5] = 0; orig[6] = 14; orig[7] = 13; orig[8] = 2; orig[9] = 12; orig[10] = 11
orig.encode()
orig_c = [x.coeffs for x in orig.poly]

# Test current state with ALL checks
print("=== 2-error test (must be 100%) ===")
ok2 = fail2 = wrong2 = 0
for _ in range(300):
    recv = ReedSolomon(4, 11)
    for i in range(15): recv[i] = orig_c[i]
    indices = random.sample(range(15), 2)
    for i in indices: recv[i] = orig_c[i] ^ random.randint(1,14)
    actual = sum(1 for i in range(15) if recv[i].coeffs != orig_c[i])
    if actual < 2: continue
    d = DecodeReedSolomon(recv)
    try:
        fixed = d.decode()
        fc = [fixed[i].coeffs for i in range(15)]
        if fc == orig_c: ok2 += 1
        else: wrong2 += 1
    except ValueError: fail2 += 1
print(f"  Correct: {ok2}, Wrong: {wrong2}, FalseReject: {fail2}")

print("\n=== 3-error test ===")
ok3 = detected3 = misdecode_valid3 = misdecode_invalid3 = 0
for _ in range(500):
    recv = ReedSolomon(4, 11)
    for i in range(15): recv[i] = orig_c[i]
    indices = random.sample(range(15), 3)
    for i in indices: recv[i] = orig_c[i] ^ random.randint(1,14)
    actual = sum(1 for i in range(15) if recv[i].coeffs != orig_c[i])
    if actual < 3: continue
    d = DecodeReedSolomon(recv)
    try:
        fixed = d.decode()
        fc = [fixed[i].coeffs for i in range(15)]
        # Check if "fixed" has zero syndromes
        synd = d._calc_syndromes(fixed)
        valid = all(synd[i].coeffs == 0 for i in range(4))
        if valid:
            misdecode_valid3 += 1
        else:
            misdecode_invalid3 += 1
    except ValueError:
        detected3 += 1
total3 = detected3 + misdecode_valid3 + misdecode_invalid3
print(f"  Detected: {detected3} ({100*detected3/total3:.1f}%)")
print(f"  Misdecode (valid codeword): {misdecode_valid3} ({100*misdecode_valid3/total3:.1f}%)")
print(f"  Misdecode (INVALID - BUG): {misdecode_invalid3} ({100*misdecode_invalid3/total3:.1f}%)")

print("\n=== 4-error test ===")
detected4 = misdecode_valid4 = misdecode_invalid4 = 0
for _ in range(500):
    recv = ReedSolomon(4, 11)
    for i in range(15): recv[i] = orig_c[i]
    indices = random.sample(range(15), 4)
    for i in indices: recv[i] = orig_c[i] ^ random.randint(1,14)
    actual = sum(1 for i in range(15) if recv[i].coeffs != orig_c[i])
    if actual < 4: continue
    d = DecodeReedSolomon(recv)
    try:
        fixed = d.decode()
        synd = d._calc_syndromes(fixed)
        valid = all(synd[i].coeffs == 0 for i in range(4))
        if valid: misdecode_valid4 += 1
        else: misdecode_invalid4 += 1
    except ValueError:
        detected4 += 1
total4 = detected4 + misdecode_valid4 + misdecode_invalid4
print(f"  Detected: {detected4} ({100*detected4/total4:.1f}%)")
print(f"  Misdecode (valid codeword): {misdecode_valid4} ({100*misdecode_valid4/total4:.1f}%)")
print(f"  Misdecode (INVALID - BUG): {misdecode_invalid4} ({100*misdecode_invalid4/total4:.1f}%)")
