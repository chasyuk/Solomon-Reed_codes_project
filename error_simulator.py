import random

def introduce_random_errors(rs_poly, num_errors):
    """
    Introduces exactly `num_errors` random symbol errors into the Reed-Solomon polynomial.
    This simulates random bit-flips across the transmission.
    """
    length = len(rs_poly.poly)
    if num_errors > length:
        raise ValueError("Cannot introduce more errors than the length of the codeword.")
    
    # Pick random unique positions
    error_positions = random.sample(range(length), num_errors)
    
    for pos in error_positions:
        # A symbol in GF(2^8) is between 0 and 255.
        # We need to change the current value to a DIFFERENT random value.
        current_val = rs_poly[pos].coeffs
        new_val = current_val
        while new_val == current_val:
            new_val = random.randint(0, 255)
        
        rs_poly[pos] = new_val
        
    return error_positions

def introduce_burst_error(rs_poly, start_index, burst_length):
    """
    Introduces consecutive errors starting at `start_index` for `burst_length` symbols.
    This simulates a scratch on a CD or a physical smudge on a QR code.
    """
    length = len(rs_poly.poly)
    corrupted_positions = []
    
    for i in range(start_index, min(start_index + burst_length, length)):
        # Flip all bits in the symbol to ensure it is thoroughly corrupted
        current_val = rs_poly[i].coeffs
        rs_poly[i] = current_val ^ 0xFF 
        corrupted_positions.append(i)
        
    return corrupted_positions

def introduce_custom_errors(rs_poly, error_dict):
    """
    Introduces specific bit-flips at specific indices using XOR logic.
    `error_dict` format: {index: XOR_mask, ...}
    
    Example: {3: 0b00000001, 10: 0xFF}
    This will flip the last bit of the symbol at index 3, and flip ALL bits at index 10.
    """
    for pos, xor_mask in error_dict.items():
        if 0 <= pos < len(rs_poly.poly):
            current_val = rs_poly[pos].coeffs
            rs_poly[pos] = current_val ^ xor_mask
            
    return list(error_dict.keys())

# --- Quick Test ---
if __name__ == "__main__":
    from reed_solomon import ReedSolomon
    from decode import DecodeReedSolomon
    
    print("--- Testing Error Generators ---")
    rs = ReedSolomon(8, 223)
    # Fill with dummy data
    for i in range(223): rs[i] = 65 
    rs.encode()
    
    # 1. Random Errors
    print("\n1. Introducing 5 Random Errors...")
    positions = introduce_random_errors(rs, 5)
    print(f"Corrupted positions: {positions}")
    
    # 2. Burst Error
    print("\n2. Introducing a Burst Error of length 4...")
    positions = introduce_burst_error(rs, start_index=150, burst_length=4)
    print(f"Corrupted positions: {positions}")
    
    # 3. Custom Errors (Bit manipulation)
    print("\n3. Introducing Custom Bit-Flips...")
    # Flip the 1st bit at index 10, flip all bits at index 11
    positions = introduce_custom_errors(rs, {10: 0b00000001, 11: 0b11111111})
    print(f"Corrupted positions: {positions}")
    
    # Try decoding
    decoder = DecodeReedSolomon(rs)
    try:
        fixed = decoder.decode()
        print("\nSUCCESS! The decoder successfully found and fixed all the errors!")
    except Exception as e:
        print(f"\nFAILED to decode: {e}")
