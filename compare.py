import time
import random
import matplotlib.pyplot as plt
import numpy as np

# Import implementations
from reed_solomon import ReedSolomon
from decode import DecodeReedSolomon
import reedsolo
import unireedsolomon as urs
import bchlib
import galois

def benchmark_custom(message, m=8, k=223):
    start = time.perf_counter()
    rs = ReedSolomon(m, k)
    for i, val in enumerate(message):
        rs[i] = val
    rs.encode()
    encode_time = time.perf_counter() - start
    
    decoder = DecodeReedSolomon(rs)
    start = time.perf_counter()
    decoded = decoder.decode()
    if decoded is rs: 
        from copy import deepcopy
        decoded = deepcopy(rs)
    decoded.get_original()
    decode_time_no_err = time.perf_counter() - start

    rs[3] = rs[3].coeffs ^ 1
    rs[10] = rs[10].coeffs ^ 1
    rs[14] = rs[14].coeffs ^ 1
    
    decoder_err = DecodeReedSolomon(rs)
    start = time.perf_counter()
    decoded_err = decoder_err.decode()
    decoded_err.get_original()
    decode_time_err = time.perf_counter() - start
    
    return encode_time, decode_time_no_err, decode_time_err

def benchmark_reedsolo(message, k=223, n=255):
    msg_bytes = bytearray(message)
    rs = reedsolo.RSCodec(n - k)
    
    start = time.perf_counter()
    encoded = rs.encode(msg_bytes)
    encode_time = time.perf_counter() - start
    
    start = time.perf_counter()
    rs.decode(encoded)
    decode_time_no_err = time.perf_counter() - start
    
    encoded[3] ^= 1
    encoded[10] ^= 1
    encoded[14] ^= 1
    start = time.perf_counter()
    rs.decode(encoded)
    decode_time_err = time.perf_counter() - start
    
    return encode_time, decode_time_no_err, decode_time_err

def benchmark_urs(message, k=223, n=255):
    msg_str = "".join(chr(c) for c in message)
    coder = urs.rs.RSCoder(n, k)
    
    start = time.perf_counter()
    encoded = coder.encode(msg_str)
    encode_time = time.perf_counter() - start
    
    start = time.perf_counter()
    coder.decode(encoded)
    decode_time_no_err = time.perf_counter() - start
    
    enc_list = list(encoded)
    enc_list[3] = chr(ord(enc_list[3]) ^ 1)
    enc_list[10] = chr(ord(enc_list[10]) ^ 1)
    enc_list[14] = chr(ord(enc_list[14]) ^ 1)
    encoded_err = "".join(enc_list)
    
    start = time.perf_counter()
    coder.decode(encoded_err)
    decode_time_err = time.perf_counter() - start
    
    return encode_time, decode_time_no_err, decode_time_err

def benchmark_bch(message):
    data = bytearray(message)
    # Configure BCH for comparable size: t=16 (error correcting capability), m=13 (Galois Field bits)
    bch = bchlib.BCH(16, m=13)
    
    # Encoding
    start = time.perf_counter()
    ecc = bch.encode(data)
    encode_time = time.perf_counter() - start
    
    # Decoding without errors
    start = time.perf_counter()
    bch.decode(data, ecc)
    decode_time_no_err = time.perf_counter() - start
    
    # Decoding with errors
    corrupted_data = bytearray(data)
    corrupted_data[3] ^= 1
    corrupted_data[10] ^= 1
    corrupted_data[14] ^= 1
    
    start = time.perf_counter()
    ecc_fixed = bytearray(ecc)
    bch.decode(corrupted_data, ecc_fixed)
    bch.correct(corrupted_data, ecc_fixed)
    decode_time_err = time.perf_counter() - start
    
    return encode_time, decode_time_no_err, decode_time_err

def benchmark_galois_bch(message):
    # Convert byte message to bit array for galois
    msg_bits = np.unpackbits(np.array(message, dtype=np.uint8))
    pad = np.zeros(1871 - len(msg_bits), dtype=np.uint8)
    msg_bits_padded = np.concatenate((msg_bits, pad))
    
    msg_gf2 = galois.GF2(msg_bits_padded)
    # Using similar parameters: t=16. n=2047, k=1871 has t=16
    bch = galois.BCH(2047, 1871)
    
    # Encoding
    start = time.perf_counter()
    encoded = bch.encode(msg_gf2)
    encode_time = time.perf_counter() - start
    
    # Decoding without errors
    start = time.perf_counter()
    bch.decode(encoded)
    decode_time_no_err = time.perf_counter() - start
    
    # Decoding with errors
    corrupted = encoded.copy()
    corrupted[24] ^= 1
    corrupted[80] ^= 1
    corrupted[112] ^= 1
    
    start = time.perf_counter()
    bch.decode(corrupted)
    decode_time_err = time.perf_counter() - start
    
    return encode_time, decode_time_no_err, decode_time_err

def run_benchmarks(iterations=5):
    print(f"Running benchmarks ({iterations} iterations)...")
    m = 8
    k = 223
    
    custom_times = [0.0, 0.0, 0.0]
    reedsolo_times = [0.0, 0.0, 0.0]
    urs_times = [0.0, 0.0, 0.0]
    bch_times = [0.0, 0.0, 0.0]
    galois_times = [0.0, 0.0, 0.0]
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        message = [random.randint(0, 255) for _ in range(k)]
        
        c_times = benchmark_custom(message)
        r_times = benchmark_reedsolo(message)
        u_times = benchmark_urs(message)
        b_times = benchmark_bch(message)
        g_times = benchmark_galois_bch(message)
        
        for j in range(3):
            custom_times[j] += c_times[j]
            reedsolo_times[j] += r_times[j]
            urs_times[j] += u_times[j]
            bch_times[j] += b_times[j]
            galois_times[j] += g_times[j]
            
    custom_times = [t / iterations for t in custom_times]
    reedsolo_times = [t / iterations for t in reedsolo_times]
    urs_times = [t / iterations for t in urs_times]
    bch_times = [t / iterations for t in bch_times]
    galois_times = [t / iterations for t in galois_times]
    
    return custom_times, reedsolo_times, urs_times, bch_times, galois_times

def plot_results(custom, reedsolo, urs, bch, galois_bch):
    labels = ['Encode', 'Decode (No Errors)', 'Decode (3 Errors)']
    x = np.arange(len(labels))
    
    # Graph 1: RS Comparison
    width = 0.25
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(x - width, custom, width, label='Custom (Yours)', color='skyblue')
    ax1.bar(x, reedsolo, width, label='reedsolo', color='lightgreen')
    ax1.bar(x + width, urs, width, label='unireedsolomon', color='salmon')

    ax1.set_ylabel('Time (seconds) - Log Scale')
    ax1.set_title('Reed-Solomon Implementations Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    fig1.tight_layout()
    fig1.savefig('benchmark_results.png')
    
    # Graph 2: Custom vs BCH
    width2 = 0.25
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x - width2, custom, width2, label='Custom RS (Yours)', color='skyblue')
    ax2.bar(x, galois_bch, width2, label='galois BCH (Pure Python+NumPy)', color='gold')
    ax2.bar(x + width2, bch, width2, label='bchlib (C ext)', color='plum')
    
    ax2.set_ylabel('Time (seconds) - Log Scale')
    ax2.set_title('Custom RS vs Python BCH vs C BCH Speed Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    fig2.tight_layout()
    fig2.savefig('bch_comparison.png')

    print("\nResults saved to benchmark_results.png and bch_comparison.png")

    print("\nAverage Times (seconds):")
    print(f"{'Implementation':<30} | {'Encode':<15} | {'Decode (Clean)':<15} | {'Decode (Errors)':<15}")
    print("-" * 85)
    print(f"{'Custom (Yours)':<30} | {custom[0]:<15.5f} | {custom[1]:<15.5f} | {custom[2]:<15.5f}")
    print(f"{'reedsolo':<30} | {reedsolo[0]:<15.5f} | {reedsolo[1]:<15.5f} | {reedsolo[2]:<15.5f}")
    print(f"{'unireedsolomon':<30} | {urs[0]:<15.5f} | {urs[1]:<15.5f} | {urs[2]:<15.5f}")
    print(f"{'galois BCH (Pure Python)':<30} | {galois_bch[0]:<15.5f} | {galois_bch[1]:<15.5f} | {galois_bch[2]:<15.5f}")
    print(f"{'bchlib (C ext)':<30} | {bch[0]:<15.5f} | {bch[1]:<15.5f} | {bch[2]:<15.5f}")

if __name__ == "__main__":
    c_times, r_times, u_times, b_times, g_times = run_benchmarks(5)
    plot_results(c_times, r_times, u_times, b_times, g_times)
