# src/compare_huffman.py
from collections import Counter
from math import log2
from pathlib import Path
import sys
import importlib

import huffman as hufflib
import source

def entropy_from_probs(probs):
    return -sum(p * log2(p) for p in probs.values() if p > 0)

def encode_with_pypi_huffman(text: str):
    """
    Codifica con la librería 'huffman' de PyPI (byte-level, usa UTF-8).
    Devuelve: (bitstr:str, decoded_text:str)
    """
    data = text.encode("utf-8")
    freq = Counter(data)  # {byte:int}

    # codebook: {b'\x41': '010' } o {b'\x41': b'010'} según versión
    cb = hufflib.codebook([(bytes([b]), c) for b, c in freq.items()])

    # Construir bitstring unificando a str
    parts = []
    for b in data:
        v = cb[bytes([b])]
        bitstr_piece = v.decode("ascii") if isinstance(v, (bytes, bytearray)) else v
        parts.append(bitstr_piece)
    bitstr = "".join(parts)

    # Decodificar por prefijos (simple)
    inv = {}
    for k, v in cb.items():
        ks = k  # byte key
        vs = v.decode("ascii") if isinstance(v, (bytes, bytearray)) else v
        inv[vs] = ks

    out = bytearray()
    buf = ""
    for bit in bitstr:
        buf += bit
        if buf in inv:
            out.extend(inv[buf])
            buf = ""
    decoded = out.decode("utf-8", errors="strict")
    return bitstr, decoded

def run_case(text: str, label: str):
    # --- Tu implementación (char-level) ---
    enc, dec = source.build_huffman_dict(text)
    bits = source.encode_text(text, enc)
    rx = source.decode_bits(bits, dec)
    assert rx == text, f"Round-trip FAILED (source) in {label}"

    probs = source.symbol_probs(text)
    H = source.entropy(probs)
    Lavg = source.avg_code_length(probs, enc)
    total_bits_src = len(bits)

    # baseline fijo 8 bits por byte de UTF-8
    utf8_len = len(text.encode("utf-8"))
    baseline_bits = 8 * utf8_len if utf8_len else 1
    ratio_src = total_bits_src / baseline_bits

    # --- Librería PyPI ---
    bitstr_lib, rx_lib = encode_with_pypi_huffman(text)
    assert rx_lib == text, f"Round-trip FAILED (pypi huffman) in {label}"
    total_bits_lib = len(bitstr_lib)
    ratio_lib = total_bits_lib / baseline_bits

    print(f"\n=== {label} ===")
    print(f"Chars={len(text)} | UTF-8 bytes={utf8_len}")
    print(f"[source.py  (char)]  bits={total_bits_src:>8}  H={H:.4f}  L̄={Lavg:.4f}  ratio={ratio_src:.4f}")
    print(f"[pypi huffman(bytes)] bits={total_bits_lib:>8}                    ratio={ratio_lib:.4f}")
    if any(ord(ch) > 127 for ch in text):
        print("  * Aviso: texto no-ASCII → comparación char-level vs byte-level (no estrictamente equivalente).")


def test_huffman_mine():
    print("Test Huffman Mine")
    text = "ADBADEDBBDD"

    # Construir diccionarios
    enc, dec = source.build_huffman_dict(text)

    # Codificar
    bits = source.encode_text(text, enc)
    encoded_str = "".join(str(b) for b in bits)

    print("Texto original:", text)
    print("Códigos Huffman:", {ch: "".join(map(str, code)) for ch, code in enc.items()})
    print("Encoded:", encoded_str)

    # Decodificar para validar
    decoded = source.decode_bits(bits, dec)
    print("Decoded:", decoded)

def test_huffman_lib():
    print("Test Huffman Lib")
    text = "ADBADEDBBDD"
    data = text.encode("utf-8")

    # Construir codebook con frecuencias de cada byte
    freq = Counter(data)
    codes = hufflib.codebook([(bytes([b]), c) for b, c in freq.items()])

    # Codificar a bitstring (unificando a str)
    parts = []
    for b in data:
        v = codes[bytes([b])]
        bitstr_piece = v.decode("ascii") if isinstance(v, (bytes, bytearray)) else v
        parts.append(bitstr_piece)
    encoded_str = "".join(parts)

    # Decodificar por prefijos
    inv = {}
    for k, v in codes.items():
        vs = v.decode("ascii") if isinstance(v, (bytes, bytearray)) else v
        inv[vs] = k  # byte como valor

    buf, out = "", bytearray()
    for bit in encoded_str:
        buf += bit
        if buf in inv:
            out.extend(inv[buf])
            buf = ""

    decoded = out.decode("utf-8")

    print("Texto original:", text)
    print("Códigos Huffman:", {k.decode(): (v.decode() if isinstance(v,(bytes,bytearray)) else v)
                                 for k, v in codes.items()})
    print("Encoded:", encoded_str)
    print("Decoded:", decoded)

if __name__ == "__main__":
    # Casos rápidos
    # run_case("ADBADEDBBDD", "ASCII corto")
    # run_case("ABRACADABRA ABRACADABRA " * 20, "ASCII repetitivo")
    # run_case("mañana habrá música en el café ☕🎶", "No-ASCII (acentos/emojis)")

    test_huffman_mine()
    test_huffman_lib()
