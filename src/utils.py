from typing import Dict
from math import log2, ceil

# Colores para la terminal
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"

def efficiency(entropy_bits: float, avg_len: float) -> float:
    if avg_len <= 0:
        return 0.0
    return entropy_bits / avg_len

def fixed_length_bits(alphabet_size: int) -> int:
    # ej: ASCII ~ log2(256)=8
    return ceil(log2(max(1, alphabet_size)))
