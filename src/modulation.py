# Módulo C – Modulación/Demodulación
# TODO: implementar BPSK mínima y extender a QPSK/QAM.
import numpy as np
from typing import Tuple

def map_bits(bits: np.ndarray, scheme: str = "BPSK", M: int = 2) -> Tuple[np.ndarray, float, float]:
    raise NotImplementedError("TODO: map_bits (BPSK mínimo)")

def demap_symbols(sym: np.ndarray, scheme: str = "BPSK", M: int = 2) -> np.ndarray:
    raise NotImplementedError("TODO: demap_symbols (BPSK mínimo)")
