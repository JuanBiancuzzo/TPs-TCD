# Módulo C – Modulación/Demodulación
# TODO: implementar BPSK mínima y extender a QPSK/QAM.
import numpy as np
from typing import Tuple
from report import Reporter

class Modulation:
    def __init__(self, scheme: str = "BPSK", M: int = 2):
        self.scheme = scheme
        self.M = M

    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        raise NotImplementedError("TODO: map_bits (BPSK mínimo)")

    def decode(self, sym: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: demap_symbols (BPSK mínimo)")
