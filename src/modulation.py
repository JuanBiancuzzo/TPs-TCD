# Módulo C – Modulación/Demodulación
# TODO: implementar BPSK mínima y extender a QPSK/QAM.
import numpy as np
from typing import Tuple
from report import Reporter
from utils import BLUE
from enum import Enum

class Esquema(Enum):
    FSK = "M-FSK"
    PSK = "M-PSK"

class Modulation:
    def __init__(self, scheme: Esquema = Esquema.PSK, M: int = 2):
        self.scheme = scheme
        self.M = M
        self.k = int(np.log2(M))

    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        if self.scheme == Esquema.FSK:
            return self.encodeFSK(bits, reporter)

        raise NotImplementedError(f"TODO: el esquema {self.scheme} no esta implementado")

    def encodeFSK(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray:
        reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-FSK con {self.k} bits a símbolos")

        mapeo = np.zeros((len(bits) // self.k, self.M))

        for pos in range(len(bits) // self.k):
            numeroSimbolo = 0
            for i in range(self.k):
                numeroSimbolo += bits[pos * self.k + i] << i
            mapeo[pos, numeroSimbolo] = 1

        return mapeo

    def decode(self, sym: np.ndarray) -> np.ndarray:
        if self.scheme == Esquema.FSK:
            return self.decodeFSK(sym)

        raise NotImplementedError(f"TODO: el esquema {self.scheme} no esta implementado")

    def decodeFSK(self, sym: np.ndarray) -> np.ndarray:
        simbolos = np.argmax(sym, axis = 1)
        mapeo = np.zeros(self.k * len(simbolos))

        for pos, simbolo in enumerate(simbolos):
            for i in range(self.k):
                mapeo[pos * self.k + i] = 1 & (simbolo >> i)

        return mapeo