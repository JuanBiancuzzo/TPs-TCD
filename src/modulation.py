# Módulo C – Modulación/Demodulación
# TODO: implementar BPSK mínima y extender a QPSK/QAM.
import numpy as np
from typing import Tuple
from report import Reporter
from utils import BLUE
from enum import Enum

class Scheme(Enum):
    FSK = "M-FSK"
    PSK = "M-PSK"

class Modulation:
    def __init__(self, scheme: Scheme = Scheme.PSK, M: int = 2):
        self.scheme = scheme
        self.M = M
        self.k = int(np.log2(M))
        self.addedBits = 0

        if scheme == Scheme.FSK:
            self.symbols = np.eye(M) # Tendría que ser sqrt(E_s)

        else: 
            raise NotImplementedError(f"TODO: el esquema {self.scheme} no esta implementado")


    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        mapping = None
        cantidadSimbolos = int(np.ceil(len(bits) / self.k))
        self.addedBits = len(bits) % self.k

        if self.scheme == Scheme.FSK:
            reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-FSK con {self.k} bits a símbolos")
            mapping = np.zeros((cantidadSimbolos, self.M))

        for pos in range(cantidadSimbolos):
            num = 0
            numElements = self.k if pos * (self.k + 1) < len(bits) else self.addedBits 
            for i in range(numElements):
                num += bits[pos * self.k + i] << i

            mapping[pos] += self.symbols[num]

        return mapping

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