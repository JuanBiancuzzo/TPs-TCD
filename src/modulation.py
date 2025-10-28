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

# Vamos a tomar E_b = 1
class Modulation:
    def __init__(self, scheme: Scheme = Scheme.PSK, M: int = 2):
        self.scheme = scheme
        self.M = M
        self.N = 0
        self.k = int(np.log2(M))
        self.addedBits = 0

        self.batchs = 1000
        if scheme == Scheme.FSK:
            # Como E_b = 1 entonces E_s = k * E_b = k => sqrt(E_s) = sqrt(k)
            self.symbols = np.sqrt(self.k) * np.eye(M)
            self.N = M

        else: 
            raise NotImplementedError(f"TODO: el esquema {self.scheme} no esta implementado")

    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        mapping = None
        numSymbols = int(np.ceil(len(bits) / self.k))
        self.addedBits = len(bits) % self.k

        if self.scheme == Scheme.FSK:
            reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-FSK con {self.k} bits a símbolos")

        mapping = np.zeros((numSymbols, self.N))

        for pos in range(numSymbols):
            num = 0
            numElements = self.k if (pos + 1) * self.k < len(bits) else self.addedBits 
            for i in range(numElements):
                num += bits[pos * self.k + i] << i
            mapping[pos] += self.symbols[num]

        return mapping

    def decode(self, sym: np.ndarray) -> np.ndarray:
        numBatchs = int(np.ceil(len(sym) / self.batchs))
        bits = np.zeros(self.k * len(sym), dtype = int)

        for i in range(numBatchs):
            endBatchElement = numBatchs * (i + 1)
            if i >= numBatchs - 1: # el ultimo batch
                endBatchElement = len(sym)
            batch = sym[numBatchs * i : endBatchElement]

            norms = np.sum((self.symbols[:, :, np.newaxis] - batch[:, :, np.newaxis].T) ** 2, axis = 1)

            for pos, simbolo in enumerate(np.argmin(norms, axis = 0)):
                for j in range(self.k):
                    # Transformamos el numero del simbolo a la representacion del bit
                    bits[i * self.batchs + pos * self.k + j] = 1 & (simbolo >> i)
        
        return bits[:len(bits) - self.addedBits]
