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
        self.M = M  #Número de símbolos
        self.N = 0  
        self.k = int(np.log2(M))    #bits por símbolo
        self.addedBits = 0  #padding bits
        self.batchs = 1000  #tamaño de batch 

        if scheme == Scheme.FSK:
            # Como E_b = 1 entonces E_s = k * E_b = k => sqrt(E_s) = sqrt(k)
            self.symbols = np.sqrt(self.k) * np.eye(M)
            self.N = M 
        
        else:
            self.N = 1 if M == 2 else 2 
            if M == 2: 
                self.symbols = np.array([[1],[-1]]) 
            else:
                phases = 2* np.pi *np.arange(M)/M # ángulos de fase: n*2pi/M, n= 0,1,2..
                self.symbols = np.sqrt(self.k) * np.column_stack((np.cos(phases), np.sin(phases)))

    def _bin_to_gray(n:int) -> int:
        return n^(n>>1)
    def _gray_to_bin(g: int) -> int:
        b = 0
        while g: 
            b ^= g
            g >>= 1
        return b
        
    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        mapping = None
        numSymbols = int(np.ceil(len(bits) / self.k)) #número de símbolos requeridos
        resto = len(bits) % self.k
        self.addedBits = 0 if resto==0 else (self.k - resto)
        if self.scheme == Scheme.FSK:
            reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-FSK con {self.k} bits a símbolos")
        else: 
            reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-PSK con {self.k} bits a símbolos")
        
        
        mapping = np.zeros((numSymbols, self.N))
        # se mapea cada grupo de k bits a un símbolo
        for pos in range(numSymbols):

            # cuántos bits necesito para este símbolo
            is_last = (pos == numSymbols - 1)
            numElements = self.k if (not is_last or resto == 0) else resto

            # grupo de bits -> int
            num = 0 
            for i in range(numElements): 
                num |= (int(bits[pos * self.k + i]) & 1) << i

            # si el grupo tiene menos de k bits, se desplaza el índice a izquierda k-r
            shift = 0
            if is_last and resto!= 0:
                shift = self.k - resto
            if self.scheme == Scheme.FSK:
                sym_idx = (num << shift)
            else:
                #PSK con Gray
                sym_idx = (self._bin_to_gray(num) << shift)

            sym_idx = int(sym_idx) % self.M
            mapping[pos] = self.symbols[sym_idx]

        return mapping

    def decode(self, sym: np.ndarray) -> np.ndarray:
        numSymbols = len(sym)
        numBatchs = int(np.ceil(len(sym)/self.batchs))
        bits = np.zeros(self.k * numSymbols, dtype=int)

        for b in range(numBatchs):
            # se lee de a batches de tamaño fijo
            start = b* self.batchs
            end = min((b+1)*self.batchs, numSymbols)
            batch = sym[start:end]

            if batch.size == 0:
                continue
            # se calculan las distancias euclídeas a todos los posibles símbolos
            # self.symbols: (M,N), batch: (batch_len, N)
            diffs = self.symbols[:,None,:] - batch[None,:, :]   
            norms = np.sum(diffs**2, axis = 2) # (M, batch_len)

            # se busca el más cercano y se lo convierte a bits
            nearest = np.argmin(norms, axis = 0)
            for pos, symbol in enumerate(nearest):
                global_idx = start + pos
                is_last = (global_idx == numSymbols-1)

                # si el último símbolo fue largo menor que k, se desplaza a la inversa
                if self.scheme == Scheme.FSK:
                    bin_idx = int(symbol)
                    if is_last and self.addedBits > 0:
                        bin_idx >>= self.addedBits
                else:
                    #PSK
                    if is_last and self.addedBits > 0:
                        # shift inverso antes de Gray->bin
                        shifted = int(symbol) >> self.addedBits
                        bin_idx = self._gray_to_bin(shifted)
                    else:
                        bin_idx = self._gray_to_bin(int(symbol))
                #bits en LSB-first
                for j in range(self.k):
                    bits[global_idx * self.k + j] = (bin_idx >> j) & 1
        # se slicea los bits de sobra que se pudieran haber agregado en el encoding
        total_bits = self.k * numSymbols
        return bits[:total_bits - self.addedBits]
        
