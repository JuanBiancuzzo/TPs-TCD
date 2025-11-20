# Módulo C – Modulación/Demodulación
# TODO: implementar BPSK mínima y extender a QPSK/QAM.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Tuple
from report import Reporter
from utils import BLUE
from enum import Enum

class Scheme(Enum):
    FSK = "M-FSK"
    PSK = "M-PSK"

    def __str__(self):
        if self == Scheme.FSK:
            return "FSK"
        else: 
            return "PSK"

# Vamos a tomar E_b = 1
class Modulation:
    def __init__(self, scheme: Scheme = Scheme.PSK, M: int = 2):
        self.scheme = scheme
        self.M = M               # Número de símbolos
        self.N = 0               # Dimensiones
        self.k = int(np.log2(M)) # bits por símbolo
        self.added_bits = 0      # padding bits
        self.batchs = 1000       # tamaño de batch 

        if scheme == Scheme.FSK:
            # Como E_b = 1 entonces E_s = k * E_b = k => sqrt(E_s) = sqrt(k)
            self.symbols = np.sqrt(self.k) * np.eye(M)
            self.N = M 
        
        else:
            self.N = 1 if M == 2 else 2 
            if M == 2: 
                symbols = np.sqrt(self.k) * np.array([[1],[-1]]) 
            else:
                
                phases = 2 * np.pi * np.arange(M) / M  # ángulos de fase: n*2pi/M, n=0..M-1
                symbols = np.sqrt(self.k) * np.column_stack((np.cos(phases), np.sin(phases)))

            self.symbols = np.zeros((self.M, self.N))
            for i in range(M):
                self.symbols[i] += symbols[Modulation._bin_to_gray(i)]
    
    @staticmethod
    def _bin_to_gray(n: int) -> int:
        return n ^ (n >> 1)
        
    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray: 
        # Calcular la energia media de simbolo y de bit
        mapping = None
        num_symbols = int(np.ceil(len(bits) / self.k)) #número de símbolos requeridos
        resto = len(bits) % self.k
        self.added_bits = 0 if resto == 0 else (self.k - resto)

        reporter.append_line("Modulación", BLUE, f"Mapeando de {self.M}-{self.scheme} con {self.k} bits a símbolos")
        self.graph_constalation(reporter)
        
        mapping = np.zeros((num_symbols, self.N))
        # se mapea cada grupo de k bits a un símbolo
        for pos in range(num_symbols):

            # cuántos bits necesito para este símbolo
            is_last = (pos == num_symbols - 1)
            num_elements = self.k if (not is_last or resto == 0) else resto

            # grupo de bits -> int (LSB first)
            num = 0 
            for i in range(num_elements): 
                # LSB-first: bit at position i goes to position i
                num |= (int(bits[pos * self.k + i]) & 1) << i

            # si el grupo tiene menos de k bits, se desplaza el índice a izquierda k-r
            shift = 0
            if is_last and resto != 0:
                shift = self.k - resto
            sym_idx = int(num << shift) % self.M

            mapping[pos] = self.symbols[sym_idx]

        return mapping

    def decode(self, sym: np.ndarray) -> np.ndarray:
        num_symbols = len(sym)
        num_batchs = int(np.ceil(len(sym)/self.batchs))
        bits = np.zeros(self.k * num_symbols, dtype=int)

        for b in range(num_batchs):
            # se lee de a batches de tamaño fijo
            start = b* self.batchs
            end = min((b+1)*self.batchs, num_symbols)
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
                is_last = (global_idx == num_symbols - 1)

                # si el último símbolo fue largo menor que k, se desplaza a la inversa
                bin_idx = int(symbol)
                if is_last and self.added_bits > 0:
                    bin_idx >>= self.added_bits
                        
                #bits en LSB-first
                for j in range(self.k):
                    bits[global_idx * self.k + j] = (bin_idx >> j) & 1

        # se slicea los bits de sobra que se pudieran haber agregado en el encoding
        total_bits = self.k * num_symbols
        return bits[:total_bits - self.added_bits]
        
    def estimated_simbol_energy(self, sys: np.ndarray) -> np.ndarray:
        # Por la relación de Parseval se puede calcular la energía de simbolo por 
        # la norma al cuadrado de cada vector
        return np.linalg.norm(sys, axis = 1)**2

    def estimated_bit_energy(self, sys: np.ndarray) -> np.ndarray:
        # Usando que e_b = e_s / log_2(M) podemos reutilizar lo que ya calculamos
        return self.estimated_simbol_energy(sys) / self.k

    def graph_constalation(self, reporter: Reporter) -> None:
        if self.scheme == Scheme.FSK and self.N > 2:
            reporter.append_line("Modulación", BLUE, f"No se puede graficar la constelación para la {self.M}-{self.scheme}")
            return

        def graph(ax):
            xs = self.symbols[:, 0]
            ys = np.zeros(self.symbols[:, 0].shape)
            if self.scheme != Scheme.PSK or self.N != 1:
                ys = self.symbols[:, 1]
            symbol_color = "blue"
            ax.scatter(xs, ys, marker = "x", color = symbol_color)

            for i, (x, y) in enumerate(zip(xs * 1.1, ys * 1.1)):
                ax.text(x, y, f"{i:0{self.k}b}", ha = "center", va = "center")

            extend = 0.4
            lim = (np.min(xs) - extend, np.max(xs) + extend)

            boundary_color = "black"
            if self.scheme == Scheme.FSK:
                ax.plot([ *lim ], [ *lim ], ls = "--", color = boundary_color)

            else:
                mag = np.max(np.linalg.norm(self.symbols, axis = 1))
                phases = 2 * np.pi * np.arange(100) / 100
                ax.plot(mag * np.cos(phases), mag * np.sin(phases), ls = "--")

                mag *= 2
                for i in range(self.M):
                    phase = 2 * np.pi * (i + 0.5) / self.M
                    ax.plot([ 0, mag * np.cos(phase) ] , [ 0, mag * np.sin(phase) ], ls = "--", color = boundary_color)

            ax.set_xlim(lim)
            ax.set_ylim(lim)

            ax.grid()
            ax.legend(handles = [ 
                Line2D([0], [0], color = symbol_color, marker = "x", label = "Símbolos"),
                Line2D([0], [0], color = boundary_color, lw = 2, ls = "--", label = "Frontera de decisión"),
            ], loc = "upper right")

        fig = plt.figure(figsize = (10, 10))
        ax = fig.add_subplot(111)
        reporter.graph(
            graph_name = f"Contelacion_{self.M}-{self.scheme}",
            axis = ax,
            graph = graph,
        )

        reporter.append_line("Modulación", BLUE, f"Generando constelación {self.M}-{self.scheme}")