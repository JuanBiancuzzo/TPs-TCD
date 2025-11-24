# Módulo E – Codificación de canal (códigos lineales de bloques)
# TODO: implementar paso a paso (generadora G, matriz H, síndromes, encode/decode)

import numpy as np
from typing import Tuple
from report import Reporter
from pipeline import EncoderDecoder
from utils import BLUE
from itertools import combinations

class ChannelCoding(EncoderDecoder):
    def __init__(self, n: int, k: int, matriz_generadora: np.ndarray):
        self.n = n
        self.k = k
        syndrom_bits = self.n - self.k
        self.base = np.flip(np.logspace(0, syndrom_bits - 1, num = syndrom_bits, base = 2, endpoint = True, dtype = int))

        self.G = matriz_generadora
        self.H = self.generar_matriz_H()
        self.table = self.tabla_sindromes(self.H)

    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray:
        reporter.append_line("Codificación", BLUE, "Creando códigos de lineas")

        resto = len(bits) % self.k
        self.added_bits = 0 if resto == 0 else (self.k - resto)

        # Ajustamos los bits para que ya tengan la cantidad justa
        bits = np.concat((bits, np.zeros(self.added_bits)))
        num_blocks = len(bits) // self.k # número de blockes de k bits
        new_bits = np.zeros(self.n * num_blocks)

        for i, message in enumerate(np.split(bits, num_blocks)):
            # U = mensaje x G
            new_bits[i * self.n:(i + 1) * self.n] += (message @ self.G) % 2

        return new_bits

    def decode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray:
        num_blocks = len(bits) // self.n # número de blockes de n bits

        new_bits = np.zeros(self.k * num_blocks)
        for i, message in enumerate(np.split(bits, num_blocks)):
            # sindrome = mensaje x H^T
            syndrom = (message @ self.H.T) % 2

            num = int(syndrom @ self.base) # Lo transforma en número
            if num is self.tabla_sindromes:
                e = self.tabla_sindromes[num]

                # Sumamos el error para sacarlo
                message = message ^ e

            # Nos quedamos con los primeros k bits, que serían el mensaje original
            new_bits[i * self.k:(i + 1) * self.k] += message[:self.k]

        range_bits = slice(self.k * num_blocks - self.added_bits)
        return new_bits[range_bits]

    def generar_matriz_H(self) -> np.ndarray:
        # G = [ I_k : P^{(n - k) x k}]
        P = self.G[:, self.k:]

        # H = [ I_(n - k) : P^T]
        return np.concat(( np.eye(self.n - self.k), P.T ), axis = 1)

    def tabla_sindromes(self, H: np.ndarray) -> dict:
        # Nos guardamos e*H^T, y nos devuelve el error que se agregó
        table_syndrome = {}
        syndrom_bits = self.n - self.k
        max_syndromes = 2**syndrom_bits # Actualmente 1024

        for e in NumberGenerator(self.n):
            # Necesitamos generar todos los números de un bit, después de dos, etc.
            syndrom = (e @ H.T) % 2 # Tiene tamaño (10,)
            num = int(syndrom @ self.base) # Lo transforma en número

            if num in table_syndrome:
                continue

            table_syndrome[num] = e
            if len(table_syndrome) >= max_syndromes:
                break

        return table_syndrome

    def dist_minima(self) -> Tuple[int, int, int]:
        """Devuelve (dmin, e, t)"""
        dmin = self.k
        for num in range(1, 2**self.k):
            # bin lo transforma en string: 14 => 0b1110, y después contamos los 1's
            dmin = min(dmin, bin(num).count("1"))

        return dmin, dmin - 1, int(np.floor((dmin - 1) / 2))

class NumberGenerator:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        self.count = 1
        self.combinations = combinations(range(self.n), self.count)
        return self

    def __next__(self):

        try:
            exp = list(self.combinations.__next__())

        except StopIteration:
            self.count += 1
            if self.count > self.n:
                raise StopIteration

            self.combinations = combinations(range(self.n), self.count)
            exp = list(self.combinations.__next__())

        e = np.zeros(self.n)
        e[np.array(exp, dtype = int)] += 1
        return np.flip(e)
