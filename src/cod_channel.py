# Módulo E – Codificación de canal (códigos lineales de bloques)
# TODO: implementar paso a paso (generadora G, matriz H, síndromes, encode/decode)

import numpy as np
from typing import Tuple
from report import Reporter
from pipeline import EncoderDecoder
from utils import BLUE
from itertools import combinations

ENCODER = "Codificación"
SAMPLE_SIZE = 3
DESFASE = 10

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
        reporter.append_line(ENCODER, BLUE, "Creando códigos de lineas")

        resto = len(bits) % self.k
        self.added_bits = 0 if resto == 0 else (self.k - resto)

        # Ajustamos los bits para que ya tengan la cantidad justa
        bits = np.concatenate((bits, np.zeros(self.added_bits)))
        num_blocks = len(bits) // self.k # número de blockes de k bits
        new_bits = np.zeros(self.n * num_blocks)

        for i, message in enumerate(np.split(bits, num_blocks)):
            # U = mensaje x G
            new_bits[i * self.n:(i + 1) * self.n] += (message @ self.G) % 2

        table = [
            "|              e                |         e H^T       |",
            "|-------------------------------|---------------------|",
        ]
        skip = True
        for i, (num, e) in enumerate(self.table.items()):
            if (i > 20) and (i + 5) < len(self.table):
                if skip:
                    skip = False
                    table.append("...")
            else:
                eHT = np.array([(num >> j) & 1 for j in range(self.n - self.k)], dtype=int)
                table.append(f"|{print_array(e.astype(int))}|{print_array(eHT)}|")

        dmin, e, t = self.dist_minima()
        reporter.append_metrics(ENCODER, "\n".join([
            f"- Valor de dmin: {dmin}",
            f"- Valor de e: {e}",
            f"- Valor de t: {t}",
            "\n- Matriz G: ",
            *( print_array(line.astype(int)) for line in self.G ),
            "\n- Matriz H: ",
            *( print_array(line.astype(int)) for line in self.H ),
            "\n#### Tabla de simbolos",
            *table,
            "\n#### Muestra (codificación → canal → decodificación)",
            f"- Secuencia inicial: {bits[DESFASE * self.k:(DESFASE + SAMPLE_SIZE) * self.k].astype(int)}",
            f"- m * G: {new_bits[DESFASE * self.n:(DESFASE + SAMPLE_SIZE) * self.n].astype(int)}",
        ]))

        return new_bits

    def decode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray:
        num_blocks = len(bits) // self.n # número de blockes de n bits

        new_bits = np.zeros(self.k * num_blocks)
        syndroms = []
        errors = []
        messages = []
        for i, message in enumerate(np.split(bits, num_blocks)):
            # sindrome = mensaje x H^T
            syndrom = (message @ self.H.T) % 2
            if len(syndroms) < SAMPLE_SIZE + DESFASE:
                syndroms.append(syndrom)

            num = int(syndrom @ self.base) # Lo transforma en número
            e = self.table[num]

            # Sumamos el error para sacarlo
            message = message ^ e

            if len(errors) < SAMPLE_SIZE + DESFASE:
                errors.append(e)

            if len(messages) < SAMPLE_SIZE + DESFASE:
                messages.append(message)

            # Nos quedamos con los primeros k bits, que serían el mensaje original
            new_bits[i * self.k:(i + 1) * self.k] += message[:self.k]

        reporter.append_metrics(ENCODER, "\n".join([
            f"- Recibido: {bits[DESFASE * self.n:(DESFASE + SAMPLE_SIZE) * self.n].astype(int)}",
            f"- e * H^T: \n\t{'\n\t'.join(syndrom.__str__() for syndrom in syndroms[-SAMPLE_SIZE:])}",
            f"- e: \n\t{'\n\t'.join(error.__str__() for error in errors[-SAMPLE_SIZE:])}",
            f"- Decodificador: {'\n\t'.join(message.__str__() for message in messages[-SAMPLE_SIZE:])}",
        ]))

        range_bits = slice(self.k * num_blocks - self.added_bits)
        return new_bits[range_bits]

    def generar_matriz_H(self) -> np.ndarray:
        # G = [ I_k : P^{(n - k) x k}]
        P = self.G[:, self.k:]

        # H = [ P^T: I_(n - k) ]
        return np.concatenate((P.T, np.eye(self.n - self.k)), axis = 1).astype(int)

    def tabla_sindromes(self, H: np.ndarray) -> dict:
        # Nos guardamos e*H^T, y nos devuelve el error que se agregó
        table_syndrome = { 0: np.zeros(self.n, dtype = int) }
        max_syndromes = 2**(self.n - self.k) # Actualmente 1024

        for e in NumberGenerator(self.n):
            # Necesitamos generar todos los números de un bit, después de dos, etc.
            syndrom = (e @ H.T) % 2 # Tiene tamaño (10,)
            num = int(syndrom @ self.base) # Lo transforma en número

            if num in table_syndrome or num == 0:
                continue

            table_syndrome[num] = e.astype(int)
            if len(table_syndrome) >= max_syndromes:
                break

        return table_syndrome

    def dist_minima(self) -> Tuple[int, int, int]:
        k, n = self.G.shape

        dmin = n + 1  # inicial grande
        # recorrer todos los mensajes no nulos u (0..2^k-1), construir codeword u·G (mod 2)
        for num in range(1, 2**k):
            # vector u en LSB-first consistente con el resto del código
            u = np.array([(num >> i) & 1 for i in range(k)], dtype=int)
            codeword = (u @ self.G) % 2
            weight = int(np.count_nonzero(codeword))
            if 0 < weight < dmin:
                dmin = weight

        if dmin == n + 1:
            dmin = 0

        e = dmin - 1
        t = int(np.floor((dmin - 1) / 2)) if dmin > 0 else 0
        return dmin, e, t

def print_array(array: np.ndarray) -> str:
    return f"{array.__str__().replace('[', ' ').replace(']', ' ')}" 

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
