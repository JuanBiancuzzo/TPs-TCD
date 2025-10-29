# Módulo E – Codificación de canal (códigos lineales de bloques)
# TODO: implementar paso a paso (generadora G, matriz H, síndromes, encode/decode)

import numpy as np
from typing import Tuple
from report import Reporter
from utils import BLUE

class ChannelCoding:
    def __init__(self, tamanio: int, matriz_generadora: np.ndarray):
        self.n = tamanio
        self.G = matriz_generadora

    def encode(self, bits: np.ndarray, reporter: Reporter) -> np.ndarray:
        reporter.append_line("Codificación", BLUE, "Creando códigos de lineas")

        # Aca deberiamos calcular H y la tabla de sindromes
        # TODO: codificar palabra (n,k)
        return bits

    def decode(self, codigo: np.ndarray) -> np.ndarray:
        # Aca usariamos H y la tabla de sindromes que calculamos antes
        # TODO: decodificar con detección/corrección de errores
        return codigo

    def generar_matriz_H(self) -> np.ndarray:
        raise NotImplementedError("TODO: calcular H a partir de G")

    def tabla_sindromes(H: np.ndarray) -> dict:
        raise NotImplementedError("TODO: construir tabla de síndromes")

    def dist_minima(self) -> Tuple[int, int, int]:
        """Devuelve (dmin, e, t)"""
        raise NotImplementedError("TODO: calcular distancia mínima, errores detectables (e) y corregibles (t)")

