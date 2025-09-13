# Módulo E – Codificación de canal (códigos lineales de bloques)
# TODO: implementar paso a paso (generadora G, matriz H, síndromes, encode/decode)

import numpy as np
from typing import Tuple

def generar_matriz_H(G: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO: calcular H a partir de G")

def tabla_sindromes(H: np.ndarray) -> dict:
    raise NotImplementedError("TODO: construir tabla de síndromes")

def codificar(bits: np.ndarray, G: np.ndarray) -> np.ndarray:
    raise NotImplementedError("TODO: codificar palabra (n,k)")

def decodificar(codigo: np.ndarray, H: np.ndarray, sindromes: dict) -> np.ndarray:
    raise NotImplementedError("TODO: decodificar con detección/corrección de errores")

def dist_minima(G: np.ndarray) -> Tuple[int, int, int]:
    """Devuelve (dmin, e, t)"""
    raise NotImplementedError("TODO: calcular distancia mínima, errores detectables (e) y corregibles (t)")
