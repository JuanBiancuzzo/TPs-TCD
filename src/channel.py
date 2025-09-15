# Módulo D – Efectos del canal (AWGN, atenuación)
# TODO: implementar AWGN con Eb/N0 y atenuación uniforme [0.5, 0.9].
import numpy as np
from report import Reporter

class Channel:
    def __init__(self, eb_n0_db: float, with_fading: bool = True, rng=None):
        self.eb_n0_db = eb_n0_db
        self.with_fading = with_fading
        self.rng = rng

    def encode(self, sym: np.ndarray, reporter: Reporter) -> np.ndarray:
        raise NotImplementedError("TODO: aplicar_canal (AWGN + atenuación)")

    def decode(self, sym: np.ndarray) -> np.ndarray:
        return sym
