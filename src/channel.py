# Módulo D – Efectos del canal (AWGN, atenuación)
# TODO: implementar AWGN con Eb/N0 y atenuación uniforme [0.5, 0.9].
import numpy as np
from src.report import Reporter
from src.utils import BLUE

class Channel:
    def __init__(self, eb_n0_db: float, with_fading: bool = True, rng=None):
        self.eb_n0_db = eb_n0_db
        self.with_fading = with_fading
        self.rng = rng

    def encode(self, sym: np.ndarray, reporter: Reporter) -> np.ndarray:
        reporter.append_line("Canal", BLUE, "Aplicando AWGN/atenuación")

        # TODO: aplicar_canal (AWGN)
        return sym

    def decode(self, sym: np.ndarray) -> np.ndarray:
        return sym
