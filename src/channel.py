# Módulo D – Efectos del canal (AWGN, atenuación)
import numpy as np
from report import Reporter
from pipeline import EncoderDecoder
from utils import BLUE

class Channel(EncoderDecoder):
    def __init__(self, eb_n0_db: float, with_fading: bool = False, rng = None):
        self.eb_n0_db = eb_n0_db
        self.with_fading = with_fading
        self.rng = rng

    def encode(self, sym: np.ndarray, reporter: Reporter) -> np.ndarray:
        reporter.append_line("Canal", BLUE, "Aplicando AWGN/atenuación")

        num_sym = sym.shape[0] if sym.ndim > 1 else 1
        rng = self.rng if self.rng is not None else np.random.default_rng()
        if self.with_fading:
            fades = rng.uniform(0.5, 0.9, size = num_sym)
            # aplicar la atenuación a los símbolos
            fades_reshaped = fades.reshape(-1, *(1,) * (sym.ndim - 1))
            sym = sym * fades_reshaped
            reporter.append_line("Canal", BLUE, f"Atenuación aplicada (min={fades.min()}, max = {fades.max()}) ")

        # se calcula sigma de ruido desde el Eb/N0 asumiendo Eb = 1
        ebn0_lin = 10**(self.eb_n0_db / 10)
        N0 = 1 / ebn0_lin
        sigma = np.sqrt(N0 / 2)
        noise = rng.normal(0, sigma, sym.shape)
        reporter.append_line("Canal", BLUE, f"Eb/N0={self.eb_n0_db} dB -> sigma = {sigma:.4f}")
        return sym + noise 

    def decode(self, sym: np.ndarray, reporter: Reporter) -> np.ndarray:
        return sym
