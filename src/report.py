from abc import ABC, abstractmethod

import numpy as np
from matplotlib.gridspec import SubplotSpec
from matplotlib.axis import Axis
import matplotlib.pyplot as plt

from collections import Counter
from collections.abc import Callable

from math import log2, ceil
from pathlib import Path
from typing import List, Any, Tuple
from utils import GREEN, BLUE, RED, YELLOW, MAGENTA, RESET

# para los tests usé una versión simplificada de Reporter
##class Reporter:
##    def __init__(self):
##        self.lines = []
##    
##    def append_line(self, category: str, color: str, message: str):
##        """Simple logging method"""
##        print(f"{category}: {message}")

class Reporter(ABC):
    @abstractmethod
    def report_results(self, file_name: str, headers: List[str], data: List[List[Any]]) -> None:
        pass

    @abstractmethod
    def append_metrics(self, from_encoder: str, lines: str) -> None:
        pass

    @abstractmethod
    def append_line(self, from_encoder: str, color: str, line: str) -> None:
        pass

    @abstractmethod
    def graph(self, graph_name: str, axis: np.ndarray, graph: Callable[[np.ndarray], None]) -> None:
        pass

    @abstractmethod
    def show(self) -> None:
        pass

class EmptyReporter(Reporter):
    def report_results(self, file_name: str, headers: List[str], data: List[List[Any]]):
        pass

    def append_metrics(self, from_encoder: str, lines: str) -> None:
        pass

    def append_line(self, from_encoder: str, color: str, line: str):
        pass

    def graph(self, graph_name: str, axis: np.ndarray, graph: Callable[[np.ndarray], None]) -> None:
        pass

    def show(self):
        pass

class ReporterTerminal(Reporter):
    def __init__(self, out_prefix: str, encoding: str = "utf-8"):
        self.out_prefix = out_prefix
        self.encoding = encoding

        self.metric_encoders = {}

    def report_results(self, file_name: str, headers: List[str], data: List[List[Any]]):
        csv_path = Path(f"{self.out_prefix}_{file_name}")
        with csv_path.open("w", encoding=self.encoding) as f:
            f.write(",".join(headers) + "\n")
            for line in data:
                f.write(",".join(line) + "\n")

        self.append_line("Reporter", BLUE, f"CSV → {csv_path}")
        return csv_path

    def append_metrics(self, from_encoder: str, lines: str) -> None:
        if not from_encoder in self.metric_encoders:
            self.metric_encoders[from_encoder] = []
        self.metric_encoders[from_encoder].append(lines)

    def append_line(self, from_encoder: str, color: str, line: str):
        print(f"{color}[{from_encoder}]{RESET} {line}")

    def graph(self, graph_name: str, axis: np.ndarray, graph: Callable[[np.ndarray], None]) -> None:
        graph_path = f"{self.out_prefix}_{graph_name}"
        graph(axis)
        plt.tight_layout()
        plt.savefig(graph_path)
        self.append_line("Reporter", BLUE, f"Gráfico → {graph_path}")
        return graph_path

    def show(self):
        metrics = Path(f"{self.out_prefix}_metricas.md")
        metric_lines = []
        for from_encoder, lines in self.metric_encoders.items():
            metric_lines.append(f"### Resultados {from_encoder}")
            metric_lines += lines
            metric_lines.append("\n")

        metrics.write_text("\n".join(metric_lines), encoding = self.encoding)

def _printable(ch: str) -> str:
    # Representación legible para espacios y controles
    if ch == "\n": return "\\n"
    if ch == "\t": return "\\t"
    if ch == "\r": return "\\r"
    if ch == " ":  return "␠"
    return ch

def _bits_to_str(bits) -> str:
    return "".join("1" if b else "0" for b in bits)

def _encmap_ref(enc_dict) -> dict[str, str]:
    # source.build_huffman_dict → {char: tuple[int]} → "0101"
    return {ch: "".join(str(b) for b in code) for ch, code in enc_dict.items()}

def _encmap_mine(adapter) -> dict[str, str]:
    # adapter expone .h.caracteres: Dict[ch, List[bool]]
    return {ch: _bits_to_str(code) for ch, code in adapter.h.caracteres.items()}

def _entropy_from_probs(probs: dict[str, float]) -> float:
    return -sum(p * log2(p) for p in probs.values() if p > 0)

def _avg_len(probs: dict[str, float], encmap: dict[str, str]) -> float:
    return sum(probs[ch] * len(encmap[ch]) for ch in probs.keys())

def _fixed_length_bits(k: int) -> int:
    return ceil(log2(max(1, k)))

def _first_nonempty_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip() != "":
            return ln
    return text.splitlines()[0] if "\n" in text else text
