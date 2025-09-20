from collections import Counter
from math import log2, ceil
from pathlib import Path
from typing import List, Any
from utils import GREEN, BLUE, RED, YELLOW, MAGENTA, RESET

class Reporter:
    def __init__(self, out_prefix: str, encoding: str = "utf-8"):
        self.out_prefix = out_prefix
        self.encoding = encoding

        self.lines = []
        self.metrics = Path(f"{out_prefix}_metricas.md")
        self.metrics.write_text("")

    def report_results(self, file_name: str, headers: List[str], data: List[List[Any]]):
        csv_path = Path(f"{self.out_prefix}_{file_name}")
        with csv_path.open("w", encoding=self.encoding) as f:
            f.write(",".join(headers) + "\n")
            for line in data:
                f.write(",".join(line) + "\n")

        self.append_line("Reporter", BLUE, f"CSV → {csv_path}")
        return csv_path

    def append_metrics(self, lines: str):
        self.metrics.write_text(
            self.metrics.read_text(encoding=self.encoding) + lines, encoding=self.encoding
        )

    def append_line(self, from_encoder: str, color: str, line: str):
        self.lines.append(f"{color}[{from_encoder}]{RESET} {line}")

    def show(self):
        for line in self.lines:
            print(line)

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
