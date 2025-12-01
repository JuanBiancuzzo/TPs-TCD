# Módulo A – Read file
from pathlib import Path

from pipeline import EncoderDecoder
from report import Reporter
from utils import BLUE

class File(EncoderDecoder):
    def __init__(self, out_prefix: str, encoding: str = "utf-8"):
        self.out_prefix = out_prefix
        self.encoding = encoding

    def encode(self, path_in: str, reporter: Reporter) -> str:
        """
        Leer el archivo de entrada, con el encoding especificado
        """
        if reporter is not None:
            reporter.append_line("File", BLUE, f"Leyendo archivo: {path_in}")
        return Path(path_in).read_text(encoding=self.encoding)

    def decode(self, text: str, reporter: Reporter) -> str:
        out_path = Path(f"{self.out_prefix}_recibido.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding=self.encoding)
        return out_path