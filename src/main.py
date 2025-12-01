"""
TP TA137 – Programa principal (Módulo A)
- Orquesta: Fuente(Huffman) -> [Modulación -> Canal -> Demodulación] -> Fuente(dec)
"""
from pathlib import Path
import numpy as np

# Módulos del proyecto
import pipeline
import file            # Módulo A – Read file
import source          # Módulo B – Huffman
import modulation      # Módulo C – Modulación/Demodulación
import channel         # Módulo D – Efectos del canal (AWGN, atenuación)
import cod_channel     # Módulo E – Codificación de canal (códigos lineales de bloques)
import report
from utils import GREEN, MAGENTA, RESET
from cli import parse_args, handle_special_modes

print(f"\n{MAGENTA}¡Bienvenido al TP TA137!{RESET}\n")

def main():
    args = parse_args()

    if handle_special_modes(args):
        return

    matriz_g = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    ])

    pipe = pipeline.Pipeline([
        file.File(out_prefix=args.out_prefix),
        source.Source(),
        cod_channel.ChannelCoding(n = 15, k = 5, matriz_generadora = matriz_g),
        modulation.Modulation(scheme = modulation.Scheme.PSK, M = 2**3),
        channel.Channel(eb_n0_db = 2),
    ], report.ReporterTerminal(args.out_prefix))

    path_out = pipe.run(args.path_in)
    print(f"{GREEN}[Salida]{RESET} Texto recibido -> {path_out}\n")

if __name__ == "__main__":
    main()
