"""
TP TA137 - Programa principal (Módulo A)
- Orquesta: Fuente(Huffman) -> [Modulación -> Canal -> Demodulación] -> Fuente(dec)
"""
from pathlib import Path
import numpy as np

# Módulos del proyecto
import modulation
from utils import MAGENTA, RESET
from cli import parse_args, dry_run, run_huffman_only, run_system_analysis_mode, run_complete_mode

def main() -> None:
    print(f"\n{MAGENTA}¡Bienvenido al TP TA137!{RESET}\n")

    args = parse_args()
    Path(args.out_prefix).mkdir(parents = True, exist_ok = True)

    if args.dry_run:
        dry_run(args.path_in, args.out_prefix)
        return

    if args.huffman_only:
        run_huffman_only(args.path_in, args.out_prefix)
        return
    
    matriz_g = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    ])

    if args.analyze_system:
        run_system_analysis_mode(args.path_in, args.out_prefix, matriz_g)
        return

    scheme = modulation.Scheme.PSK
    M = 2**3
    eb_no_db = 6

    run_complete_mode(args.path_in, args.out_prefix, matriz_g, scheme, M, eb_no_db)

if __name__ == "__main__":
    main()
