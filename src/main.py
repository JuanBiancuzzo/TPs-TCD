"""
TP TA137 – Programa principal (Módulo A)
- Orquesta: Fuente(Huffman) -> [Modulación -> Canal -> Demodulación] -> Fuente(dec)
- Soporta: --dry-run, --huffman-only, y salida de reportes vía report.generate_report_files
"""

import argparse
from pathlib import Path
import numpy as np

# Módulos del proyecto
import pipeline
import file            # Read file (Módulo A)
import source          # Huffman (Módulo B)
import modulation      # TODO (Módulo C)
import channel         # TODO (Módulo D)
import cod_channel     # TODO (Módulo E)
import viz             # TODO (gráficos)
import report
from utils import GREEN, BLUE, RED, YELLOW, MAGENTA, RESET

print(f"\n{MAGENTA}¡Bienvenido al TP TA137!{RESET}\n")

def parse_args():
    """
    Define y parsea CLI flags.
    Defaults:
      --in data/input/texto.txt
      --out-prefix data/output/run1
    """
    ap = argparse.ArgumentParser(description="TP TA137")
    ap.add_argument("--in", dest="path_in", default="data/input/texto.txt",
                    help="ruta al .txt de entrada (default: data/input/texto.txt)")
    ap.add_argument("--out-prefix", default="data/output/run1",
                    help="prefijo de salida (default: data/output/run1)")
    ap.add_argument("--ebn0", type=float, default=6.0, help="Eb/N0 en dB (para canal)")
    ap.add_argument("--dry-run", action="store_true", help="no procesa: copia el texto tal cual")
    ap.add_argument("--huffman-only", action="store_true",
                    help="ejecuta solo Módulo B (cod/dec de fuente)")
    return ap.parse_args()

def dry_run(path_in: str, out_prefix: str):
    """
    Copia el archivo de entrada a la salida sin procesar.
    Útil para validar estructura de I/O.
    """
    pipe = pipeline.Pipeline([
        file.File(out_prefix),
    ])

    path_out = pipe.run(path_in)
    
    print(f"\n{BLUE}[DryRun]{RESET} Se copió el archivo de entrada a la salida (sin procesar).\n")
    print(f"{GREEN}[Salida]{RESET} {path_out}\n")

def run_huffman_only(path_in: str, out_prefix: str):
    """
    Ejecuta solo Módulo B:
      - construye diccionarios Huffman
      - codifica -> decodifica
      - guarda recibido
      - genera reporte de tabla de códigos y métricas
    """

    pipe = pipeline.Pipeline([
        file.File(out_prefix),
        source.Source(),
    ], report.Reporter(out_prefix))

    path_out = pipe.run(path_in)
    print(f"{GREEN}[Salida]{RESET} {path_out}\n")

def main():
    """
    Orquestador del pipeline.
    - dry-run: copia directa
    - huffman-only: solo Módulo B + reportes
    - full pipeline: B -> C -> D -> C^-1 -> B^-1 (TODO módulos C/D/E)
    """
    args = parse_args()

    if args.dry_run:
        dry_run(args.path_in, args.out_prefix)
        print(f"{YELLOW}[TODO]{RESET} Implementar: fuente -> cod_canal.encode -> modulación -> canal -> demodulación -> cod_canal.decode -> fuente\n")
        return

    if args.huffman_only:
        run_huffman_only(args.path_in, args.out_prefix)
        return

    # ---------- Pipeline completo (TODO C/D/E) ----------
    print("[Fuente] Construyendo código Huffman...")
    text = Path(args.path_in).read_text(encoding="utf-8")
    enc, dec = source.build_huffman_dict(text)
    bits_tx  = source.encode_text(text, enc)

    print("[Modulación] Mapeando bits a símbolos (ej. BPSK)...")
    bits_tx_arr = np.array(bits_tx, dtype=np.uint8)
    sym_tx, Es, Eb = modulation.map_bits(bits_tx_arr, scheme="BPSK", M=2)

    print(f"{BLUE}[Canal]{RESET} Aplicando AWGN/atenuación...")
    sym_rx = channel.aplicar_canal(sym_tx, eb_n0_db=args.ebn0, with_fading=True)

    print(f"{BLUE}[Demodulación]{RESET} Símbolos -> bits...")
    bits_rx = modulation.demap_symbols(sym_rx, scheme="BPSK", M=2)

    print(f"{BLUE}[Fuente]{RESET} Decodificando Huffman...")
    texto_rx = source.decode_bits(bits_rx.tolist(), dec)

    out_path = Path(f"{args.out_prefix}_recibido.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(texto_rx, encoding="utf-8")
    print(f"{GREEN}[Salida]{RESET} Texto recibido -> {out_path}\n")

if __name__ == "__main__":
    main()
