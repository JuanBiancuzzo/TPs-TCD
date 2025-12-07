"""
Funciones auxiliares para la interfaz de línea de comandos.
Módulo que contiene parse_args, dry_run, y run_huffman_only.
"""
import numpy as np
from pathlib import Path

import pipeline
import file            # Módulo A – Read file
import source          # Módulo B – Huffman
import modulation      # Módulo C – Modulación/Demodulación
import channel         # Módulo D – Efectos del canal (AWGN, atenuación)
import cod_channel     # Módulo E – Codificación de canal (códigos lineales de bloques)
import argparse
import report
import analysis

from utils import GREEN, BLUE, YELLOW, RESET

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

    ap.add_argument("--analyze-system", "--module-f", dest="analyze_system", action="store_true",
                    help="ejecuta Módulo F: análisis del sistema (BER/SER vs Eb/N0)")

    ap.add_argument("--without-code", dest="witout_code", action="store_true",
                    help="ejecuta el pipeline completo sin el bloque de codificacion de canal")

    return ap.parse_args()

def dry_run(path_in: str, out_prefix: str):
    """
    Copia el archivo de entrada a la salida sin procesar.
    Útil para validar estructura de I/O.
    """
    pipe = pipeline.Pipeline([
        file.File(out_prefix),
    ], report.ReporterTerminal(out_prefix))

    path_out = pipe.run(path_in)
    
    print(f"{BLUE}[DryRun]{RESET} Se copió el archivo de entrada a la salida (sin procesar).")
    print(f"{GREEN}[Salida]{RESET} {path_out}")

def run_huffman_only(path_in: str, out_prefix: str):
    """
    Ejecuta solo Módulo B:
      - construye diccionarios Huffman
      - codifica -> decodifica
      - guarda recibido
      - genera reporte de tabla de códigos y métricas
    """

    pipe = pipeline.Pipeline([
        file.File(out_prefix = out_prefix),
        source.Source(),
    ], report.ReporterTerminal(out_prefix))

    path_out = pipe.run(path_in)
    print(f"{GREEN}[Salida]{RESET} {path_out}\n")

def run_system_analysis_mode(path_in: str, out_prefix: str, matriz_g: np.ndarray):
    """
    Ejecuta Módulo F: Análisis del sistema.
    
    Realiza análisis de BER/SER vs Eb/N0:
      - Sin codificación de canal (todas las combinaciones)
      - Con codificación de canal (combinación fija: PSK, M=8)
      - Genera gráficos comparativos
    """
    reporter = report.ReporterAnalysis()
    reporter.append_line(analysis.ENCODER, YELLOW, "Iniciando análisis del sistema")
    
    # Configuración de parámetros
    ebn0_range = list(range(0, 11))  # 0 a 10 dB
    mod_schemes = [modulation.Scheme.PSK, modulation.Scheme.FSK]
    M_list = [2, 4, 8, 16]
    
    dfs = []
    for use_channel_coding in [False, True]:
        dfs.append(analysis.run_system_analysis(
            path_in = path_in,
            ebn0_db_range = ebn0_range,
            matriz_g = matriz_g,
            schemes = mod_schemes,
            M_list = M_list,
            use_channel_coding = use_channel_coding,
            reporter = reporter,
        ))

    dfs_without_code, dfs_with_code = dfs[0], dfs[1]
    
    # Guardar resultados en CSV
    for scheme, df in dfs_with_code.items():
        csv_path = Path(out_prefix) / f"analysis_results_with_code_of_{scheme}.csv"
        df.to_csv(csv_path, index = False)
        reporter.append_line(analysis.ENCODER, BLUE, f"Resultados guardados de {scheme} con codificacion en: {csv_path}")

    for scheme, df in dfs_without_code.items():
        csv_path = Path(out_prefix) / f"analysis_results_without_code_of_{scheme}.csv"
        df.to_csv(csv_path, index = False)
        reporter.append_line(analysis.ENCODER, BLUE, f"Resultados guardados de {scheme} sin codificacion en: {csv_path}")
    
    reporter.append_line(analysis.ENCODER, BLUE, "Generando graficos")
    plot_paths = analysis.generate_all_plots(
        dfs_without_code = dfs_without_code,
        dfs_with_code = dfs_with_code,
        output_dir = out_prefix,
        reporter = reporter,
    )

    reporter.append_line(analysis.ENCODER, YELLOW, "Análisis completado exitosamente")
    reporter.append_line(analysis.ENCODER, YELLOW, f"\t- Gráficos generados: {len(plot_paths)}")
    reporter.append_line(analysis.ENCODER, YELLOW, "\n\t\t".join([ f"- {path}" for path in plot_paths ]))

def run_without_code(path_in: str, out_prefix: str, scheme: modulation.Scheme, M: int, eb_no_db: float):
    pipe = pipeline.Pipeline([
        file.File(out_prefix = out_prefix),
        source.Source(),
        modulation.Modulation(scheme = scheme, M = M),
        channel.Channel(eb_n0_db = eb_no_db),
    ], report.ReporterTerminal(out_prefix))

    path_out = pipe.run(path_in)
    print(f"{GREEN}[Salida]{RESET} Texto recibido -> {path_out}\n")

def run_complete_mode(path_in: str, out_prefix: str, matriz_g: np.ndarray, scheme: modulation.Scheme, M: int, eb_no_db: float):
    k, n = matriz_g.shape

    pipe = pipeline.Pipeline([
        file.File(out_prefix = out_prefix),
        source.Source(),
        cod_channel.ChannelCoding(n = n, k = k, matriz_generadora = matriz_g),
        modulation.Modulation(scheme = scheme, M = M, scale_energy = k / n),
        channel.Channel(eb_n0_db = eb_no_db),
    ], report.ReporterTerminal(out_prefix))

    path_out = pipe.run(path_in)
    print(f"{GREEN}[Salida]{RESET} Texto recibido -> {path_out}\n")
