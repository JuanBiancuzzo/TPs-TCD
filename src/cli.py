"""
Funciones auxiliares para la interfaz de línea de comandos.
Módulo que contiene parse_args, dry_run, y run_huffman_only.
"""

import argparse
import pipeline
import file
import source
import report
import analysis
import pandas as pd
from pathlib import Path
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
        file.File(out_prefix),
        source.Source(),
    ], report.ReporterTerminal(out_prefix))

    path_out = pipe.run(path_in)
    print(f"{GREEN}[Salida]{RESET} {path_out}\n")


def run_system_analysis_mode(out_prefix: str):
    """
    Ejecuta Módulo F: Análisis del sistema.
    
    Realiza análisis de BER/SER vs Eb/N0:
      - Sin codificación de canal (todas las combinaciones)
      - Con codificación de canal (combinación fija: PSK, M=8)
      - Genera gráficos comparativos
    """
    print(f"{BLUE}[Módulo F]{RESET} Iniciando análisis del sistema...\n")
    
    # Configurar reporter
    reporter = report.ReporterTerminal(out_prefix)
    # Usar data/analysis para todos los archivos de análisis
    analysis_dir = Path("data/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuración de parámetros
    ebn0_range = range(0, 11)  # 0 a 10 dB
    n_bits = 100_000
    mod_schemes = ("psk", "fsk")
    M_list = (2, 4, 8, 16)
    
    print(f"{BLUE}[Módulo F]{RESET} Parámetros:")
    print(f"  - Eb/N0: {ebn0_range.start} a {ebn0_range.stop-1} dB")
    print(f"  - Esquemas: {', '.join(mod_schemes).upper()}")
    print(f"  - M: {M_list}")
    print(f"  - Bits por simulación: {n_bits:,}")
    print()
    
    # 1. Análisis sin codificación de canal
    print(f"{BLUE}[Módulo F]{RESET} Ejecutando análisis sin codificación de canal...")
    print("  (Esto puede tomar varios minutos)\n")
    
    df_no_coding = analysis.run_system_analysis(
        ebn0_db_range=ebn0_range,
        n_bits=n_bits,
        mod_schemes=mod_schemes,
        M_list=M_list,
        use_channel_coding=False,
        reporter=reporter,
    )
    
    # 2. Análisis con codificación de canal (combinación fija: PSK, M=8)
    print(f"\n{BLUE}[Módulo F]{RESET} Ejecutando análisis con codificación de canal...")
    print("  Combinación: PSK, M=8")
    print("  (Esto puede tomar varios minutos)\n")
    
    df_with_coding = analysis.run_system_analysis(
        ebn0_db_range=ebn0_range,
        n_bits=n_bits,
        mod_schemes=("psk",),
        M_list=(8,),
        use_channel_coding=True,
        reporter=reporter,
    )
    
    # Combinar DataFrames
    df_combined = pd.concat([df_no_coding, df_with_coding], ignore_index=True)
    
    # Guardar resultados en CSV
    csv_path = analysis_dir / "analysis_results.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"{GREEN}[Módulo F]{RESET} Resultados guardados en: {csv_path}\n")
    
    # 3. Generar gráficos
    print(f"{BLUE}[Módulo F]{RESET} Generando gráficos...\n")
    plot_paths = []
    
    try:
        # Pe vs Eb/N0 para PSK
        path = analysis.plot_pe_vs_ebn0(df_combined, "psk", str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando Pe PSK: {e}")
    
    try:
        # Pb vs Eb/N0 para PSK
        path = analysis.plot_pb_vs_ebn0(df_combined, "psk", str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando Pb PSK: {e}")
    
    try:
        # Pe vs Eb/N0 para FSK
        path = analysis.plot_pe_vs_ebn0(df_combined, "fsk", str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando Pe FSK: {e}")
    
    try:
        # Pb vs Eb/N0 para FSK
        path = analysis.plot_pb_vs_ebn0(df_combined, "fsk", str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando Pb FSK: {e}")
    
    try:
        # Comparación PSK vs FSK para M=8
        path = analysis.plot_psk_vs_fsk(df_combined, 8, str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando PSK vs FSK: {e}")
    
    try:
        # Comparación con/sin código para PSK-8
        path = analysis.plot_coding_comparison(df_combined, "psk", 8, str(analysis_dir), reporter)
        plot_paths.append(path)
        print(f"  ✅ {Path(path).name}")
    except Exception as e:
        print(f"  ⚠️  Error generando comparación codificación: {e}")
    
    # Resumen final
    print(f"\n{GREEN}[Módulo F]{RESET} Análisis completado exitosamente!")
    print(f"\n{BLUE}Resumen:{RESET}")
    print(f"  - Resultados CSV: {csv_path}")
    print(f"  - Gráficos generados: {len(plot_paths)}")
    for path in plot_paths:
        print(f"    • {Path(path).name}")
    print()


def handle_special_modes(args):
    """
    Maneja los modos especiales de ejecución (dry-run, huffman-only, analyze-system).
    Retorna True si se ejecutó alguno de estos modos, False en caso contrario.
    """
    if args.dry_run:
        dry_run(args.path_in, args.out_prefix)
        print(f"{YELLOW}[TODO]{RESET} Implementar: fuente -> cod_canal.encode -> modulación -> canal -> demodulación -> cod_canal.decode -> fuente\n")
        return True

    if args.huffman_only:
        run_huffman_only(args.path_in, args.out_prefix)
        return True
    
    if args.analyze_system:
        run_system_analysis_mode(args.out_prefix)
        return True

    return False

