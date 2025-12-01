#!/usr/bin/env python3
"""
Ejemplo de uso de las funciones de gráficos del módulo de análisis.
Ejecuta una simulación y genera todos los gráficos disponibles.
"""
import sys
from pathlib import Path

# Agregar src/ al path
project_root = Path(__file__).parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from analysis import (
    run_system_analysis,
    plot_pe_vs_ebn0,
    plot_pb_vs_ebn0,
    plot_psk_vs_fsk,
    plot_coding_comparison,
    generate_all_plots,
)
from report import ReporterTerminal

def main():
    print("=" * 70)
    print("Ejemplo: Generación de gráficos de análisis del sistema")
    print("=" * 70)
    
    # Configuración de la simulación
    out_prefix = "data/output/analysis_example"
    reporter = ReporterTerminal(out_prefix)
    
    print("\n1. Ejecutando simulación...")
    print("   Parámetros:")
    print("   - Eb/N0: 0 a 10 dB (paso 1)")
    print("   - Esquemas: PSK y FSK")
    print("   - M: 2, 4, 8, 16")
    print("   - Bits: 50,000")
    print("   - Con y sin codificación de canal")
    print("\n   Esto puede tomar varios minutos...\n")
    
    # Ejecutar análisis con más parámetros para tener gráficos completos
    df = run_system_analysis(
        ebn0_db_range=range(0, 11),  # 0 a 10 dB
        n_bits=50_000,
        mod_schemes=("psk", "fsk"),
        M_list=(2, 4, 8, 16),
        use_channel_coding=False,  # Primero sin código
        reporter=reporter,
    )
    
    # Ejecutar también con codificación para comparación
    print("\n2. Ejecutando simulación con codificación de canal...\n")
    df_coded = run_system_analysis(
        ebn0_db_range=range(0, 11),
        n_bits=50_000,
        mod_schemes=("psk",),
        M_list=(8,),  # Solo M=8 para la comparación de codificación
        use_channel_coding=True,
        reporter=reporter,
    )
    
    # Combinar DataFrames
    import pandas as pd
    df_combined = pd.concat([df, df_coded], ignore_index=True)
    
    print("\n3. Generando gráficos individuales...\n")
    
    # Gráfico Pe vs Eb/N0 para PSK
    try:
        path = plot_pe_vs_ebn0(df_combined, "psk", "data/output", reporter)
        print(f"   ✅ Pe vs Eb/N0 (PSK): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Gráfico Pb vs Eb/N0 para PSK
    try:
        path = plot_pb_vs_ebn0(df_combined, "psk", "data/output", reporter)
        print(f"   ✅ Pb vs Eb/N0 (PSK): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Gráfico Pe vs Eb/N0 para FSK
    try:
        path = plot_pe_vs_ebn0(df_combined, "fsk", "data/output", reporter)
        print(f"   ✅ Pe vs Eb/N0 (FSK): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Gráfico Pb vs Eb/N0 para FSK
    try:
        path = plot_pb_vs_ebn0(df_combined, "fsk", "data/output", reporter)
        print(f"   ✅ Pb vs Eb/N0 (FSK): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Comparación PSK vs FSK para M=8
    try:
        path = plot_psk_vs_fsk(df_combined, 8, "data/output", reporter)
        print(f"   ✅ PSK vs FSK (M=8): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Comparación con/sin código para PSK-8
    try:
        path = plot_coding_comparison(df_combined, "psk", 8, "data/output", reporter)
        print(f"   ✅ Comparación codificación (PSK-8): {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    print("\n4. Generando todos los gráficos automáticamente...\n")
    try:
        paths = generate_all_plots(df_combined, "data/output", reporter)
        print(f"   ✅ Generados {len(paths)} gráficos")
        for path in paths:
            print(f"      - {path}")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
    
    # Guardar DataFrame combinado
    csv_path = "data/output/analysis_example_results.csv"
    df_combined.to_csv(csv_path, index=False)
    print(f"\n5. Resultados guardados en: {csv_path}")
    
    print("\n" + "=" * 70)
    print("¡Completado! Revisa los gráficos en data/output/")
    print("=" * 70)

if __name__ == "__main__":
    main()
