"""
Script de prueba para run_system_analysis
Ejecuta una simulación rápida con pocos parámetros para verificar que funciona
"""
import sys
from pathlib import Path

# Agregar src/ al path para que los imports relativos funcionen
project_root = Path(__file__).parent.parent
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
import pandas as pd

def main():
    print("=" * 60)
    print("Probando run_system_analysis")
    print("=" * 60)
    
    # Prueba rápida con pocos parámetros
    print("\nEjecutando simulación con:")
    print("  - Eb/N0: [0, 3, 6] dB")
    print("  - Esquemas: PSK")
    print("  - M: [2, 4]")
    print("  - Bits: 10,000")
    print("  - Sin codificación de canal")
    print("\nEsto puede tomar unos segundos...\n")
    
    df = run_system_analysis(
        ebn0_db_range=[0, 3, 6],
        n_bits=10_000,
        mod_schemes=("psk",),
        M_list=(2, 4),
        use_channel_coding=False,
        reporter=None,
    )
    
    print("\n" + "=" * 60)
    print("Resultados:")
    print("=" * 60)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumnas: {list(df.columns)}")
    
    # Verificar que las columnas teóricas estén presentes
    expected_cols = ['mod_scheme', 'M', 'ebn0_db', 'Pb_sim', 'Pe_sim', 'Pb_theory', 'Pe_theory', 'coded']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"\n⚠️  ADVERTENCIA: Faltan columnas: {missing_cols}")
    else:
        print("\n✅ Todas las columnas esperadas están presentes")
    
    print("\nPrimeras filas (con valores teóricos):")
    print(df.head(10))
    
    print("\n" + "=" * 60)
    print("Comparación Simulado vs Teórico:")
    print("=" * 60)
    # Mostrar comparación para cada configuración
    for idx, row in df.iterrows():
        print(f"\n{row['mod_scheme'].upper()}-{row['M']}, Eb/N0={row['ebn0_db']:.1f} dB:")
        print(f"  BER: sim={row['Pb_sim']:.6f}, theory={row['Pb_theory']:.6f}, diff={abs(row['Pb_sim'] - row['Pb_theory']):.6f}")
        print(f"  SER: sim={row['Pe_sim']:.6f}, theory={row['Pe_theory']:.6f}, diff={abs(row['Pe_sim'] - row['Pe_theory']):.6f}")
    
    print("\n" + "=" * 60)
    print("Resumen estadístico:")
    print("=" * 60)
    summary = df.groupby(['mod_scheme', 'M']).agg({
        'Pb_sim': ['mean', 'min', 'max'],
        'Pe_sim': ['mean', 'min', 'max'],
        'Pb_theory': ['mean', 'min', 'max'],
        'Pe_theory': ['mean', 'min', 'max']
    })
    print(summary)
    
    print("\n" + "=" * 60)
    print("Prueba completada exitosamente!")
    print("=" * 60)
    
    # Guardar resultados en CSV
    output_file = "data/output/test_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResultados guardados en: {output_file}")
    
    # Generar gráficos
    print("\n" + "=" * 60)
    print("Generando gráficos...")
    print("=" * 60)
    
    try:
        # Gráfico Pe vs Eb/N0 para PSK
        path = plot_pe_vs_ebn0(df, "psk", "data/output")
        print(f"✅ Generado: {path}")
    except Exception as e:
        print(f"⚠️  Error generando gráfico Pe PSK: {e}")
    
    try:
        # Gráfico Pb vs Eb/N0 para PSK
        path = plot_pb_vs_ebn0(df, "psk", "data/output")
        print(f"✅ Generado: {path}")
    except Exception as e:
        print(f"⚠️  Error generando gráfico Pb PSK: {e}")
    
    try:
        # Comparación PSK vs FSK para M=4 (si hay datos)
        if len(df[(df["M"] == 4) & (df["coded"] == False)]) > 0:
            path = plot_psk_vs_fsk(df, 4, "data/output")
            print(f"✅ Generado: {path}")
    except Exception as e:
        print(f"⚠️  Error generando gráfico PSK vs FSK: {e}")
    
    print("\n" + "=" * 60)
    print("Prueba de gráficos completada!")
    print("=" * 60)

if __name__ == "__main__":
    main()

