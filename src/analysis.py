"""
Módulo F - Análisis del sistema
Realiza análisis comparativo del rendimiento del sistema de comunicación
bajo diferentes condiciones (esquemas de modulación, niveles M, codificación de canal, etc.)
"""
from typing import Iterable, Sequence, Optional
import numpy as np
import pandas as pd
from scipy.special import erfc
from math import log2, pi, sin
from pathlib import Path
import matplotlib.pyplot as plt

from report import Reporter, EmptyReporter
from modulation import Modulation, Scheme
from channel import Channel
from cod_channel import ChannelCoding
from utils import BLUE, YELLOW


def q_function(x: float) -> float:
    return 0.5 * erfc(x / np.sqrt(2))


def pe_psk_theoretical(M: int, ebn0_linear: float) -> float:
    """
    Probabilidad de error de símbolo teórica para M-PSK.
    
    Usa la aproximación de vecinos más cercanos:
    Pe ≈ 2 * Q(sqrt(2*k*Eb/N0) * sin(pi/M))
    donde k = log2(M)
    
    Args:
        M: Número de símbolos (2, 4, 8, 16, ...)
        ebn0_linear: Eb/N0 en escala lineal (no dB)
    
    Returns:
        Probabilidad de error de símbolo teórica
    """
    k = log2(M)
    if k <= 0 or ebn0_linear <= 0:
        return 1.0
    
    arg = np.sqrt(2 * k * ebn0_linear) * sin(pi / M)
    return 2 * q_function(arg)


def pb_psk_theoretical(M: int, ebn0_linear: float) -> float:
    """
    Probabilidad de error de bit teórica para M-PSK.
    
    Para M-PSK con codificación Gray, la aproximación es:
    Pb ≈ Pe / k para M grande
    Para M=2 (BPSK): Pb = Q(sqrt(2*Eb/N0))
    Para M=4 (QPSK): Pb ≈ Q(sqrt(2*Eb/N0))
    
    Args:
        M: Número de símbolos (2, 4, 8, 16, ...)
        ebn0_linear: Eb/N0 en escala lineal (no dB)
    
    Returns:
        Probabilidad de error de bit teórica
    """
    k = log2(M)
    if k <= 0 or ebn0_linear <= 0:
        return 1.0
    
    if M == 2:
        # BPSK: Pb = Q(sqrt(2*Eb/N0))
        return q_function(np.sqrt(2 * ebn0_linear))
    elif M == 4:
        # QPSK: Pb ≈ Q(sqrt(2*Eb/N0))
        return q_function(np.sqrt(2 * ebn0_linear))
    else:
        # Para M > 4, aproximación: Pb ≈ Pe / k
        pe = pe_psk_theoretical(M, ebn0_linear)
        return pe / k


def pe_fsk_theoretical(M: int, ebn0_linear: float) -> float:
    """
    Probabilidad de error de símbolo teórica para M-FSK.
    
    Usa la aproximación de vecinos más cercanos:
    Pe ≈ (M-1) * Q(sqrt(k*Eb/N0))
    donde k = log2(M)
    
    Args:
        M: Número de símbolos (2, 4, 8, 16, ...)
        ebn0_linear: Eb/N0 en escala lineal (no dB)
    
    Returns:
        Probabilidad de error de símbolo teórica
    """
    k = log2(M)
    if k <= 0 or ebn0_linear <= 0:
        return 1.0
    
    arg = np.sqrt(k * ebn0_linear)
    return (M - 1) * q_function(arg)


def pb_fsk_theoretical(M: int, ebn0_linear: float) -> float:
    """
    Probabilidad de error de bit teórica para M-FSK.
    
    Para M-FSK, la aproximación es:
    Pb ≈ (M/2) / (M-1) * Pe
    
    Args:
        M: Número de símbolos (2, 4, 8, 16, ...)
        ebn0_linear: Eb/N0 en escala lineal (no dB)
    
    Returns:
        Probabilidad de error de bit teórica
    """
    if M <= 1 or ebn0_linear <= 0:
        return 1.0
    
    pe = pe_fsk_theoretical(M, ebn0_linear)
    # Para M=2 (BFSK), Pb = Pe
    if M == 2:
        return pe
    else:
        # Para M > 2: Pb ≈ (M/2) * Q(sqrt(k*Eb/N0))
        # Como Pe = (M-1) * Q(...), entonces Pb = (M/2) / (M-1) * Pe = (M/2) * Q(...)
        # El (M-1) se cancela, así que calculamos directamente
        k = log2(M)
        arg = np.sqrt(k * ebn0_linear)
        return (M / 2.0) * q_function(arg)


def run_system_analysis(
    ebn0_db_range: Iterable[float] = range(0, 11),
    n_bits: int = 100_000,
    mod_schemes: Sequence[str] = ("psk", "fsk"),
    M_list: Sequence[int] = (2, 4, 8, 16),
    use_channel_coding: bool = False,
    reporter: Optional[Reporter] = None,
) -> pd.DataFrame:
    """
    Realiza un análisis sistemático del rendimiento del sistema de comunicación.
    
    Evalúa diferentes configuraciones del sistema (esquemas de modulación, niveles M,
    uso de codificación de canal) bajo distintas condiciones de ruido (Eb/N0) y retorna
    un DataFrame con las métricas de rendimiento obtenidas (BER, SER, etc.).
    
    Args:
        ebn0_db_range: Rango de valores de Eb/N0 en dB a evaluar. Por defecto range(0, 11).
        n_bits: Número de bits a procesar para cada configuración. Por defecto 100,000.
        mod_schemes: Esquemas de modulación a evaluar. Por defecto ("psk", "fsk").
        M_list: Lista de niveles M (número de símbolos) a evaluar. Por defecto (2, 4, 8, 16).
        use_channel_coding: Si True, incluye codificación de canal en el análisis.
                           Por defecto False.
        reporter: Instancia de Reporter para generar reportes y gráficos.
                 Si es None, no se generan reportes. Por defecto None.
    
    Returns:
        pd.DataFrame: DataFrame con las métricas de rendimiento para cada configuración
                     evaluada.
    """
    # RNG reproducible
    rng = np.random.default_rng(seed=42)
    
    # Usar EmptyReporter si no se proporciona uno
    if reporter is None:
        reporter = EmptyReporter()
    
    # Matriz generadora para codificación de canal (misma que en main.py)
    matriz_g = np.array([
        [1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
    ])
    
    # Lista para almacenar resultados
    results = []
    
    # Convertir strings de mod_schemes a Scheme enum
    scheme_map = {"psk": Scheme.PSK, "fsk": Scheme.FSK}
    
    # Iterar sobre todas las combinaciones de parámetros
    for mod_scheme_str in mod_schemes:
        mod_scheme_enum = scheme_map[mod_scheme_str.lower()]
        
        for M in M_list:
            for ebn0_db in ebn0_db_range:
                # Generar bits aleatorios
                bits_original = rng.integers(0, 2, size=n_bits, dtype=np.int8)
                bits = bits_original.copy()
                
                # Codificación de canal (opcional)
                if use_channel_coding:
                    channel_coder = ChannelCoding(n=15, k=5, matriz_generadora=matriz_g)
                    bits = channel_coder.encode(bits, reporter)
                
                # Modulación
                modulator = Modulation(scheme=mod_scheme_enum, M=M)
                symbols = modulator.encode(bits, reporter)
                
                # Canal AWGN
                channel = Channel(eb_n0_db=ebn0_db, rng=rng)
                symbols_noisy = channel.encode(symbols, reporter)
                
                # Demodulación
                bits_demod = modulator.decode(symbols_noisy, reporter)
                
                # Decodificación de canal (opcional)
                if use_channel_coding:
                    bits_demod = channel_coder.decode(bits_demod, reporter)
                
                # Calcular BER (Pb_sim) - comparar bits originales con recibidos
                # Asegurar que ambos arrays tengan la misma longitud
                min_len = min(len(bits_original), len(bits_demod))
                bits_original_trunc = bits_original[:min_len]
                bits_demod_trunc = bits_demod[:min_len]
                Pb_sim = np.mean(bits_original_trunc != bits_demod_trunc).astype(float)
                
                # Calcular SER (Pe_sim) - usar método interno de Modulation
                # Este método convierte bits a símbolos y compara
                # Asegurar que ambos arrays tengan la misma longitud (múltiplo de k para evitar problemas)
                k = modulator.k
                # Truncar a múltiplo de k para evitar problemas en la conversión a símbolos
                trunc_len = (min_len // k) * k
                if trunc_len > 0:
                    bits_original_for_ser = bits_original_trunc[:trunc_len]
                    bits_demod_for_ser = bits_demod_trunc[:trunc_len]
                    Pe_sim = modulator._estimated_symbol_error_proba(
                        bits_original_for_ser, bits_demod_for_ser
                    ).astype(float)
                else:
                    Pe_sim = np.nan
                
                # Calcular valores teóricos
                ebn0_linear = 10**(ebn0_db / 10.0)
                if mod_scheme_str.lower() == "psk":
                    Pe_theory = pe_psk_theoretical(M, ebn0_linear)
                    Pb_theory = pb_psk_theoretical(M, ebn0_linear)
                elif mod_scheme_str.lower() == "fsk":
                    Pe_theory = pe_fsk_theoretical(M, ebn0_linear)
                    Pb_theory = pb_fsk_theoretical(M, ebn0_linear)
                else:
                    Pe_theory = np.nan
                    Pb_theory = np.nan
                
                # Agregar resultado
                results.append({
                    "mod_scheme": mod_scheme_str.lower(),
                    "M": M,
                    "ebn0_db": ebn0_db,
                    "Pb_sim": Pb_sim,
                    "Pe_sim": Pe_sim,
                    "Pb_theory": Pb_theory,
                    "Pe_theory": Pe_theory,
                    "coded": use_channel_coding,
                })
    
    # Crear y retornar DataFrame
    df = pd.DataFrame(results)
    return df

# ----------------------------------------------------
# ----- Gráficos -------------------------------------
# ----------------------------------------------------
def plot_pe_vs_ebn0(
    df: pd.DataFrame,
    mod_scheme: str,
    output_dir: str = "data/analysis",
    reporter: Optional[Reporter] = None,
) -> str:
    """
    Grafica Pe (probabilidad de error de símbolo) vs Eb/N0 con curvas teórica y simulada
    para diferentes valores de M (2, 4, 8, 16).
    
    Args:
        df: DataFrame con los resultados de run_system_analysis
        mod_scheme: Esquema de modulación ("psk" o "fsk")
        output_dir: Directorio donde guardar el gráfico
        reporter: Reporter opcional para logging
    
    Returns:
        Path del archivo PNG guardado
    """
    # Filtrar datos del esquema y sin codificación (para comparar con teórico)
    df_filtered = df[(df["mod_scheme"] == mod_scheme.lower()) & (df["coded"] == False)]
    
    if len(df_filtered) == 0:
        raise ValueError(f"No hay datos para el esquema {mod_scheme}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    M_list = sorted(df_filtered["M"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(M_list)))
    
    for M, color in zip(M_list, colors):
        df_M = df_filtered[df_filtered["M"] == M].sort_values("ebn0_db")
        
        # Curva teórica
        ax.semilogy(
            df_M["ebn0_db"],
            df_M["Pe_theory"],
            linestyle="--",
            color=color,
            linewidth=2,
            label=f"M={M} (teórico)",
        )
        
        # Curva simulada
        ax.semilogy(
            df_M["ebn0_db"],
            df_M["Pe_sim"],
            marker="o",
            linestyle="-",
            color=color,
            linewidth=1.5,
            markersize=6,
            label=f"M={M} (simulado)",
        )
    
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Probabilidad de Error de Símbolo (Pe)", fontsize=12)
    ax.set_title(f"Pe vs Eb/N0 - {mod_scheme.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=1e-6, top=1.0)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"analysis_Pe_vs_EbN0_{mod_scheme.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    if reporter:
        reporter.append_line("Analysis", BLUE, f"Gráfico Pe vs Eb/N0 → {output_path}")
    
    return str(output_path)


def plot_pb_vs_ebn0(
    df: pd.DataFrame,
    mod_scheme: str,
    output_dir: str = "data/analysis",
    reporter: Optional[Reporter] = None,
) -> str:
    """
    Grafica Pb (probabilidad de error de bit) vs Eb/N0 con curvas teórica y simulada
    para diferentes valores de M (2, 4, 8, 16).
    
    Args:
        df: DataFrame con los resultados de run_system_analysis
        mod_scheme: Esquema de modulación ("psk" o "fsk")
        output_dir: Directorio donde guardar el gráfico
        reporter: Reporter opcional para logging
    
    Returns:
        Path del archivo PNG guardado
    """
    # Filtrar datos del esquema y sin codificación
    df_filtered = df[(df["mod_scheme"] == mod_scheme.lower()) & (df["coded"] == False)]
    
    if len(df_filtered) == 0:
        raise ValueError(f"No hay datos para el esquema {mod_scheme}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    M_list = sorted(df_filtered["M"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(M_list)))
    
    for M, color in zip(M_list, colors):
        df_M = df_filtered[df_filtered["M"] == M].sort_values("ebn0_db")
        
        # Curva teórica
        ax.semilogy(
            df_M["ebn0_db"],
            df_M["Pb_theory"],
            linestyle="--",
            color=color,
            linewidth=2,
            label=f"M={M} (teórico)",
        )
        
        # Curva simulada
        ax.semilogy(
            df_M["ebn0_db"],
            df_M["Pb_sim"],
            marker="o",
            linestyle="-",
            color=color,
            linewidth=1.5,
            markersize=6,
            label=f"M={M} (simulado)",
        )
    
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Probabilidad de Error de Bit (Pb)", fontsize=12)
    ax.set_title(f"Pb vs Eb/N0 - {mod_scheme.upper()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=1e-6, top=1.0)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"analysis_Pb_vs_EbN0_{mod_scheme.lower()}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    if reporter:
        reporter.append_line("Analysis", BLUE, f"Gráfico Pb vs Eb/N0 → {output_path}")
    
    return str(output_path)


def plot_psk_vs_fsk(
    df: pd.DataFrame,
    M: int,
    output_dir: str = "data/analysis",
    reporter: Optional[Reporter] = None,
) -> str:
    """
    Compara Pb teórica de PSK vs FSK para un valor fijo de M.
    
    Args:
        df: DataFrame con los resultados de run_system_analysis
        M: Valor de M (número de símbolos) a comparar
        output_dir: Directorio donde guardar el gráfico
        reporter: Reporter opcional para logging
    
    Returns:
        Path del archivo PNG guardado
    """
    # Filtrar datos para M fijo y sin codificación
    df_filtered = df[(df["M"] == M) & (df["coded"] == False)]
    
    if len(df_filtered) == 0:
        raise ValueError(f"No hay datos para M={M}")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    schemes = ["psk", "fsk"]
    colors = {"psk": "blue", "fsk": "red"}
    linestyles = {"psk": "-", "fsk": "--"}
    
    for scheme in schemes:
        df_scheme = df_filtered[df_filtered["mod_scheme"] == scheme].sort_values("ebn0_db")
        
        if len(df_scheme) > 0:
            ax.semilogy(
                df_scheme["ebn0_db"],
                df_scheme["Pb_theory"],
                linestyle=linestyles[scheme],
                color=colors[scheme],
                linewidth=2,
                marker="o",
                markersize=6,
                label=f"{scheme.upper()}-{M}",
            )
    
    ax.set_xlabel("Eb/N0 (dB)", fontsize=12)
    ax.set_ylabel("Probabilidad de Error de Bit (Pb) - Teórica", fontsize=12)
    ax.set_title(f"Comparación PSK vs FSK - M={M}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=1e-6, top=1.0)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"analysis_PSK_vs_FSK_M{M}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    if reporter:
        reporter.append_line("Analysis", BLUE, f"Gráfico PSK vs FSK → {output_path}")
    
    return str(output_path)


def plot_coding_comparison(
    df: pd.DataFrame,
    mod_scheme: str,
    M: int,
    output_dir: str = "data/analysis",
    reporter: Optional[Reporter] = None,
) -> str:
    """
    Compara Pe_sim y Pb_sim vs Eb/N0 con y sin codificación de canal para una combinación fija.
    
    Args:
        df: DataFrame con los resultados de run_system_analysis
        mod_scheme: Esquema de modulación ("psk" o "fsk")
        M: Valor de M (número de símbolos)
        output_dir: Directorio donde guardar el gráfico
        reporter: Reporter opcional para logging
    
    Returns:
        Path del archivo PNG guardado
    """
    # Filtrar datos para la combinación específica
    df_filtered = df[(df["mod_scheme"] == mod_scheme.lower()) & (df["M"] == M)]
    
    if len(df_filtered) == 0:
        raise ValueError(f"No hay datos para {mod_scheme.upper()}-{M}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calcular límites dinámicos basados en los datos
    pe_max = df_filtered["Pe_sim"].max()
    pe_min = df_filtered["Pe_sim"][df_filtered["Pe_sim"] > 0].min()  # Excluir ceros
    pb_max = df_filtered["Pb_sim"].max()
    pb_min = df_filtered["Pb_sim"][df_filtered["Pb_sim"] > 0].min()  # Excluir ceros
    
    # Establecer límites con margen (1.2x el máximo, pero mínimo 0.5 para Pe y 0.2 para Pb)
    pe_top = max(pe_max * 1.2, 0.5) if not np.isnan(pe_max) else 0.5
    pe_bottom = max(pe_min * 0.5, 1e-6) if not np.isnan(pe_min) else 1e-6
    pb_top = max(pb_max * 1.2, 0.2) if not np.isnan(pb_max) else 0.2
    pb_bottom = max(pb_min * 0.5, 1e-6) if not np.isnan(pb_min) else 1e-6
    
    # Gráfico de Pe
    for coded in [False, True]:
        df_coded = df_filtered[df_filtered["coded"] == coded].sort_values("ebn0_db")
        
        if len(df_coded) > 0:
            label = "Con código" if coded else "Sin código"
            marker = "s" if coded else "o"
            linestyle = "-." if coded else "-"
            
            ax1.semilogy(
                df_coded["ebn0_db"],
                df_coded["Pe_sim"],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=label,
            )
    
    ax1.set_xlabel("Eb/N0 (dB)", fontsize=11)
    ax1.set_ylabel("Pe (simulado)", fontsize=11)
    ax1.set_title(f"Pe vs Eb/N0 - {mod_scheme.upper()}-{M}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(bottom=pe_bottom, top=pe_top)
    
    # Gráfico de Pb
    for coded in [False, True]:
        df_coded = df_filtered[df_filtered["coded"] == coded].sort_values("ebn0_db")
        
        if len(df_coded) > 0:
            label = "Con código" if coded else "Sin código"
            marker = "s" if coded else "o"
            linestyle = "-." if coded else "-"
            
            ax2.semilogy(
                df_coded["ebn0_db"],
                df_coded["Pb_sim"],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=6,
                label=label,
            )
    
    ax2.set_xlabel("Eb/N0 (dB)", fontsize=11)
    ax2.set_ylabel("Pb (simulado)", fontsize=11)
    ax2.set_title(f"Pb vs Eb/N0 - {mod_scheme.upper()}-{M}", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.set_ylim(bottom=pb_bottom, top=pb_top)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"analysis_coding_comparison_{mod_scheme.lower()}_M{M}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    if reporter:
        reporter.append_line("Analysis", BLUE, f"Gráfico comparación codificación → {output_path}")
    
    return str(output_path)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: str = "data/analysis",
    reporter: Optional[Reporter] = None,
) -> list[str]:
    """
    Genera todos los gráficos de análisis disponibles.
    
    Args:
        df: DataFrame con los resultados de run_system_analysis
        output_dir: Directorio donde guardar los gráficos
        reporter: Reporter opcional para logging
    
    Returns:
        Lista de paths de los archivos PNG generados
    """
    paths = []
    
    # Asegurar que el directorio existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Gráficos Pe vs Eb/N0 para cada esquema
    for scheme in ["psk", "fsk"]:
        if len(df[df["mod_scheme"] == scheme]) > 0:
            try:
                path = plot_pe_vs_ebn0(df, scheme, output_dir, reporter)
                paths.append(path)
            except ValueError as e:
                if reporter:
                    reporter.append_line("Analysis", YELLOW, f"⚠️  {e}")
    
    # Gráficos Pb vs Eb/N0 para cada esquema
    for scheme in ["psk", "fsk"]:
        if len(df[df["mod_scheme"] == scheme]) > 0:
            try:
                path = plot_pb_vs_ebn0(df, scheme, output_dir, reporter)
                paths.append(path)
            except ValueError as e:
                if reporter:
                    reporter.append_line("Analysis", YELLOW, f"⚠️  {e}")
    
    # Comparación PSK vs FSK para diferentes M
    M_values = sorted(df[df["coded"] == False]["M"].unique())
    for M in M_values:
        try:
            path = plot_psk_vs_fsk(df, M, output_dir, reporter)
            paths.append(path)
        except ValueError as e:
            if reporter:
                reporter.append_line("Analysis", "YELLOW", f"⚠️  {e}")
    
    # Comparación con/sin código para combinaciones disponibles
    for scheme in ["psk", "fsk"]:
        for M in M_values:
            df_subset = df[(df["mod_scheme"] == scheme) & (df["M"] == M)]
            if len(df_subset[df_subset["coded"] == True]) > 0 and len(df_subset[df_subset["coded"] == False]) > 0:
                try:
                    path = plot_coding_comparison(df, scheme, M, output_dir, reporter)
                    paths.append(path)
                except ValueError as e:
                    if reporter:
                        reporter.append_line("Analysis", "YELLOW", f"⚠️  {e}")
    
    return paths

