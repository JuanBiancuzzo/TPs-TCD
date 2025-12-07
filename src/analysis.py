"""
Módulo F - Análisis del sistema
Realiza análisis comparativo del rendimiento del sistema de comunicación
bajo diferentes condiciones (esquemas de modulación, niveles M, codificación de canal, etc.)
"""
from typing import Sequence, Dict, List
import numpy as np
import pandas as pd
from scipy.special import erfc
from math import log2, pi, sin
from pathlib import Path
import matplotlib.pyplot as plt
from enum import Enum

from report import Reporter, EmptyReporter
from file import File
from source import Source
from modulation import Modulation, Scheme
from channel import Channel
from cod_channel import ChannelCoding
from utils import BLUE, YELLOW, RED

MODULATION_ORDER = "M"
EBNO = "ebn0_db"
SIMULATION = "sim"
THEORY = "theory"

ENCODER = "Análisis"

class SymbolRep(Enum):
    SYMBOL = 0
    BIT = 1

    def __str__(self):
        if self == SymbolRep.SYMBOL:
            return "symb"
        else: 
            return "bit"


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
    return (1 if M == 2 else 2) * q_function(arg)

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
    
    return pe_psk_theoretical(M, ebn0_linear) / k

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
    return 0.5 * M * pe_fsk_theoretical(M, ebn0_linear) / (M - 1)

def run_system_analysis(
    path_in: str = "data/input/texto.txt",
    ebn0_db_range: List[float] = list(range(0, 11)),
    matriz_g: np.ndarray = np.eye(2),
    schemes: List[Scheme] = [Scheme.PSK, Scheme.FSK],
    M_list: List[int] = [2, 4, 8, 16],
    use_channel_coding: bool = False,
    reporter: Reporter = EmptyReporter(),
) -> Dict[Scheme, pd.DataFrame]:
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
    k, n = matriz_g.shape

    reporter.append_line(ENCODER, YELLOW, f"Ejecutando análisis {'con' if use_channel_coding else 'sin'} codificación de canal")
    reporter.append_line(ENCODER, YELLOW, "Parámetros:")
    reporter.append_line(ENCODER, YELLOW, "\n\t".join([
        f"- Eb/N0: ({ebn0_db_range[0]} - {ebn0_db_range[-1]}) dB",
        f"-f Esquemas: {', '.join(map(Scheme.__str__, schemes))}",
        f"- M: {M_list}",
    ]))

    file = File()
    input_text = file.encode(path_in, reporter)

    source = Source()
    complete_encoding = source.encode(input_text, reporter)

    bit_ranges = [ slice(0, len(complete_encoding) // 2**(i - 1)) for i in range(8, 0, -1) ]
    
    # Lista para almacenar resultados
    results = {}


    for scheme in schemes:
        scheme_result = []
        for M in M_list:
            for ebn0_db in ebn0_db_range:

                ebn0_linear = 10**(ebn0_db / 10.0)
                if scheme == Scheme.PSK:
                    Pe_theory = pe_psk_theoretical(M, ebn0_linear)
                    Pb_theory = pb_psk_theoretical(M, ebn0_linear)
                elif scheme == Scheme.FSK:
                    Pe_theory = pe_fsk_theoretical(M, ebn0_linear)
                    Pb_theory = pb_fsk_theoretical(M, ebn0_linear)

                scheme_result.append({
                    MODULATION_ORDER: M,
                    EBNO: ebn0_db,
                    f"{THEORY}_{SymbolRep.BIT}": Pb_theory,
                    f"{THEORY}_{SymbolRep.SYMBOL}": Pe_theory,
                })

        results[scheme] = scheme_result
    
    # Iterar sobre todas las combinaciones de parámetros

    for scheme in schemes:
        for i, M in enumerate(M_list):
            ebno_counter = 0
            double_counter = 0

            while ebno_counter < len(ebn0_db_range):
                ebn0_db = ebn0_db_range[ebno_counter]

                # Generar bits aleatorios
                bits_original = complete_encoding[bit_ranges[double_counter]]
                bits = bits_original.copy()
                
                # Codificación de canal (opcional)
                if use_channel_coding:
                    channel_coder = ChannelCoding(n = n, k = k, matriz_generadora = matriz_g)
                    bits = channel_coder.encode(bits, reporter)

                scale_energy = 1.0
                if use_channel_coding:
                    # Ec = Eb * k/n 
                    scale_energy = k / n
                
                # Modulación
                modulator = Modulation(scheme = scheme, M = M, scale_energy = scale_energy)
                symbols = modulator.encode(bits, reporter)
                
                # Canal AWGN
                channel = Channel(eb_n0_db = ebn0_db)
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
                if scheme == Scheme.PSK:
                    Pb_theory = pb_psk_theoretical(M, ebn0_linear)
                elif scheme == Scheme.FSK:
                    Pb_theory = pb_fsk_theoretical(M, ebn0_linear)

                if Pb_sim * 10 < Pb_theory:
                    double_counter += 1

                    if double_counter < len(bit_ranges):
                        reporter.append_line(ENCODER, RED, f"Se necesitó aumentar la cantidad de bits")

                    else:
                        ebno_counter += 1
                        double_counter = 0
                        reporter.append_line(ENCODER, RED, "Se llego al maximo de bits, se ignora la medicion")
                        
                    continue

                # Agregar resultado
                line = i * len(ebn0_db_range) + ebno_counter
                results[scheme][line][f"{SIMULATION}_{SymbolRep.BIT}"] = Pb_sim
                results[scheme][line][f"{SIMULATION}_{SymbolRep.SYMBOL}"] = Pe_sim

                ebno_counter += 1
                double_counter = 0

        results[scheme] = pd.DataFrame(results[scheme])
    
    return results

# ----------------------------------------------------
# ----- Gráficos -------------------------------------
# ----------------------------------------------------
def plot_prob_vs_ebn0(
    df: pd.DataFrame, # M, ebno, sim_bit, sim_symb, theory_bit, theory_symb
    representation: SymbolRep,
    scheme: Scheme,
    output_dir: str = "data/analysis",
) -> None:
    _, ax = plt.subplots(figsize = (10, 7))

    M_list = sorted(df[MODULATION_ORDER].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(M_list)))

    y_bottom = 1
    
    for M, color in zip(M_list, colors):
        df_M = df[df[MODULATION_ORDER] == M].sort_values(EBNO)
        
        # Curva teórica
        ax.semilogy(
            df_M[EBNO], df_M[f"{THEORY}_{representation}"], 
            linestyle = "--", color = color, linewidth = 2,
            label = f"M={M} (teórico)",
        )
        
        # Curva simulada
        ax.scatter(
            df_M[EBNO], df_M[f"{SIMULATION}_{representation}"], 
            marker = "o", color = color, linewidths = 1.5, s = 100,
            label=f"M={M} (simulado)",
        )

        y_bottom = min(y_bottom, 0.5 * df_M[f"{SIMULATION}_{representation}"].min())

    ax.set_xlabel("Eb/N0 (dB)", fontsize = 12)
    if representation == SymbolRep.BIT:
        ax.set_ylabel("Probabilidad de Error de Bit (Pb)", fontsize = 12)
    else:
        ax.set_ylabel("Probabilidad de Error de Símbolo (Pe)", fontsize = 12)
    
    ax.grid(True, alpha = 0.3)
    ax.legend(loc = "upper right", fontsize = 10)
    ax.set_ylim(bottom = max(1e-8 if scheme == Scheme.FSK else 1e-6, y_bottom), top = 1.0)

    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path = output_path / f"analysis_P{'e' if representation == SymbolRep.SYMBOL else 'b'}_vs_EbN0_{scheme}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(output_path)

def plot_compare_scheme_per_M(
    dfs: Dict[Scheme, pd.DataFrame],
    M: int,
    representation: SymbolRep,
    output_dir: str = "data/analysis",
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
    _, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(dfs)))

    for color, (scheme, df) in zip(colors, dfs.items()):
        df_M = df[df[MODULATION_ORDER] == M].sort_values(EBNO)
        if len(df_M) == 0:
            raise ValueError(f"En modulacion {scheme} no hay orden = {M}")

        ax.semilogy(
            df_M[EBNO], df_M[f"{THEORY}_{representation}"],
            linestyle = "--", color = color, linewidth = 2, 
            label = f"{M}-{scheme} (teórico)",
        )

        ax.scatter(
            df_M[EBNO], df_M[f"{SIMULATION}_{representation}"], 
            marker = "o", color = color, linewidths = 1.5, s = 100,
            label = f"{M}-{scheme} (simulado)",
        )
    
    ax.set_xlabel("Eb/N0 (dB)", fontsize = 12)
    if representation == SymbolRep.BIT:
        ax.set_ylabel("Probabilidad de Error de Bit (Pb)", fontsize = 12)
    else:
        ax.set_ylabel("Probabilidad de Error de Símbolo (Pe)", fontsize = 12)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(bottom=1e-6, top=1.0)
    
    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path = output_path / f"analysis_{'_vs_'.join(map(Scheme.__str__, dfs.keys()))}_M_{M}_{representation}.png"

    plt.savefig(output_path, dpi = 150, bbox_inches = "tight")
    plt.close()
    
    return str(output_path)

def plot_coding_comparison(
    dfs_without_code: Dict[Scheme, pd.DataFrame],
    dfs_with_code: Dict[Scheme, pd.DataFrame],
    scheme: Scheme,
    M: int,
    output_dir: str = "data/analysis",
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
    _, axis = plt.subplots(1, 2, figsize = (14, 6))

    lines = ["-.", "--"]
    markers = ["s", "o"]
    labels = ["Con código", "Sin código"]

    df_without_code = dfs_without_code[scheme]
    df_with_code = dfs_with_code[scheme]
    
    # Gráfico de Pe
    for ax, rep in zip(list(axis), [SymbolRep.SYMBOL, SymbolRep.BIT]):
        for df, line, marker, label in zip([df_with_code, df_without_code], lines, markers, labels):
            df_M = df[df[MODULATION_ORDER] == M].sort_values(EBNO)

            ax.semilogy(
                df_M[EBNO], df_M[f"{SIMULATION}_{rep}"],
                marker = marker, linestyle = line, linewidth = 2, markersize = 6,
                label=label,
            )

        ylims = (
            0.5 * min(df[f"{SIMULATION}_{rep}"][df[MODULATION_ORDER] == M].min() for df in [df_with_code, df_without_code]),
            1.2 * max(df[f"{SIMULATION}_{rep}"][df[MODULATION_ORDER] == M].max() for df in [df_with_code, df_without_code]),
        )
    
        ax.set_xlabel("Eb/N0 (dB)", fontsize = 11)
        if rep == SymbolRep.BIT:
            ax.set_ylabel("Probabilidad de Error de Bit (Pb)", fontsize = 11)
        else:
            ax.set_ylabel("Probabilidad de Error de Símbolo (Pe)", fontsize = 11)

        ax.grid(True, alpha = 0.3)
        ax.legend(loc = "upper right", fontsize = 10)
        ax.set_ylim(ylims)

    plt.tight_layout()
    
    # Guardar gráfico
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"analysis_coding_comparison_{scheme}_M_{M}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return str(output_path)

def generate_all_plots(
    dfs_without_code: Dict[Scheme, pd.DataFrame],
    dfs_with_code: Dict[Scheme, pd.DataFrame],
    output_dir: str = "data/analysis",
    reporter: Reporter = EmptyReporter(),
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
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    
    for scheme, df in dfs_without_code.items():
        try:
            # Gráficos Pe vs Eb/N0 para cada esquema
            path = plot_prob_vs_ebn0(df, SymbolRep.SYMBOL, scheme, output_dir)
            reporter.append_line("Analysis", BLUE, f"Gráfico Pe vs Eb/N0 → {path}")
            paths.append(path)

            # Gráficos Pb vs Eb/N0 para cada esquema
            path = plot_prob_vs_ebn0(df, SymbolRep.BIT, scheme, output_dir)
            reporter.append_line("Analysis", BLUE, f"Gráfico Pb vs Eb/N0 → {path}")
            paths.append(path)

        except ValueError as e:
            reporter.append_line("Analysis", YELLOW, f"⚠️ Error en comparacion entre Ms con error: {e}")
    
    M_values = [2, 4, 8, 16]
    schemes_codes = [ Scheme.PSK, Scheme.FSK ]

    # Comparación de esquemas para diferentes M
    for M in M_values: 
        try:
            path = plot_compare_scheme_per_M(dfs_without_code, M, SymbolRep.SYMBOL, output_dir)
            reporter.append_line("Analysis", BLUE, f"Gráfico comparacion (Pe) esquemas con M = {M} → {path}")
            paths.append(path)

            path = plot_compare_scheme_per_M(dfs_without_code, M, SymbolRep.BIT, output_dir)
            reporter.append_line("Analysis", BLUE, f"Gráfico comparacion (Pb) esquemas con M = {M} → {path}")
            paths.append(path)

        except ValueError as e:
            reporter.append_line("Analysis", YELLOW, f"⚠️  Error en la comparacion entre esquemas con error: {e}")
    
    # Comparación con/sin código para combinaciones disponibles
    for scheme in schemes_codes:
        for M in M_values:
            try:
                path = plot_coding_comparison(dfs_without_code, dfs_with_code, scheme, M, output_dir)
                reporter.append_line("Analysis", BLUE, f"Gráfico comparación codificación → {path}")
                paths.append(path)

            except ValueError as e:
                reporter.append_line("Analysis", YELLOW, f"⚠️ Error comparacion con/sin bloque de codificacion de canal con error: {e}")
    
    return paths

