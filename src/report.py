from collections import Counter
from math import log2, ceil
from pathlib import Path

def _printable(ch: str) -> str:
    # Representación legible para espacios y controles
    if ch == "\n": return "\\n"
    if ch == "\t": return "\\t"
    if ch == "\r": return "\\r"
    if ch == " ":  return "␠"
    return ch

def _bits_to_str(bits) -> str:
    return "".join("1" if b else "0" for b in bits)

def _encmap_ref(enc_dict) -> dict[str, str]:
    # source.build_huffman_dict → {char: tuple[int]} → "0101"
    return {ch: "".join(str(b) for b in code) for ch, code in enc_dict.items()}

def _encmap_mine(adapter) -> dict[str, str]:
    # adapter expone .h.caracteres: Dict[ch, List[bool]]
    return {ch: _bits_to_str(code) for ch, code in adapter.h.caracteres.items()}

def _entropy_from_probs(probs: dict[str, float]) -> float:
    return -sum(p * log2(p) for p in probs.values() if p > 0)

def _avg_len(probs: dict[str, float], encmap: dict[str, str]) -> float:
    return sum(probs[ch] * len(encmap[ch]) for ch in probs.keys())

def _fixed_length_bits(k: int) -> int:
    return ceil(log2(max(1, k)))

def _first_nonempty_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip() != "":
            return ln
    return text.splitlines()[0] if "\n" in text else text

def generate_report_files(text: str, encmap: dict[str, str], out_prefix: str):
    out_dir = Path(out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Probabilidades
    cnt = Counter(text)
    total = sum(cnt.values()) or 1
    probs = {ch: cnt[ch]/total for ch in cnt.keys()}

    # Métricas
    H = _entropy_from_probs(probs)
    Lavg = _avg_len(probs, encmap)
    Lmin = min(len(c) for c in encmap.values()) if encmap else 0
    eff = H / Lavg if Lavg > 0 else 0.0
    Lfix = _fixed_length_bits(len(probs))

    # 1) Tabla CSV: char, prob, code, len
    csv_path = Path(f"{out_prefix}_tabla_huffman.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("char,prob,code,len\n")
        for ch, p in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])):
            code = encmap.get(ch, "")
            f.write(f"{_printable(ch)},{p:.6f},{code},{len(code)}\n")

    # 2) Muestra: una línea → bits → decodif (se completa afuera)
    sample_path = Path(f"{out_prefix}_muestra.txt")
    sample_line = _first_nonempty_line(text)[:120]  # evita líneas gigantes
    sample_path.write_text(sample_line, encoding="utf-8")

    # 3) Métricas: MD sencillo (apto para copiar a LaTeX)
    md_path = Path(f"{out_prefix}_metricas.md")
    md = []
    md.append("### Resultados Huffman\n")
    md.append(f"- Entropía H: **{H:.4f}** bits/símbolo")
    md.append(f"- Longitud mínima (Lmin): **{Lmin}** bits")
    md.append(f"- Longitud promedio (L̄): **{Lavg:.4f}** bits/símbolo")
    md.append(f"- Eficiencia (η = H/L̄): **{eff*100:.2f}%**")
    md.append(f"- Código de longitud fija (p.ej. ASCII para k símbolos): **{Lfix}** bits/símbolo")
    md.append(f"- Alfabeto (k): **{len(probs)}** símbolos")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {
        "csv": str(csv_path),
        "sample_line": sample_line,
        "metrics_md": str(md_path),
        "H": H,
        "Lavg": Lavg,
        "Lmin": Lmin,
        "eff": eff,
        "Lfix": Lfix,
    }
