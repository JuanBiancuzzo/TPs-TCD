from collections import Counter
from math import log2
from typing import Dict, List, Tuple, Any
import heapq

from pipeline import EncoderDecoder
from report import Reporter, _first_nonempty_line, _printable 
import utils

Bit = int  # Representa un bit (0 o 1)
Code = Tuple[Bit, ...]  # Código binario como tupla de bits (prefijo libre)
EncDict = Dict[str, Code]  # Diccionario: símbolo -> código
DecDict = Dict[Code, str]  # Diccionario: código -> símbolo

class Source(EncoderDecoder):
    def encode(self, text: str, reporter: Reporter | None) -> List[Bit]:
        """
        Codifica un texto a una secuencia de bits usando un diccionario de Huffman.

        Args:
            text: cadena de entrada.
            enc: diccionario {símbolo: código}.

        Returns:
            Lista de bits (0/1).
        """
        self.probs = _symbol_probs(text)
        tree = _build_huffman_tree(self.probs)

        self.encoder: EncDict = {}
        _walk(tree, tuple(), self.encoder)
        self.decoder: DecDict = {code: ch for ch, code in self.encoder.items()}

        if reporter is not None:
            self._report(text, reporter)

        return self._encode_text(text)

    def decode(self, bits: List[Bit]) -> str:
        """
        Decodifica una secuencia de bits en un texto usando el diccionario inverso.

        Args:
            bits: lista de bits (0/1).
            dec: diccionario {código: símbolo}.

        Returns:
            Texto decodificado.
        """
        out_chars: List[str] = []    # texto reconstruido
        buf: List[Bit] = []          # buffer de bits acumulados
        dec_keys = set(self.decoder.keys())
        lens = sorted({len(k) for k in dec_keys})  # longitudes posibles de códigos
        maxlen = max(lens) if lens else 1
        as_tuple = tuple  # alias para optimizar

        for b in bits:
            buf.append(b)
            if len(buf) > maxlen:    # seguridad: si algo va mal, limpiar
                buf.clear()
                continue
            t = as_tuple(buf)
            if t in self.decoder:             # si el buffer matchea un código
                out_chars.append(self.decoder[t])
                buf.clear()

        # Si quedan bits sin cerrar, se ignoran (robustez ante ruido)
        return "".join(out_chars)

    def _encode_text(self, text: str) -> List[Bit]:
        bits: List[Bit] = []
        for ch in text:
            bits.extend(self.encoder[ch])  # concatena el código del símbolo
        return bits

    def _report(self, text: str, reporter: Reporter):
        sample_line     = _first_nonempty_line(text)[:120]
        sample_bits     = self._encode_text(sample_line)
        sample_bits_str = "".join(str(b) for b in sample_bits)
        sample_dec      = self.decode(sample_bits)

        H     = self.entropy()
        Lavg  = self.avg_code_length()
        eff   = utils.efficiency(H, Lavg)
        Lfix  = utils.fixed_length_bits(len(self.probs))
        Lmin =  min([len("".join(str(b) for b in code)) for _, code in self.encoder.items()])

        headers = ["char", "prob", "code", "len"]
        data = []
        for ch, p in sorted(self.probs.items(), key=lambda kv: (-kv[1], kv[0])):
            code = "".join(str(b) for b in self.encoder[ch])
            data.append([_printable(ch), f"{p:.6f}", code, str(len(code))])

        result_path = reporter.report_results("tabla_huffman.csv", headers, data)
        
        line = f"H={H:.4f} | Lavg={Lavg:.4f} | η={eff*100:.2f}% | L_fijo≈{Lfix}"
        reporter.append_line("Fuente/Huffman", utils.BLUE, line)

        report_metrics = "\n".join([
            "### Resultados Huffman\n",
            f"- Entropía H: **{H:.4f}** bits/símbolo",
            f"- Longitud mínima (Lmin): **{Lmin}** bits",
            f"- Longitud promedio (L̄): **{Lavg:.4f}** bits/símbolo",
            f"- Eficiencia (η = H/L̄): **{eff*100:.2f}%**",
            f"- Código de longitud fija (p.ej. ASCII para k símbolos): **{Lfix}** bits/símbolo",
            f"- Alfabeto (k): **{len(self.probs)}** símbolos",
            "\n---\n",
            "#### Muestra (línea → binario → decodificada)\n",
            f"- Línea: `{sample_line}`",
            f"- Binario: `{sample_bits_str}`",
            f"- Decodificada: `{sample_dec}`",
            "\n#### Tabla CSV generada",
            f"- `{result_path}`",
        ])
        reporter.append_metrics(report_metrics)

    # --------------------------------------------------------------------
    # Métricas de código
    # --------------------------------------------------------------------
    def code_lengths(self) -> Dict[str, int]:
        """
        Devuelve la longitud (en bits) de cada código.

        Args:
            enc: diccionario {símbolo: código}.

        Returns:
            {símbolo: longitud}.
        """
        return {ch: len(code) for ch, code in self.encoder.items()}

    def avg_code_length(self) -> float:
        """
        Calcula la longitud promedio del código de Huffman.

        Fórmula: L̄ = ∑ p(ch) * |código(ch)|

        Args:
            probs: {símbolo: probabilidad}.
            enc: diccionario {símbolo: código}.

        Returns:
            Longitud promedio en bits/símbolo.
        """
        L = 0.0
        for ch, p in self.probs.items():
            L += p * len(self.encoder[ch])
        return L

    def entropy(self) -> float:
        """
        Calcula la entropía de la fuente.

        Fórmula: H = -∑ p log₂ p

        Args:
            probs: {símbolo: probabilidad}.

        Returns:
            Entropía H en bits/símbolo.
        """
        return -sum(p * log2(p) for p in self.probs.values() if p > 0)


# --------------------------------------------------------------------
# Probabilidades de aparición de símbolos
# --------------------------------------------------------------------
def _symbol_probs(text: str) -> Dict[str, float]:
    """
    Calcula la probabilidad de aparición de cada símbolo en el texto.

    Args:
        text: cadena de entrada.

    Returns:
        Diccionario {símbolo: probabilidad}.
    """
    cnt = Counter(text)            # contar ocurrencias
    n = sum(cnt.values()) or 1     # total de símbolos (evita div. por cero)
    return {ch: c / n for ch, c in cnt.items()}


# --------------------------------------------------------------------
# Construcción del árbol de Huffman (nucleo del algoritmo)
# --------------------------------------------------------------------
def _build_huffman_tree(probs: Dict[str, float]) -> Any:
    """
    Construye el árbol de Huffman a partir de probabilidades.

    Usa un heap de mínimos (cola de prioridad) donde cada nodo es:
        (probabilidad, id_unico, payload)

    - probabilidad: peso acumulado del nodo.
    - id_unico: evita errores al comparar nodos con misma probabilidad.
    - payload: puede ser un símbolo (hoja) o una tupla (subárbol).

    Args:
        probs: diccionario {símbolo: probabilidad}.

    Returns:
        Árbol de Huffman como estructura anidada (hojas=chars, nodos=(izq,der)).
    """
    q = [(p, i, ch) for i, (ch, p) in enumerate(probs.items())]
    if not q:
        q = [(1.0, 0, '\n')]  # caso borde: texto vacío
    heapq.heapify(q)

    uid = len(q)
    while len(q) > 1:
        # extraer los dos nodos menos probables
        p1, _, n1 = heapq.heappop(q)
        p2, _, n2 = heapq.heappop(q)
        # combinarlos en un nuevo nodo interno
        heapq.heappush(q, (p1 + p2, uid, (n1, n2)))
        uid += 1

    return q[0][2]  # raíz del árbol


# --------------------------------------------------------------------
# Recorrido del árbol para asignar códigos binarios
# --------------------------------------------------------------------
def _walk(node: Any, prefix: Tuple[int, ...], out: Dict[str, Code]):
    """
    Recorre el árbol de Huffman en profundidad y asigna códigos binarios.

    Convenciones:
    - Hijo izquierdo → bit 0
    - Hijo derecho   → bit 1

    Args:
        node: nodo actual (símbolo o subárbol).
        prefix: tupla de bits acumulados hasta este nodo.
        out: diccionario de salida {símbolo: código}.
    """
    if isinstance(node, str):                  # caso hoja
        out[node] = prefix or (0,)             # caso borde: un solo símbolo
    else:                                      # caso nodo interno
        left, right = node
        _walk(left, prefix + (0,), out)
        _walk(right, prefix + (1,), out)
