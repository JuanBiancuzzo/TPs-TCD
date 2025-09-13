from collections import Counter
from math import log2
from typing import Dict, List, Tuple, Any
import heapq

Bit = int  # Representa un bit (0 o 1)
Code = Tuple[Bit, ...]  # Código binario como tupla de bits (prefijo libre)
EncDict = Dict[str, Code]  # Diccionario: símbolo -> código
DecDict = Dict[Code, str]  # Diccionario: código -> símbolo


# --------------------------------------------------------------------
# 1. Probabilidades de aparición de símbolos
# --------------------------------------------------------------------
def symbol_probs(text: str) -> Dict[str, float]:
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
# 2. Construcción del árbol de Huffman (nucleo del algoritmo)
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
# 3. Recorrido del árbol para asignar códigos binarios
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


# --------------------------------------------------------------------
# 4. Construcción de diccionarios de codificación/decodificación
# --------------------------------------------------------------------
def build_huffman_dict(text: str) -> Tuple[EncDict, DecDict]:
    """
    Genera los diccionarios de codificación y decodificación de Huffman.

    Args:
        text: cadena de entrada.

    Returns:
        (enc, dec):
        - enc: {símbolo: código}
        - dec: {código: símbolo}
    """
    probs = symbol_probs(text)
    tree = _build_huffman_tree(probs)
    enc: EncDict = {}
    _walk(tree, tuple(), enc)
    dec: DecDict = {code: ch for ch, code in enc.items()}
    return enc, dec


# --------------------------------------------------------------------
# 5. Codificación de texto a bits
# --------------------------------------------------------------------
def encode_text(text: str, enc: EncDict) -> List[Bit]:
    """
    Codifica un texto a una secuencia de bits usando un diccionario de Huffman.

    Args:
        text: cadena de entrada.
        enc: diccionario {símbolo: código}.

    Returns:
        Lista de bits (0/1).
    """
    bits: List[Bit] = []
    for ch in text:
        bits.extend(enc[ch])  # concatena el código del símbolo
    return bits


# --------------------------------------------------------------------
# 6. Decodificación de bits a texto
# --------------------------------------------------------------------
def decode_bits(bits: List[Bit], dec: DecDict) -> str:
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
    dec_keys = set(dec.keys())
    lens = sorted({len(k) for k in dec_keys})  # longitudes posibles de códigos
    maxlen = max(lens) if lens else 1
    as_tuple = tuple  # alias para optimizar

    for b in bits:
        buf.append(b)
        if len(buf) > maxlen:    # seguridad: si algo va mal, limpiar
            buf.clear()
            continue
        t = as_tuple(buf)
        if t in dec:             # si el buffer matchea un código
            out_chars.append(dec[t])
            buf.clear()

    # Si quedan bits sin cerrar, se ignoran (robustez ante ruido)
    return "".join(out_chars)


# --------------------------------------------------------------------
# 7. Métricas de código
# --------------------------------------------------------------------
def code_lengths(enc: EncDict) -> Dict[str, int]:
    """
    Devuelve la longitud (en bits) de cada código.

    Args:
        enc: diccionario {símbolo: código}.

    Returns:
        {símbolo: longitud}.
    """
    return {ch: len(code) for ch, code in enc.items()}

def avg_code_length(probs: Dict[str, float], enc: EncDict) -> float:
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
    for ch, p in probs.items():
        L += p * len(enc[ch])
    return L

def entropy(probs: Dict[str, float]) -> float:
    """
    Calcula la entropía de la fuente.

    Fórmula: H = -∑ p log₂ p

    Args:
        probs: {símbolo: probabilidad}.

    Returns:
        Entropía H en bits/símbolo.
    """
    return -sum(p * log2(p) for p in probs.values() if p > 0)
