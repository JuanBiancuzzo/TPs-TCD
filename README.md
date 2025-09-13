# TP TA137 – Pipeline base (Python, sin notebooks)

Este repo es un **esqueleto simple** para ir completando módulo por módulo.

## Estructura
```
tp-ta137/
├── README.md
├── requirements.txt
├── data/
│   ├── input/            # poner acá el .txt de entrada
│   └── output/           # resultados: texto recibido + figuras
└── src/
    ├── main.py           # orquestador CLI (pipeline)
    ├── source.py         # Módulo B (Huffman) – TODO
    ├── modulation.py     # Módulo C (modulaciones) – TODO
    ├── channel.py        # Módulo D (AWGN, atenuación) – TODO
    ├── cod_channel.py    # Módulo E - TODO    
    ├── viz.py            # Gráficos – TODO
    └── utils.py          # utilidades mínimas (BER, RNG, etc.)
```

## Uso rápido
1. Crear venv e instalar dependencias
    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Copiar un archivo en `data/input/texto.txt` (crear si no existe).

3. Ejecutar el pipeline en modo **dry-run** (pasa el archivo “tal cual” para probar estructura):
    ```bash
    python src/main.py --dry-run
    ```
    Genera:
    - `data/output/run1_recibido.txt` (copia del input, por ahora)
    - logs de cada etapa indicando **TODOs**.

4. Para ver cuales son los argumentos de ejecución disponible:
    ```bash
    python src/main.py -h
    ```
