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

3. Para ver cuales son los argumentos de ejecución disponible:
    ```bash
    python src/main.py -h
    ```

    ```
    options:
    -h, --help            show this help message and exit
    --in PATH_IN          ruta al .txt de entrada (default: data/input/texto.txt)
    --out-prefix OUT_PREFIX
                            prefijo de salida (default: data/output/run1)
    --ebn0 EBN0           Eb/N0 en dB (para canal)
    --dry-run             no procesa: copia el texto tal cual
    --huffman-only        ejecuta solo Módulo B (cod/dec de fuente)
    ```
4. Ejecutar el trabajo entero:
    ```
    python src/main.py
    ```

    Esto ejecuta:
    - Lectura del archivo (file.py)
    - Codificación Huffman (source.py)
    - Codificación de canal (cod_channel.py)
    - Modulación (modulation.py)
    - Canal con ruido (channel.py)
    - Demodulación (automática)
    - Decodificación de canal (automática)
    - Decodificación Huffman (automática)
    - Escritura del resultado (automática)

5. Donde ver los resultados
    Los resultados se guardan en `data/output/` con el prefijo que indiques (por defecto `run1`).

    - data/output/run1_recibido.txt (texto final)
    - data/output/run1_metricas.md (métricas: entropia H, longitud promedio de codigo, eficiencia, probabilidad de error y energias de simbolo y bit)
    - data/output/run1_tabla_huffman.csv (tabla de códigos huffman, Columnas: char, prob, code, len, ascii)
    - data/output/run1_Contelacion_*.png (gráficos de constelacion). Ejemplos:
        - run1_Contelacion_8-PSK.png: constelación teórica
        - run1_Contelacion_8-PSK_con_datos.png: constelación con datos recibidos

    Ejemplo de salida en terminal:
    ```
    [File] Leyendo archivo: data/input/texto.txt
    [Fuente/Huffman] Construyendo código Huffman
    [Fuente/Huffman] H=4.2359 | Lavg=4.2674 | η=99.26% | L_fijo≈6
    [Codificación] Creando códigos de lineas
    [Modulación] Mapeando de 8-PSK con 3 bits a símbolos
    [Canal] Aplicando AWGN/atenuación
    [Canal] Eb/N0=6.0 dB -> sigma = 0.316
    [Salida] Texto recibido -> data/output/run1_recibido.txt
    ```

## Análisis del Sistema (Módulo F)

### 6. Ejecutar análisis del sistema

Para realizar el análisis completo del rendimiento del sistema (BER/SER vs Eb/N0):

```bash
python src/main.py --analyze-system
```

Esto ejecuta:
- Análisis sin codificación de canal para todas las combinaciones (PSK/FSK, M=2,4,8,16)
- Análisis con codificación de canal para PSK M=8
- Generación de gráficos comparativos
- Guardado de resultados en `data/analysis/analysis_results.csv`

**Nota:** Este proceso puede tomar varios minutos ya que realiza múltiples simulaciones.

### 7. Generar reporte de análisis

Para generar el reporte markdown con todos los gráficos y tablas resumen:

```bash
python generate_report.py
```

Esto genera:
- Gráficos faltantes de comparación PSK vs FSK (M=2, 4, 8, 16)
- Reporte markdown completo en `data/analysis/REPORTE_ANALISIS_SISTEMA.md`

El reporte incluye:
- Gráficos de Pe y Pb vs Eb/N0 para PSK y FSK
- Comparaciones PSK vs FSK para diferentes valores de M
- Comparación con/sin codificación de canal
- Tablas resumen con valores calculados dinámicamente desde los datos
