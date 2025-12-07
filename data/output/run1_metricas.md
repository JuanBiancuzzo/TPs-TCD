### Resultados Fuente/Huffman
- Entropía H: **4.2359** bits/símbolo
- Longitud mínima (Lmin): **3** bits
- Longitud promedio (L̄): **4.2674** bits/símbolo
- Eficiencia (η = H/L̄): **99.26%**
- Código de longitud fija (p.ej. ASCII para k símbolos): **6** bits/símbolo
- Alfabeto (k): **42** símbolos

---

#### Tabla CSV generada
- `data\output\run1_tabla_huffman.csv`

#### Muestra durante todo el proceso

- Línea:        `Esto es una linea de prueba para ver el resultado obtenido despues de todo este proceso.`
- Decodificada: `Esto es una linea de prueba para rae nersenbMeut obtenido despues de todo este proceso.`


### Resultados Codificación
- Valor de dmin: 4
- Valor de e: 3
- Valor de t: 1

- Matriz G: 
 1 0 0 0 0 1 1 0 1 1 0 1 0 0 1 
 0 1 0 0 0 1 0 1 0 0 1 1 1 1 0 
 0 0 1 0 0 0 1 1 1 1 0 0 1 1 1 
 0 0 0 1 0 1 1 0 0 1 1 1 1 0 0 
 0 0 0 0 1 1 0 1 1 0 0 1 1 1 1 

- Matriz H: 
 1 1 0 1 1 1 0 0 0 0 0 0 0 0 0 
 1 0 1 1 0 0 1 0 0 0 0 0 0 0 0 
 0 1 1 0 1 0 0 1 0 0 0 0 0 0 0 
 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0 
 1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 
 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 
 1 1 0 1 1 0 0 0 0 0 0 1 0 0 0 
 0 1 1 1 1 0 0 0 0 0 0 0 1 0 0 
 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 
 1 0 1 0 1 0 0 0 0 0 0 0 0 0 1 

#### Tabla de simbolos
|              e                |         e H^T       |
|-------------------------------|---------------------|
| 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 | 1 0 0 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 | 0 1 0 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 | 0 0 1 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 | 0 0 0 1 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 | 0 0 0 0 1 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 | 0 0 0 0 0 1 0 0 0 0 |
| 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 | 0 0 0 0 0 0 1 0 0 0 |
| 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 1 0 0 |
| 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 1 0 |
| 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0 1 |
| 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 | 1 1 1 1 0 0 1 1 0 1 |
| 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 | 0 0 1 1 1 1 0 0 1 1 |
| 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 | 1 1 1 0 0 1 1 1 1 0 |
| 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0 1 1 1 1 0 0 1 0 1 |
| 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 1 0 0 1 0 1 1 0 1 1 |
| 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 | 1 1 0 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 | 1 0 1 0 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 | 1 0 0 1 0 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 | 1 0 0 0 1 0 0 0 0 0 |
| 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 | 1 0 0 0 0 1 0 0 0 0 |
...
| 1 0 0 1 0 1 1 0 0 0 0 0 0 1 0 | 1 1 1 0 1 0 1 0 1 1 |
| 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 | 0 0 0 1 1 1 1 1 0 0 |
| 0 0 0 0 0 0 1 1 1 0 1 1 0 0 0 | 0 0 0 1 1 0 1 1 1 0 |
| 0 0 0 0 0 1 0 1 1 1 1 0 0 0 0 | 0 0 0 0 1 1 1 1 0 1 |
| 0 0 0 0 0 1 1 1 1 0 1 0 0 0 0 | 0 0 0 0 1 0 1 1 1 1 |

#### Muestra (codificación → canal → decodificación)
- Secuencia inicial: [0 1 0 1 0 1 1 1 1 1 0 1 0 0 0]
- m * G: [0 1 0 1 0 0 1 1 0 1 0 0 0 1 0 1 1 1 1 1 0 1 1 1 1 0 0 0 1 1 0 1 0 0 0 1 0
 1 0 0 1 1 1 1 0]
- Recibido: [1 1 0 1 0 0 1 1 0 1 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0
 1 0 0 1 1 1 0 0]
- e * H^T: 
	[1 1 0 1 1 0 1 0 0 0]
	[1 1 0 0 1 1 1 1 0 0]
	[1 1 0 0 1 1 1 1 1 0]
- e: 
	[1 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
	[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0]
	[0 0 0 1 0 0 0 0 0 0 0 0 0 1 0]
- Decodificador: [0 1 0 1 0 0 1 1 0 1 0 0 0 1 0]
	[1 1 1 1 1 0 1 1 1 1 0 0 0 1 1]
	[0 1 0 0 0 1 0 1 0 0 1 1 1 1 0]


### Resultados Modulación
- Energía media de símbolo: **1.000** Joule
- Energía media de bit: **0.333** Joule
- Probabilidad de error de símbolo: **0.2800**
- Probabilidad de error de bit: **0.0960**

#### Gráfico de constelación generado
- `data/output/run1_Contelacion_8-PSK.png`


### Resultados Canal

#### Gráfico de constelación generado con datos
- `data/output/run1_Contelacion_8-PSK_con_datos.png`

