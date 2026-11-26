# Equipo1_PI_InteligenciaArtificial

Proyecto integrador de la EE **Inteligencia Artificial**, Licenciatura en Ingeniería de Software,
Universidad Veracruzana, Facultad de Contaduría y Administración, región Coatzacoalcos-Minatitlán.
Docente: Ángel Yram González Bartolo.

## Integrantes

| Nombre | Matrícula |
|---|---|
| ADRIANO COBO KING BRIAN | S23028345 |
| BELTRAN SANTOS RAFAEL ALEJANDRO | S23017374 |
| FRANCO TOLEDO YAEL | S23017346 |
| GUTIERREZ SANTIAGO JESICA MARISSA | S23017365 |
| LEON CRUZ JOSE EMILIO | S23017360 |
| SANCHEZ SAUCEDO RENE EMILIANO | S23017381 |

## Descripción del proyecto

**Caso de estudio:** elegir qué modelos forman un *ensemble* para detectar posibles fallas
de maquinaria a partir de cuatro lecturas de sensores: temperatura, vibración, presión y
corriente.

- Se entrenan seis clasificadores, todos programados en Python puro: regresión logística,
  árbol de decisión, Random Forest, KNN, SVM lineal y una red neuronal pequeña.
- Los modelos se entrenan con 600 lecturas **sintéticas** (semilla 2026), divididas en
  60% entrenamiento, 20% validación y 20% prueba.
- Cada solución es un vector de 6 bits: el bit *i* vale 1 si el modelo *Mi* participa en
  la votación por mayoría.
- Las soluciones se evalúan con este costo:

  `Costo(E) = 1.0 × (1 − accuracy de validación) + 0.08 × (modelos activos / 6)`

| Módulo | Contenido |
|---|---|
| 1 | Escenario: generación de datos, partición y entrenamiento de los 6 modelos |
| 2 | Hill Climbing |
| 3 | Simulated Annealing |
| 4 | Comparación HC vs SA, verificación exhaustiva y consistencia |
| 5 | Algoritmo genético: inicialización de la población |
| 6 | Algoritmo genético: selección de padres por torneo (k = 3) |
| 7 | Algoritmo genético: cruce de un punto y mutación por inversión de bit |
| 8 | Algoritmo genético: ciclo generacional completo con reemplazo de los peores |
| 9 | Comparación global definitiva (HC, SA y AG) |
| 10 | CSP: variables, dominios, restricciones unaria, binaria y global, y factibilidad de las soluciones de HC, SA y AG |
| 11 | Clasificación con discriminante lineal: _pendiente_ |
| 12 | Agrupamiento con K-medias: _pendiente_ |

## Estructura

```
Equipo1_PI_InteligenciaArtificial/
├── src/
│   ├── proyecto.py      # programa principal: menú y módulos 1-10
│   ├── escenario.py     # datos sintéticos, partición, votación y función de costo
│   ├── modelos.py       # los 6 clasificadores
│   ├── busqueda.py      # módulos 2 y 3: Hill Climbing y Simulated Annealing
│   ├── genetico.py      # módulos 5-8: algoritmo genético
│   ├── csp.py           # módulo 10: CSP y verificación de restricciones
│   └── generar_csv.py   # regenera los archivos de data/
├── data/
│   ├── discriminante_entrenamiento.csv   # 10 datos por clase (módulo 11)
│   ├── discriminante_casos_nuevos.csv    # casos nuevos a clasificar (módulo 11)
│   └── kmedias_maquinas.csv              # 24 máquinas para agrupar (módulo 12)
├── docs/                # documento final en PDF
└── README.md
```

## Requisitos

- Python **3.10 o superior**.
- No necesita bibliotecas externas: solo usa la biblioteca estándar.

## Ejecución

Desde la carpeta raíz del repositorio:

```bash
python3 src/proyecto.py
```

Se abre el menú; escribe el número del módulo (1 a 10) o `0` para salir. Los módulos se
pueden ejecutar en cualquier orden: si uno necesita resultados de otro (por ejemplo, el 10
necesita las soluciones de HC, SA y AG), el programa los calcula primero y lo indica.

Para ejecutar todos los módulos seguidos, sin menú:

```bash
python3 src/proyecto.py --demo
```

Opciones adicionales:

| Opción | Uso |
|---|---|
| `--semilla-sa N` | Semilla de Simulated Annealing (predeterminada: 7) |
| `--inicio 110011` | Estado inicial de HC y SA |
| `--detallado` | Imprime todas las propuestas de SA |
| `--repeticiones N` | Corridas de la prueba de consistencia del módulo 4 (predeterminado: 20) |
| `--salida ruta.json` | Archivo con las trazas del módulo 4 (predeterminado: `resultados/resultados.json`) |

Para regenerar los archivos de `data/`:

```bash
python3 src/generar_csv.py
```

> Los datos son sintéticos: simulan lecturas de sensores con fines académicos y no
> representan mediciones industriales reales.
