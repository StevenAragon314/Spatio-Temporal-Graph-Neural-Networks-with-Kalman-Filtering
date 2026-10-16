# Pronóstico Espacio-Temporal de Dengue en Costa Rica: De Modelos Estadísticos a Redes Neuronales sobre Grafos y Filtro de Kalman

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-PyTorch%20Geometric-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Un marco de modelado espacio-temporal de extremo a extremo diseñado para pronosticar la incidencia semanal del dengue a nivel cantonal en Costa Rica. Este proyecto implementa una progresión en complejidad de tres niveles, desde líneas base estadísticas multivariadas hasta arquitecturas geométricas y modelos de espacio de estados diferenciables (ST-GNN + Filtro de Kalman) con el propósito de desacoplar la transmisión biológica latente del ruido de medición, el subregistro y los rezagos de reporte.

---

## 📌 Fundamento Teórico y Académico

Este proyecto toma como base e inspiración el trabajo de investigación de **Marianne Peña Wust**, investigadora y estudiante de la **Escuela de Estadística de la Universidad de Costa Rica (UCR)**. Su investigación establece la base metodológica para el planteamiento del problema epidemiológico, las series temporales de dengue a nivel cantonal y la comprensión fundamental de la propagación del vector en el territorio costarricense.

---

## 🔬 Progresión Metodológica de Modelos

Las series temporales epidemiológicas de dengue presentan dos dificultades críticas: **difusión espacial** (movilidad humana y conectividad entre cantones contiguos) y **ruido en la observación** (rezagos administrativos, saturación diagnóstica y subregistro). Para abordar esto con rigor analítico, el repositorio estructura y evalúa tres niveles progresivos:


```

```
              ┌─────────────────────────────────────────────────────────┐
              │ 1. Nivel Base Clásico: SARIMAX Independiente / VAR      │
              │    - Dinámica temporal pura por cantón o bloques        │
              └──────────────────────────┬──────────────────────────────┘
                                         │ [Incorpora topología espacial explícita]
                                         ▼
              ┌─────────────────────────────────────────────────────────┐
              │ 2. Deep Learning Geométrico: ST-GNN (A3T-GCN / STGCN)   │
              │    - Convoluciones sobre matriz de adyacencia (W)       │
              │    - Mecanismos de atención temporal y celdas GRU/LSTM  │
              └──────────────────────────┬──────────────────────────────┘
                                         │ [Desacopla dinámica latente del ruido]
                                         ▼
              ┌─────────────────────────────────────────────────────────┐
              │ 3. Espacio de Estados: ST-GNN + Filtro de Kalman        │
              │    - Propagación del estado latente vía GNN             │
              │    - Filtrado probabilístico de ruido de reporte        │
              └─────────────────────────────────────────────────────────┘

```

```

### 1. Nivel I: Línea Base Clásica (VAR / SARIMAX Independiente)
* **Objetivo:** Establecer la métrica de referencia mínima mediante series temporales tradicionales.
* **Mecánica:** Modela la estacionalidad anual (temporadas lluviosas en el trópico) y rezagos autorregresivos ($t-1, \dots, t-k$).
* **Limitación:** Trata los cantones de forma aislada o asume matrices de covarianza densas que se vuelven inestables al escalar a más de 80 nodos.

### 2. Nivel II: Red Neuronal Espacio-Temporal sobre Grafos (ST-GNN)
* **Objetivo:** Aprender de manera simultánea la autocorrelación temporal y la difusión de contagios entre cantones vecinos.
* **Dimensión espacial:** Capas de convolución sobre grafos ($\text{ChebConv}$ / $\text{GNN}$) que operan sobre una matriz de adyacencia cantonal $W \in \mathbb{R}^{N \times N}$ (contigüidad y distancia entre centroides).
* **Dimensión temporal:** Celdas recurrentes (LSTM) o convoluciones 1D acopladas a mecanismos de atención temporal.
* **Limitación:** Asume que los casos reportados semanales son una verdad determinista, ajustando parámetros al ruido y a las distorsiones de reporte.

### 3. Nivel III: Espacio de Estados Avanzado (ST-GNN + Filtro de Kalman)
* **Objetivo:** Separar la dinámica epidemiológica latente del error aleatorio y administrativo del sistema de vigilancia.
* **Ecuación de estado (dinámica latente):**
  $$x_t = f_{\theta}(x_{t-1}, W) + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
  donde $f_{\theta}$ es la red espacio-temporal parametrizada sobre la topología $W$, y $Q$ es la covarianza del proceso de transmisión.
* **Ecuación de medida (observación con ruido):**
  $$y_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R)$$
  donde $R$ captura la incertidumbre diagnóstica, retrasos en la notificación y subregistro.
* **Inferencia:** Esquema de Filtro de Kalman Diferenciable integrado al bucle de retropropagación de PyTorch, entregando predicciones puntuales e intervalos de credibilidad basados en la covarianza estimada.

---

## 🗂️ Estructura del Repositorio

```bash
├── data/
│   ├── raw/                   
│   ├── spatial/               
│   └── processed/             
├── notebooks/
│   ├── 01_eda_espacial.ipynb  
│   └── 02_evaluacion.ipynb    
├── src/
│   ├── data/
│   │   ├── adyacencia.py      
│   │   └── dataset.py         
│   ├── models/
│   │   ├── baselines.py       
│   │   ├── stgnn.py           
│   │   └── kalman_gnn.py     
│   └── utils/
│       ├── metricas.py       
│       └── visualizador.py   
├── tests/                     
├── environment.yml            
├── requirements.txt           
└── README.md

```

---

## 📜 Citas y Reconocimientos

Si utilizas este código o estructura metodológica en tu trabajo o investigación, por favor cita tanto este repositorio como la investigación base:

```bibtex
@misc{dengue_stgnn_cr_2026,
  author = {Esteven Aragon Portillo},
  title = {Spatio-Temporal Graph Neural Networks with Kalman Filtering for Dengue Surveillance in Costa Rica},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{[https://github.com/tu-usuario/dengue-stgnn-costarica](https://github.com/tu-usuario/dengue-stgnn-costarica)}}
}

@unpublished{pena_wust_dengue_ucr,
  author = {Peña Wust, Marianne},
  title = {Estudio y Modelado Estadístico de la Dinámica Epidemiológica del Dengue en Costa Rica},
  school = {Escuela de Estadística, Universidad de Costa Rica (UCR)},
  note = {Fundamentación epidemiológica y formulación de datos}
}
