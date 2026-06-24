# ⚽ Football Predictor

App Streamlit mobile-first para predecir resultados y goles de partidos entre selecciones nacionales.

## Fuentes de datos

| Fuente | Datos | Estado |
|--------|-------|--------|
| [martj42/international_results](https://github.com/martj42/international_results) | 49,000+ partidos históricos (1872–hoy) | ✅ Auto-descarga |
| Feature engineering propio | Forma reciente, H2H, contexto | ✅ Incluido |
| [Kaggle – FIFA World Ranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Rankings FIFA históricos | ⚠️ Manual |
| API-Football | xG, posesión, estadísticas avanzadas | 📋 Opcional |

## Setup local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Añadir rankings FIFA (opcional pero recomendado)

1. Descarga el dataset de Kaggle: https://www.kaggle.com/datasets/cashncarry/fifaworldranking
2. Coloca el CSV en `data/raw/fifa_ranking.csv`
3. El CSV debe tener columnas: `rank_date`, `country_full`, `rank`
4. Reinicia la app — se activarán automáticamente las features de ranking

## Deploy gratuito en Streamlit Community Cloud

1. Sube el proyecto a un repositorio GitHub
2. Entra a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona el repo y `app.py` como archivo principal
5. Click en **Deploy** → URL pública en ~2 minutos

La app es completamente funcional desde el celular (diseño mobile-first).

## Estructura del proyecto

```
football_predictor/
├── app.py                    # App Streamlit principal
├── requirements.txt
├── src/
│   ├── fetch_historical.py   # Fuente 1: historial de partidos
│   ├── fetch_rankings.py     # Fuente 4: rankings FIFA
│   ├── feature_pipeline.py   # Construcción de features (forma, H2H, contexto)
│   └── model.py              # Modelo predictivo (resultado + goles)
└── data/
    ├── raw/                  # Datos descargados
    └── processed/            # Features procesadas (cache)
```

## Features del modelo

- **Forma reciente**: puntos, GF, GA promedio últimos 5 partidos
- **H2H**: historial directo entre los dos equipos
- **Contexto**: peso de competencia, ventaja de localía
- **Rankings FIFA**: diferencia de ranking (si se provee el CSV)

## Modelo

- Resultado (H/D/A): RandomForestClassifier
- Goles totales: RandomForestRegressor
- Split temporal (no aleatorio) para evitar data leakage


## Datos avanzados de StatsBomb (xG)

El modelo usa un enfoque **híbrido** que mejora la precisión en torneos de selecciones:

- **Modelo base**: forma, H2H, contexto — para todos los partidos
- **Modelo élite**: añade xG real de StatsBomb — cuando ambos equipos tienen datos

### Torneos de élite incluidos (gratis, StatsBomb Open Data)
- FIFA World Cup 2022 y 2018
- UEFA Euro 2024 y 2020
- Copa América 2024
- African Cup of Nations 2023

### Precisión (test out-of-sample 2023-2025)
| | Modelo base | Modelo híbrido |
|---|---|---|
| Total | 52.7% | **54.7%** |
| Mundiales/EURO/Copa América | 45.7% | **54.8%** |

### Regenerar el resumen StatsBomb
Si StatsBomb publica nuevos torneos:
```bash
python src/build_statsbomb_summary.py
```
Esto descarga los eventos, calcula xG por selección y guarda un resumen ligero
en `data/processed/statsbomb_team_stats.parquet` (no guarda los eventos crudos).


---

## Actualización: Rating Elo (mejora principal)

Tras una auditoría que detectó **data leakage** en el xG, se rediseñó el enfoque:

### Qué cambió
- **Añadido rating Elo propio**, calculado desde 32,000 partidos (metodología World Football Elo, la que adoptó FIFA en 2018). Es el predictor #1: aporta el 55% de la decisión del modelo.
- **Arreglado el leakage del xG**: ahora se calcula con ventana temporal correcta.
- **Descartado el xG como feature predictiva**: el Elo lo supera por mejor cobertura (100% de selecciones vs 6 torneos) y sin leakage. El xG se conserva solo para visualización.

### Precisión honesta (test out-of-sample 2023-2025, sin leakage)
| Configuración | Total | Mundiales/EURO/CA |
|---|---|---|
| Original (forma + H2H) | 52.8% | 45.7% |
| **+ Elo (modelo final)** | **56.9%** | **49.5%** |

El Elo recalcula su valor cada 24h con los partidos nuevos. La predicción de resultado 1X2 en fútbol tiene un techo real de 52-56% según la literatura académica; este modelo está en el rango alto.

### Nueva pestaña: Ranking Elo
Ranking mundial de selecciones por fuerza, con buscador.


---

## Actualización: motor XGBoost

Se reemplazó RandomForest por **XGBoost** (gradient boosting), el estándar de la industria.

### Validación walk-forward (robusta, multi-año)
| Año test | RandomForest | XGBoost |
|---|---|---|
| 2019 | 57.3% | 58.2% |
| 2021 | 60.6% | 61.6% |
| 2022 | 51.8% | 54.2% |
| 2023 | 56.4% | 59.9% |
| 2024 | 55.2% | 57.7% |
| 2025 | 59.5% | 61.7% |
| **Promedio** | **56.8%** | **58.9%** |

XGBoost ganó en los 6/6 años — la mejora es consistente, no suerte de un split.

### Precisión final del modelo (out-of-sample, sin leakage)
- **Total: 59.4%**
- **Mundiales/EURO/Copa América: 56.9%** (vs 45.7% del modelo original)
- Goles totales MAE: ±1.38

El recorrido completo: 45.7% → 49.5% (Elo) → 56.9% (XGBoost) en alta competencia. +11 puntos reales y honestos sobre el punto de partida.
