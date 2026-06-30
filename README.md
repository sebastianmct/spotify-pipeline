# Spotify Tracks — Pipeline de Machine Learning

Pipeline de datos reproducible que ingesta, limpia, valida, entrena un modelo de clasificación y carga resultados del dataset de canciones de Spotify en MySQL. Todos los servicios de infraestructura corren en contenedores Docker.

---

## Dataset

**Fuente**: Spotify Web API / Kaggle Spotify Tracks Dataset

**Archivo**: `data/external/spotify_tracks.csv` — 5,000 registros con 20 columnas de características de audio y metadatos de canciones.

| Columna original | Descripción | Tipo |
|---|---|---|
| `track_id` | Identificador único de la canción | String |
| `artists` | Nombre del artista o banda | String |
| `album_name` | Nombre del álbum | String |
| `track_name` | Nombre de la canción | String |
| `popularity` | Popularidad (0–100) calculada por Spotify | Integer |
| `duration_ms` | Duración en milisegundos | Integer |
| `explicit` | Indica si la canción tiene contenido explícito | Boolean |
| `danceability` | Qué tan bailable es la canción (0.0–1.0) | Float |
| `energy` | Medida de intensidad y actividad (0.0–1.0) | Float |
| `key` | Tonalidad musical (0=C, 1=C#, ..., 11=B) | Integer |
| `loudness` | Volumen general en decibelios (dB) | Float |
| `mode` | Modalidad (0=menor, 1=mayor) | Integer |
| `speechiness` | Presencia de palabras habladas (0.0–1.0) | Float |
| `acousticness` | Probabilidad de ser acústica (0.0–1.0) | Float |
| `instrumentalness` | Probabilidad de ser instrumental (0.0–1.0) | Float |
| `liveness` | Probabilidad de ser grabada en vivo (0.0–1.0) | Float |
| `valence` | Positividad musical (0.0=negativo, 1.0=positivo) | Float |
| `tempo` | Tempo estimado en BPM | Float |
| `time_signature` | Compás musical (3–7) | Integer |
| `track_genre` | Género musical asignado | String |

**Tamaño aproximado**: 0.73 MB en bruto, columnas mixtas (numéricas + categóricas).

---

## Arquitectura del Pipeline

```
data/external/
  └── spotify_tracks.csv
        │
        ├─ 01_ingesta.py       (Stage 1)
        │
        ▼
data/raw/
  └── spotify_tracks_raw.csv
        │
        ├─ 02_limpieza.py      (Stage 2)
        │
        ▼
data/processed/
  └── spotify_tracks_clean.csv
        │
        ├─ 03_validacion.py    (Stage 3)
        │
        ▼
data/validated/
  ├── spotify_tracks_validated.csv
  └── spotify_tracks_rejected.csv
        │
data/reports/
  ├── validation_report_YYYYMMDD_HHMMSS.json
  └── cleaning_report_YYYYMMDD_HHMMSS.json
        │
        ├─ 04_carga.py         (Stage 4)
        │
        ▼
data/model/
  └── spotify_popularity_classifier.pkl

data/reports/
  ├── classification_report_YYYYMMDD_HHMMSS.json
  ├── classification_report_YYYYMMDD_HHMMSS.txt
  ├── confusion_matrix_YYYYMMDD_HHMMSS.png
  └── metrics_YYYYMMDD_HHMMSS.json

data/dashboard/
  └── spotify_dashboard_data.csv

MySQL (spotify_ml_db)
  ├── spotify_tracks
  ├── model_metrics
  └── predictions

logs/
  ├── 01_ingesta_YYYYMMDD_HHMMSS.log
  ├── 02_limpieza_YYYYMMDD_HHMMSS.log
  ├── 03_validacion_YYYYMMDD_HHMMSS.log
  └── 04_carga_YYYYMMDD_HHMMSS.log
```

### Infraestructura Docker

```
pipeline_network (bridge)
  ├── mysql              — MySQL 8.0, puerto 3306
  ├── phpmyadmin         — phpMyAdmin latest, puerto 8080
  └── pipeline           — contenedor Python que ejecuta scripts
```

---

## Requisitos previos

### Opción A: Ejecución con Docker (recomendado)

- Docker Engine 20.10+
- Docker Compose 2.0+
- 2GB RAM disponible
- 500MB espacio en disco

### Opción B: Ejecución local (requiere más configuración)

- Python 3.10+
- MySQL Server 8.0+ instalado localmente (opcional)
- Dependencias Python (ver `requirements.txt`)

---

## Puesta en marcha

### 1. Clonar y configurar variables de entorno

```bash
git clone https://github.com/sebastianmct/spotify-pipeline.git
cd spotify-pipeline
cp .env.example .env
```

El archivo `.env` incluye valores por defecto:

```
SPOTIFY_DATASET_PATH=data/external/spotify_tracks.csv
SAMPLE_SIZE=5000
TEST_SIZE=0.2
RANDOM_STATE=42
N_ESTIMATORS=200
DB_HOST=mysql
DB_PORT=3306
DB_USER=pipeline_user
DB_PASSWORD=pipelinepass
DB_NAME=spotify_ml_db
```

**Nota**: En Docker, `DB_HOST` debe ser `mysql` (nombre del servicio). Si ejecutas localmente, usa `localhost`.

### 2. Levantar la infraestructura

```bash
docker-compose up -d
```

Verificar que todos los servicios estén corriendo:

```bash
docker-compose ps
```

Esperar a que MySQL esté completamente listo (20-30 segundos):

```bash
docker-compose logs mysql | grep "ready for connections"
```

### 3. Ejecutar el pipeline

El pipeline se ejecuta automáticamente al iniciar el contenedor. Para ejecutar manualmente o re-ejecutar:

```bash
# Ejecutar todas las etapas secuencialmente
docker-compose exec pipeline python scripts/01_ingesta.py
docker-compose exec pipeline python scripts/02_limpieza.py
docker-compose exec pipeline python scripts/03_validacion.py
docker-compose exec pipeline python scripts/04_carga.py

# O usar el script orquestador
docker-compose exec pipeline python scripts/run_pipeline.py
```

### 4. Monitorear el progreso

Ver logs en tiempo real:

```bash
docker-compose logs -f pipeline
```

Ver logs históricos de una etapa específica:

```bash
tail -f logs/04_carga_*.log
```

---

## Etapas del Pipeline

### Stage 1: Ingesta (`scripts/01_ingesta.py`)

**Objetivo**: Cargar el dataset crudo de canciones de Spotify y guardarlo en formato estandarizado.

**Funcionalidades**:
- Carga dataset desde archivo CSV en `data/external/spotify_tracks.csv`
- Filtra solo las columnas esperadas del esquema
- Valida estructura mínima (20 columnas requeridas)
- Si el archivo no existe, genera un dataset sintético de muestra
- Guarda en `data/raw/spotify_tracks_raw.csv`

**Salida**:
- Archivo: `data/raw/spotify_tracks_raw.csv`
- Registros: 5,000 (por defecto, configurable vía `SAMPLE_SIZE`)
- Log: `logs/01_ingesta_YYYYMMDD_HHMMSS.log`

---

### Stage 2: Limpieza y Transformación (`scripts/02_limpieza.py`)

**Objetivo**: Limpiar datos crudos, normalizar rangos de audio, optimizar tipos de datos y crear variables derivadas para el modelo.

**Transformaciones aplicadas**:

| Transformación | Detalle |
|---|---|
| Eliminar duplicados | Por `track_id` |
| Manejo de valores nulos | `drop` en campos críticos, `fill_unknown` en texto, `fill_median` en numéricas, `fill_mode` en categóricas |
| Normalizar texto | Trim de espacios, lower case en `track_genre` |
| Validar rangos de audio | Clip `danceability`–`valence` a [0,1]; `loudness` a [-60, 5]; reemplazar `tempo` <= 0 con mediana |
| Crear columna derivada | `duration_min` = `duration_ms / 60000` |
| Crear target categórico | `popularity_category`: Baja ([0–33]), Media ([34–66]), Alta ([67–100]) |
| Agrupar géneros raros | `track_genre_grouped`: géneros con < 0.5% de frecuencia se agrupan como `other` |
| Optimizar tipos de datos | Conversión a tipos enteros y flotantes más pequeños; categóricas a `category` |

**Salida**:
- Archivo: `data/processed/spotify_tracks_clean.csv`
- Log: `logs/02_limpieza_YYYYMMDD_HHMMSS.log`
- Reporte: `data/reports/cleaning_report_YYYYMMDD_HHMMSS.json`

---

### Stage 3: Validación Estructural y Semántica (`scripts/03_validacion.py`)

**Objetivo**: Validar integridad y coherencia de datos. Separar registros válidos de rechazados.

**Validaciones estructurales**:
- Presencia de columnas obligatorias (21 columnas del esquema)
- Tipos de datos correctos por columna
- Unicidad de `track_id`
- No nulidad de campos críticos (`track_id`, `track_name`, `popularity`, `track_genre`, `popularity_category`)

**Validaciones semánticas**:
- Características de audio acotadas a [0,1] (`danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`)
- Rangos numéricos validados (`popularity` [0,100], `loudness` [-60,5], `tempo` [0,250], `duration_ms` [1000, 1,800,000], `key` [0,11], `mode` [0,1], `time_signature` [1,7])
- Valores categóricos válidos (`explicit` True/False, `popularity_category` Baja/Media/Alta)
- Coherencia: `popularity_category` debe coincidir con los bins de `popularity`

**Detección de anomalías**:
- Géneros raros (< 0.1% de frecuencia)
- Outliers en `duration_ms` (IQR > 1.5)
- Valores extremos de `loudness` (> 0 dB)

**Salida**:
- Válidos: `data/validated/spotify_tracks_validated.csv`
- Rechazados: `data/validated/spotify_tracks_rejected.csv`
- Reporte: `data/reports/validation_report_YYYYMMDD_HHMMSS.json`
- Log: `logs/03_validacion_YYYYMMDD_HHMMSS.log`

---

### Stage 4: Entrenamiento, Evaluación y Carga (`scripts/04_carga.py`)

**Objetivo**: Entrenar un clasificador Random Forest para predecir la categoría de popularidad, evaluar su rendimiento y cargar resultados a MySQL.

#### Modelo entrenado

**Algoritmo**: Random Forest Classifier

**Parámetros**:
| Parámetro | Valor |
|---|---|
| `n_estimators` | 200 (configurable vía `N_ESTIMATORS`) |
| `max_depth` | Sin límite |
| `min_samples_split` | 5 |
| `class_weight` | `balanced` |
| `random_state` | 42 (configurable vía `RANDOM_STATE`) |
| `n_jobs` | -1 (usa todos los cores) |
| `test_size` | 0.2 (80% entrenamiento, 20% prueba) |

**Features utilizadas (15 total)**:
- Numéricas (13): `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_min`, `key`, `mode`, `time_signature`
- Categóricas codificadas (2): `explicit`, `track_genre_grouped`

**Target**: `popularity_category` (Baja, Media, Alta) — clasificación multiclase.

#### Estructura de tablas en MySQL

**Tabla `spotify_tracks`**:
```sql
CREATE TABLE spotify_tracks (
    track_id              VARCHAR(50)    NOT NULL PRIMARY KEY,
    artists               VARCHAR(500),
    album_name            VARCHAR(500),
    track_name            VARCHAR(500),
    popularity            SMALLINT,
    duration_ms           INT,
    duration_min          FLOAT,
    explicit              TINYINT(1)     DEFAULT 0,
    danceability          FLOAT,
    energy                FLOAT,
    `key`                 TINYINT,
    loudness              FLOAT,
    mode                  TINYINT,
    speechiness           FLOAT,
    acousticness          FLOAT,
    instrumentalness      FLOAT,
    liveness              FLOAT,
    valence               FLOAT,
    tempo                 FLOAT,
    time_signature        TINYINT,
    track_genre           VARCHAR(50),
    track_genre_grouped   VARCHAR(50),
    popularity_category   VARCHAR(10),

    INDEX idx_popularity      (popularity),
    INDEX idx_genre           (track_genre),
    INDEX idx_pop_category    (popularity_category),
    INDEX idx_explicit        (explicit),
    INDEX idx_danceability    (danceability),
    INDEX idx_energy          (energy)
);
```

**Tablas de auditoría**:
- `model_metrics`: Métricas de rendimiento y parámetros de cada entrenamiento
- `predictions`: Predicciones individuales por canción con nivel de confianza

#### Artefactos generados (locales)

| Artefacto | Ruta |
|---|---|
| Modelo serializado | `data/model/spotify_popularity_classifier.pkl` |
| Matriz de confusión | `data/reports/confusion_matrix_YYYYMMDD_HHMMSS.png` |
| Classification report (texto) | `data/reports/classification_report_YYYYMMDD_HHMMSS.txt` |
| Classification report (JSON) | `data/reports/classification_report_YYYYMMDD_HHMMSS.json` |
| Métricas (JSON) | `data/reports/metrics_YYYYMMDD_HHMMSS.json` |
| Datos para dashboard | `data/dashboard/spotify_dashboard_data.csv` |

**Salida**:
- Log: `logs/04_carga_YYYYMMDD_HHMMSS.log`
- Accuracy: ~0.34 (modelo base, por debajo del umbral esperado de 0.60)

---

## Monitoreo y Acceso a Datos

### phpMyAdmin

Acceder en [http://localhost:8080](http://localhost:8080)

**Credenciales**:
- Usuario: `pipeline_user`
- Contraseña: `pipelinepass`
- Host: `mysql` (automático en Docker)

**Pasos para conectar**:
1. Ir a http://localhost:8080
2. Iniciar sesión con las credenciales anteriores
3. Base de datos: `spotify_ml_db`
4. Tablas disponibles: `spotify_tracks`, `model_metrics`, `predictions`

### Consultas útiles

```sql
-- Ver total de canciones cargadas
SELECT COUNT(*) as total_canciones FROM spotify_tracks;

-- Distribución por categoría de popularidad
SELECT popularity_category, COUNT(*) as total,
       ROUND(COUNT(*) / (SELECT COUNT(*) FROM spotify_tracks) * 100, 2) AS pct_total
FROM spotify_tracks
GROUP BY popularity_category;

-- Características promedio por categoría
SELECT popularity_category,
       ROUND(AVG(danceability), 3) AS avg_danceability,
       ROUND(AVG(energy), 3) AS avg_energy,
       ROUND(AVG(valence), 3) AS avg_valence,
       ROUND(AVG(tempo), 2) AS avg_tempo
FROM spotify_tracks
GROUP BY popularity_category;

-- Top 10 géneros más populares
SELECT track_genre, COUNT(*) AS total,
       ROUND(AVG(popularity), 2) AS popularidad_promedio
FROM spotify_tracks
GROUP BY track_genre
ORDER BY total DESC
LIMIT 10;

-- Revisar historial de métricas del modelo
SELECT execution_timestamp, model_type,
       ROUND(accuracy, 4) AS accuracy,
       ROUND(f1_weighted, 4) AS f1_score,
       train_records, test_records,
       ROUND(training_duration_s, 2) AS duracion_seg
FROM model_metrics
ORDER BY execution_timestamp DESC;

-- Matriz de confusión desde predicciones
SELECT actual_category, predicted_category, COUNT(*) AS total
FROM predictions
GROUP BY actual_category, predicted_category
ORDER BY actual_category, predicted_category;
```

---

## KPIs de Monitoreo

El pipeline registra los siguientes indicadores clave de desempeño:

| KPI | Descripción | Meta | Estado actual |
|---|---|---|---|
| Latencia del Pipeline | Tiempo total end-to-end (Stages 1-4) | < 5 minutos | ~7-10 segundos |
| Tasa de éxito de carga | Porcentaje registros cargados exitosamente | > 98% | 100% |
| Calidad de datos | Porcentaje registros que pasan validación | > 95% | 100% |
| Tasa de duplicados | Porcentaje duplicados eliminados | < 1% | ~0% |
| Accuracy del modelo | Precisión en clasificación de popularidad | > 60% | ~34% (modelo base) |
| F1-Score (weighted) | Media armónica precision-recall ponderada | > 0.55 | ~0.34 |

---

## Estructura de archivos

```
spotify-pipeline/
├── data/
│   ├── external/
│   │   └── spotify_tracks.csv              # Dataset fuente
│   ├── raw/
│   │   └── spotify_tracks_raw.csv          # Datos ingeridos
│   ├── processed/
│   │   └── spotify_tracks_clean.csv        # Datos limpios y transformados
│   ├── validated/
│   │   ├── spotify_tracks_validated.csv    # Registros válidos
│   │   └── spotify_tracks_rejected.csv     # Registros rechazados
│   ├── model/
│   │   └── spotify_popularity_classifier.pkl  # Modelo entrenado
│   ├── dashboard/
│   │   └── spotify_dashboard_data.csv      # Datos para visualización
│   └── reports/
│       ├── cleaning_report_*.json          # Reportes de limpieza
│       ├── validation_report_*.json        # Reportes de validación
│       ├── classification_report_*.{txt,json}  # Reportes de clasificación
│       ├── confusion_matrix_*.png          # Matriz de confusión
│       └── metrics_*.json                  # Métricas del modelo
├── scripts/
│   ├── 01_ingesta.py                       # Stage 1: Ingesta de datos
│   ├── 02_limpieza.py                      # Stage 2: Limpieza y transformación
│   ├── 03_validacion.py                    # Stage 3: Validación
│   ├── 04_carga.py                         # Stage 4: Entrenamiento y carga
│   └── run_pipeline.py                     # Orquestador
├── sql/
│   ├── create_tables.sql                   # DDL de tablas
│   └── queries_analytics.sql               # Consultas de análisis
├── logs/
│   ├── 01_ingesta_*.log
│   ├── 02_limpieza_*.log
│   ├── 03_validacion_*.log
│   └── 04_carga_*.log
├── docs/
│   └── (documentación adicional)
├── Dockerfile                               # Imagen del pipeline
├── docker-compose.yml                       # Orquestación de servicios
├── requirements.txt                         # Dependencias Python
├── .env.example                             # Plantilla de variables de entorno
├── .env                                     # Variables de entorno (no commitear)
├── .gitignore                               # Archivos ignorados por Git
└── README.md                                # Este archivo
```

---

## Ejecución del Pipeline

### Opción 1: Ejecución manual (paso a paso)

```bash
# En Docker
docker-compose exec pipeline python scripts/01_ingesta.py
docker-compose exec pipeline python scripts/02_limpieza.py
docker-compose exec pipeline python scripts/03_validacion.py
docker-compose exec pipeline python scripts/04_carga.py
```

### Opción 2: Ejecución completa automatizada

```bash
# Levanta todos los servicios y ejecuta el pipeline automáticamente
docker-compose up --build
```

### Opción 3: Ejecución local (sin Docker)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar stage por stage
python scripts/01_ingesta.py
python scripts/02_limpieza.py
python scripts/03_validacion.py
python scripts/04_carga.py

# O usar el orquestador
python scripts/run_pipeline.py
```

---

## Variables de Entorno

Archivo `.env`:

```
# Spotify Dataset
SPOTIFY_DATASET_PATH=data/external/spotify_tracks.csv
SAMPLE_SIZE=5000

# Model Parameters
TEST_SIZE=0.2
RANDOM_STATE=42
N_ESTIMATORS=200

# Database (optional - for storing results)
DB_HOST=mysql                              # mysql (Docker) o localhost (local)
DB_PORT=3306
DB_USER=pipeline_user
DB_PASSWORD=pipelinepass
DB_NAME=spotify_ml_db
```

---

## Problemas comunes y soluciones

### Error: "Connection refused" en MySQL

**Causa**: El contenedor MySQL no ha terminado de iniciar.

**Solución**:
```bash
docker-compose ps                    # Verificar estado
docker-compose logs mysql            # Ver logs de MySQL
docker-compose down && docker-compose up -d   # Reiniciar
```

### phpMyAdmin: "No se puede conectar a MySQL"

**Causa**: phpMyAdmin usa un host incorrecto.

**Solución**: En Docker, el host debe ser `mysql`, no `localhost`.

### Accuracy del modelo por debajo del umbral

**Causa**: El modelo Random Forest base con features de audio crudos no logra capturar patrones complejos de popularidad.

**Sugerencias**:
- Aumentar `N_ESTIMATORS` en `.env`
- Agregar más engineered features (interacciones, polinómicas)
- Probar otros algoritmos (XGBoost, Gradient Boosting)
- Aumentar `SAMPLE_SIZE` para más datos de entrenamiento
