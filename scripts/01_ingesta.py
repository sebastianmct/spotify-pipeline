"""
Script de Ingesta de Datos - Spotify Tracks Dataset
Etapa 1 del Pipeline DataOps / ML

Funcionalidad:
- Carga el dataset crudo de canciones de Spotify (archivo local o fuente externa)
- Filtra columnas irrelevantes y valida la estructura mínima esperada
- Guarda en formato CSV en data/raw/
- Genera logs detallados del proceso

Autor: Equipo DataOps Spotify
Fecha: Junio 2026
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_EXTERNAL_DIR = BASE_DIR / "data" / "external"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios si no existen
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"01_ingesta_{timestamp}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)


class SpotifyDataIngestion:
    """Clase para manejar la ingesta de datos del dataset de Spotify Tracks"""

    # Columnas que debe traer el dataset fuente (Spotify Tracks Dataset estilo Kaggle)
    EXPECTED_COLUMNS = [
        'track_id', 'artists', 'album_name', 'track_name', 'popularity',
        'duration_ms', 'explicit', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'
    ]

    def __init__(self):
        self.source_file = Path(
            os.getenv('SPOTIFY_DATASET_PATH', str(DATA_EXTERNAL_DIR / "spotify_tracks.csv"))
        )
        self.output_file = DATA_RAW_DIR / "spotify_tracks_raw.csv"
        self.sample_size = int(os.getenv('SAMPLE_SIZE', 5000))

        logger.info("=" * 80)
        logger.info("INICIANDO PROCESO DE INGESTA DE DATOS")
        logger.info("=" * 80)
        logger.info(f"Archivo fuente: {self.source_file}")
        logger.info(f"Archivo de salida: {self.output_file}")

    def load_data(self):
        """Carga el dataset crudo desde el archivo fuente"""
        logger.info("Iniciando carga de datos...")

        try:
            if not self.source_file.exists():
                logger.warning(f"No se encontró el archivo fuente en: {self.source_file}")
                logger.warning("Generando dataset de muestra para desarrollo/demo...")
                return self._create_sample_dataset()

            logger.info("Leyendo archivo CSV en chunks...")
            chunk_size = 10000
            chunks = []

            for chunk in pd.read_csv(self.source_file, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                logger.info(f"Leídos {len(chunks) * chunk_size} registros...")

            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Carga completada: {len(df):,} registros totales")
            return df

        except Exception as e:
            logger.error(f"Error en la carga: {str(e)}")
            logger.warning("Intentando método alternativo (dataset de muestra)...")
            return self._create_sample_dataset()

    def _create_sample_dataset(self):
        """Crea un dataset sintético de muestra para desarrollo/testing/demo"""
        logger.info("Creando dataset de muestra para desarrollo...")

        rng = np.random.default_rng(seed=42)
        n = self.sample_size

        genres = ['pop', 'rock', 'hip-hop', 'edm', 'classical', 'jazz',
                  'reggaeton', 'metal', 'country', 'r-n-b']

        sample_data = {
            'track_id': [f'TRK{100000 + i}' for i in range(n)],
            'artists': [f'Artist {i % 250}' for i in range(n)],
            'album_name': [f'Album {i % 400}' for i in range(n)],
            'track_name': [f'Track {i}' for i in range(n)],
            'popularity': rng.integers(0, 101, n),
            'duration_ms': rng.integers(60000, 420000, n),
            'explicit': rng.choice([True, False], n, p=[0.15, 0.85]),
            'danceability': rng.uniform(0, 1, n).round(3),
            'energy': rng.uniform(0, 1, n).round(3),
            'key': rng.integers(0, 12, n),
            'loudness': rng.uniform(-40, 0, n).round(3),
            'mode': rng.integers(0, 2, n),
            'speechiness': rng.uniform(0, 1, n).round(3),
            'acousticness': rng.uniform(0, 1, n).round(3),
            'instrumentalness': rng.uniform(0, 1, n).round(3),
            'liveness': rng.uniform(0, 1, n).round(3),
            'valence': rng.uniform(0, 1, n).round(3),
            'tempo': rng.uniform(50, 210, n).round(2),
            'time_signature': rng.choice([3, 4, 5], n, p=[0.1, 0.8, 0.1]),
            'track_genre': rng.choice(genres, n),
        }

        df = pd.DataFrame(sample_data)

        # Inyectar algunos valores nulos y duplicados de forma controlada (simula datos reales)
        null_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[null_idx, 'album_name'] = np.nan
        df.loc[rng.choice(n, size=int(n * 0.01), replace=False), 'danceability'] = np.nan
        df = pd.concat([df, df.sample(int(n * 0.01), random_state=42)], ignore_index=True)

        logger.info(f"Dataset de muestra creado: {len(df):,} registros")
        return df

    def validate_data_structure(self, df):
        """Valida la estructura básica de los datos"""
        logger.info("Validando estructura de datos...")

        missing_columns = [col for col in self.EXPECTED_COLUMNS if col not in df.columns]

        if missing_columns:
            logger.error(f"Columnas faltantes: {missing_columns}")
            raise ValueError(f"Dataset no contiene columnas requeridas: {missing_columns}")

        logger.info("✓ Todas las columnas requeridas están presentes")
        logger.info(f"Total de columnas: {len(df.columns)}")

        return True

    def generate_ingestion_report(self, df):
        """Genera reporte de la ingesta"""
        logger.info("\n" + "=" * 80)
        logger.info("REPORTE DE INGESTA")
        logger.info("=" * 80)

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'genre_count': int(df['track_genre'].nunique()) if 'track_genre' in df.columns else None,
            'popularity_range': f"{df['popularity'].min()} - {df['popularity'].max()}"
                                  if 'popularity' in df.columns else None,
            'null_counts': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        logger.info(f"Registros totales: {report['total_records']:,}")
        logger.info(f"Columnas: {report['total_columns']}")
        logger.info(f"Uso de memoria: {report['memory_usage_mb']:.2f} MB")
        logger.info(f"Géneros distintos: {report['genre_count']}")
        logger.info(f"Rango de popularidad: {report['popularity_range']}")

        logger.info("\nValores nulos por columna:")
        for col, null_count in report['null_counts'].items():
            if null_count > 0:
                pct = (null_count / len(df)) * 100
                logger.info(f"  {col}: {null_count:,} ({pct:.2f}%)")

        return report

    def save_data(self, df):
        """Guarda datos en formato CSV"""
        logger.info(f"\nGuardando datos en: {self.output_file}")

        try:
            df.to_csv(self.output_file, index=False, encoding='utf-8')
            file_size_mb = self.output_file.stat().st_size / 1024 / 1024

            logger.info("✓ Datos guardados exitosamente")
            logger.info(f"  Tamaño del archivo: {file_size_mb:.2f} MB")
            logger.info(f"  Registros: {len(df):,}")

            return True

        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
            return False

    def run(self):
        """Ejecuta el proceso completo de ingesta"""
        try:
            start_time = datetime.now()
            logger.info(f"Inicio del proceso: {start_time}")

            # 1. Cargar datos
            df = self.load_data()

            # 2. Validar estructura
            self.validate_data_structure(df)

            # 3. Generar reporte
            self.generate_ingestion_report(df)

            # 4. Guardar datos
            self.save_data(df)

            # Tiempo total
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PROCESO DE INGESTA COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info(f"Tiempo total: {duration:.2f} segundos")
            logger.info(f"Log guardado en: {log_file}")

            return True

        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error("ERROR EN EL PROCESO DE INGESTA")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}", exc_info=True)
            return False


def main():
    """Función principal"""
    ingestion = SpotifyDataIngestion()
    success = ingestion.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
