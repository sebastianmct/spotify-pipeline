"""
Script de Limpieza y Transformación de Datos - Spotify Tracks Dataset
Etapa 2 del Pipeline DataOps / ML

Funcionalidad:
- Elimina registros duplicados
- Maneja valores nulos con estrategias configurables por columna
- Normaliza columnas de texto (artistas, álbumes, géneros)
- Valida y acota rangos de variables de audio (0-1, loudness, tempo)
- Crea características derivadas (duración en minutos, categoría de popularidad)
- Optimiza tipos de datos para reducir uso de memoria
- Genera reporte de calidad de datos

Autor: Equipo DataOps Spotify
Fecha: Junio 2026
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "data" / "reports"

# Crear directorios si no existen
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"02_limpieza_{timestamp}.log"

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


class SpotifyDataCleaner:
    """Clase para limpieza, transformación e ingeniería de características"""

    # Variables de audio acotadas entre 0 y 1 por definición de la API de Spotify
    BOUNDED_FEATURES = [
        'danceability', 'energy', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence'
    ]

    def __init__(self):
        self.input_file = DATA_RAW_DIR / "spotify_tracks_raw.csv"
        self.output_file = DATA_PROCESSED_DIR / "spotify_tracks_clean.csv"
        self.df = None
        self.cleaning_stats = {
            'initial_records': 0,
            'final_records': 0,
            'duplicates_removed': 0,
            'nulls_handled': {},
            'out_of_range_clipped': 0,
            'records_transformed': 0
        }

        logger.info("=" * 80)
        logger.info("INICIANDO PROCESO DE LIMPIEZA Y TRANSFORMACIÓN")
        logger.info("=" * 80)
        logger.info(f"Archivo de entrada: {self.input_file}")
        logger.info(f"Archivo de salida: {self.output_file}")

    def load_data(self):
        """Carga datos desde el archivo raw"""
        logger.info("Cargando datos...")

        try:
            self.df = pd.read_csv(self.input_file, low_memory=False)
            self.cleaning_stats['initial_records'] = len(self.df)

            logger.info("✓ Datos cargados exitosamente")
            logger.info(f"  Registros: {len(self.df):,}")
            logger.info(f"  Columnas: {len(self.df.columns)}")
            logger.info(f"  Memoria: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            return True

        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {self.input_file}")
            logger.error("Por favor ejecute primero el script 01_ingesta.py")
            return False
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            return False

    def remove_duplicates(self):
        """Elimina registros duplicados basándose en track_id"""
        logger.info("\n--- Eliminando Duplicados ---")

        initial_count = len(self.df)

        self.df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)

        duplicates_removed = initial_count - len(self.df)
        self.cleaning_stats['duplicates_removed'] = duplicates_removed

        logger.info(f"Registros iniciales: {initial_count:,}")
        logger.info(f"Duplicados encontrados: {duplicates_removed:,}")
        logger.info(f"Registros después de limpieza: {len(self.df):,}")

        if duplicates_removed > 0:
            pct = (duplicates_removed / initial_count) * 100
            logger.warning(f"⚠ Se eliminaron {duplicates_removed:,} duplicados ({pct:.2f}%)")
        else:
            logger.info("✓ No se encontraron duplicados")

    def handle_missing_values(self):
        """Maneja valores nulos con estrategias específicas por columna"""
        logger.info("\n--- Manejo de Valores Nulos ---")

        null_counts = self.df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if len(columns_with_nulls) == 0:
            logger.info("✓ No se encontraron valores nulos")
            return

        logger.info(f"Columnas con valores nulos: {len(columns_with_nulls)}")
        for col, count in columns_with_nulls.items():
            pct = (count / len(self.df)) * 100
            logger.info(f"  {col}: {count:,} ({pct:.2f}%)")

        # Estrategias de manejo de nulos por columna
        strategies = {
            # Columnas críticas: eliminar registros si faltan
            'track_id': 'drop',
            'track_name': 'drop',
            'popularity': 'drop',
            'track_genre': 'drop',

            # Texto descriptivo: rellenar con 'UNKNOWN'
            'artists': 'fill_unknown',
            'album_name': 'fill_unknown',

            # Variables numéricas de audio: rellenar con la mediana
            'danceability': 'fill_median',
            'energy': 'fill_median',
            'loudness': 'fill_median',
            'speechiness': 'fill_median',
            'acousticness': 'fill_median',
            'instrumentalness': 'fill_median',
            'liveness': 'fill_median',
            'valence': 'fill_median',
            'tempo': 'fill_median',
            'duration_ms': 'fill_median',

            # Variables categóricas numéricas: rellenar con la moda
            'key': 'fill_mode',
            'mode': 'fill_mode',
            'time_signature': 'fill_mode',

            # Booleanos: rellenar con False
            'explicit': 'fill_false',
        }

        rows_before = len(self.df)

        # 1. Eliminar filas con campos críticos nulos
        critical_columns = [col for col, strategy in strategies.items()
                             if strategy == 'drop' and col in self.df.columns]
        if critical_columns:
            self.df.dropna(subset=critical_columns, inplace=True)
            logger.info(f"Registros eliminados por campos críticos nulos: {rows_before - len(self.df):,}")

        # 2. Rellenar texto con 'UNKNOWN'
        for col, strategy in strategies.items():
            if strategy == 'fill_unknown' and col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    self.df[col] = self.df[col].fillna('UNKNOWN')
                    self.cleaning_stats['nulls_handled'][col] = int(null_count)
                    logger.info(f"  {col}: {null_count:,} nulos rellenados con 'UNKNOWN'")

        # 3. Rellenar numéricos con la mediana
        for col, strategy in strategies.items():
            if strategy == 'fill_median' and col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    median_value = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_value)
                    self.cleaning_stats['nulls_handled'][col] = int(null_count)
                    logger.info(f"  {col}: {null_count:,} nulos rellenados con mediana ({median_value:.3f})")

        # 4. Rellenar categóricos numéricos con la moda
        for col, strategy in strategies.items():
            if strategy == 'fill_mode' and col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    mode_value = self.df[col].mode().iloc[0]
                    self.df[col] = self.df[col].fillna(mode_value)
                    self.cleaning_stats['nulls_handled'][col] = int(null_count)
                    logger.info(f"  {col}: {null_count:,} nulos rellenados con moda ({mode_value})")

        # 5. Rellenar booleanos con False
        for col, strategy in strategies.items():
            if strategy == 'fill_false' and col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    self.df[col] = self.df[col].fillna(False)
                    self.cleaning_stats['nulls_handled'][col] = int(null_count)
                    logger.info(f"  {col}: {null_count:,} nulos rellenados con False")

    def normalize_text_columns(self):
        """Normaliza columnas de texto: espacios, mayúsculas/minúsculas, codificación"""
        logger.info("\n--- Normalizando Columnas de Texto ---")

        text_columns = ['artists', 'album_name', 'track_name']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                logger.info(f"  {col}: espacios eliminados")

        if 'track_genre' in self.df.columns:
            self.df['track_genre'] = self.df['track_genre'].astype(str).str.strip().str.lower()
            logger.info("  track_genre: normalizado a minúsculas")

    def validate_audio_feature_ranges(self):
        """Valida y acota (clip) las variables de audio fuera de su rango definido"""
        logger.info("\n--- Validando Rangos de Variables de Audio ---")

        clipped_total = 0

        # Variables acotadas entre 0 y 1
        for col in self.BOUNDED_FEATURES:
            if col in self.df.columns:
                out_of_range = ((self.df[col] < 0) | (self.df[col] > 1)).sum()
                if out_of_range > 0:
                    self.df[col] = self.df[col].clip(lower=0, upper=1)
                    clipped_total += out_of_range
                    logger.info(f"  {col}: {out_of_range:,} valores fuera de [0,1] acotados")

        # Loudness: rango típico entre -60 dB y 5 dB
        if 'loudness' in self.df.columns:
            out_of_range = ((self.df['loudness'] < -60) | (self.df['loudness'] > 5)).sum()
            if out_of_range > 0:
                self.df['loudness'] = self.df['loudness'].clip(lower=-60, upper=5)
                clipped_total += out_of_range
                logger.info(f"  loudness: {out_of_range:,} valores fuera de [-60, 5] acotados")

        # Tempo: descartar valores no positivos o irreales
        if 'tempo' in self.df.columns:
            invalid_tempo = (self.df['tempo'] <= 0).sum()
            if invalid_tempo > 0:
                median_tempo = self.df.loc[self.df['tempo'] > 0, 'tempo'].median()
                self.df.loc[self.df['tempo'] <= 0, 'tempo'] = median_tempo
                clipped_total += invalid_tempo
                logger.info(f"  tempo: {invalid_tempo:,} valores inválidos reemplazados por mediana")

        # Duración: descartar canciones con duración no positiva
        if 'duration_ms' in self.df.columns:
            invalid_duration = (self.df['duration_ms'] <= 0).sum()
            if invalid_duration > 0:
                self.df = self.df[self.df['duration_ms'] > 0]
                logger.info(f"  duration_ms: {invalid_duration:,} registros con duración inválida eliminados")

        self.cleaning_stats['out_of_range_clipped'] = int(clipped_total)

        if clipped_total == 0:
            logger.info("✓ Todas las variables de audio están dentro de rango")

    def create_additional_features(self):
        """Crea características derivadas para análisis y modelado"""
        logger.info("\n--- Creando Características Adicionales ---")

        # Duración en minutos (más legible para el dashboard)
        if 'duration_ms' in self.df.columns:
            self.df['duration_min'] = (self.df['duration_ms'] / 60000).round(2)
            logger.info("  ✓ duration_min creada")

        # Categoría de popularidad: variable objetivo para clasificación supervisada
        if 'popularity' in self.df.columns:
            bins = [-1, 33, 66, 100]
            labels = ['Baja', 'Media', 'Alta']
            self.df['popularity_category'] = pd.cut(
                self.df['popularity'], bins=bins, labels=labels
            ).astype(str)
            logger.info("  ✓ popularity_category creada (Baja / Media / Alta)")
            distribution = self.df['popularity_category'].value_counts()
            for cat, count in distribution.items():
                logger.info(f"    {cat}: {count:,} ({count / len(self.df) * 100:.2f}%)")

        # Bandera de explícito como booleano normalizado
        if 'explicit' in self.df.columns:
            self.df['explicit'] = self.df['explicit'].astype(bool)
            logger.info("  ✓ explicit normalizado a booleano")

        # Agrupación de géneros poco frecuentes en 'other' (reduce alta cardinalidad)
        if 'track_genre' in self.df.columns:
            genre_threshold = max(1, int(len(self.df) * 0.005))
            genre_counts = self.df['track_genre'].value_counts()
            rare_genres = genre_counts[genre_counts < genre_threshold].index
            self.df['track_genre_grouped'] = self.df['track_genre'].where(
                ~self.df['track_genre'].isin(rare_genres), 'other'
            )
            logger.info(f"  ✓ track_genre_grouped creada ({len(rare_genres)} géneros raros agrupados en 'other')")

        self.cleaning_stats['records_transformed'] = len(self.df)

    def optimize_datatypes(self):
        """Optimiza tipos de datos para reducir uso de memoria"""
        logger.info("\n--- Optimizando Tipos de Datos ---")

        memory_before = self.df.memory_usage(deep=True).sum() / 1024 / 1024

        categorical_threshold = 0.5
        for col in self.df.select_dtypes(include=['object']).columns:
            num_unique = self.df[col].nunique()
            num_total = len(self.df[col])

            if num_total > 0 and num_unique / num_total < categorical_threshold:
                self.df[col] = self.df[col].astype('category')
                logger.info(f"  {col}: convertido a category ({num_unique} valores únicos)")

        for col in self.df.select_dtypes(include=['int64']).columns:
            col_min = self.df[col].min()
            col_max = self.df[col].max()

            if col_min >= 0:
                if col_max < 255:
                    self.df[col] = self.df[col].astype('uint8')
                elif col_max < 65535:
                    self.df[col] = self.df[col].astype('uint16')
                elif col_max < 4294967295:
                    self.df[col] = self.df[col].astype('uint32')
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    self.df[col] = self.df[col].astype('int8')
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    self.df[col] = self.df[col].astype('int16')

        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')

        memory_after = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = ((memory_before - memory_after) / memory_before) * 100 if memory_before > 0 else 0

        logger.info(f"Memoria antes: {memory_before:.2f} MB")
        logger.info(f"Memoria después: {memory_after:.2f} MB")
        logger.info(f"Reducción: {reduction:.2f}%")

    def generate_cleaning_report(self):
        """Genera reporte detallado del proceso de limpieza"""
        logger.info("\n" + "=" * 80)
        logger.info("REPORTE DE LIMPIEZA Y TRANSFORMACIÓN")
        logger.info("=" * 80)

        self.cleaning_stats['final_records'] = len(self.df)

        logger.info("\nEstadísticas Generales:")
        logger.info(f"  Registros iniciales: {self.cleaning_stats['initial_records']:,}")
        logger.info(f"  Registros finales: {self.cleaning_stats['final_records']:,}")
        logger.info(f"  Duplicados removidos: {self.cleaning_stats['duplicates_removed']:,}")
        logger.info(f"  Valores fuera de rango acotados: {self.cleaning_stats['out_of_range_clipped']:,}")

        retention_rate = (self.cleaning_stats['final_records'] /
                           self.cleaning_stats['initial_records']) * 100
        logger.info(f"  Tasa de retención: {retention_rate:.2f}%")

        if self.cleaning_stats['nulls_handled']:
            logger.info("\nValores nulos manejados:")
            for col, count in self.cleaning_stats['nulls_handled'].items():
                logger.info(f"  {col}: {count:,}")

        logger.info("\nCalidad de Datos Final:")
        total_cells = len(self.df) * len(self.df.columns)
        null_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells) * 100
        logger.info(f"  Completitud: {completeness:.2f}%")
        logger.info(f"  Total de columnas: {len(self.df.columns)}")

        report_file = REPORTS_DIR / f"cleaning_report_{timestamp}.json"

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.cleaning_stats,
            'final_shape': {'rows': len(self.df), 'columns': len(self.df.columns)},
            'data_quality': {
                'completeness_pct': float(completeness),
                'retention_rate_pct': float(retention_rate)
            },
            'column_info': {
                col: {
                    'dtype': str(dtype),
                    'nulls': int(self.df[col].isnull().sum()),
                    'unique_values': int(self.df[col].nunique())
                }
                for col, dtype in self.df.dtypes.items()
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"\n✓ Reporte guardado en: {report_file}")

    def save_cleaned_data(self):
        """Guarda datos limpios"""
        logger.info(f"\nGuardando datos limpios en: {self.output_file}")

        try:
            logger.info("Eliminando registros con valores NaN restantes...")
            rows_before_final = len(self.df)

            null_counts = self.df.isnull().sum()
            if null_counts.sum() > 0:
                logger.warning("Columnas con NaN restantes:")
                for col, count in null_counts[null_counts > 0].items():
                    logger.warning(f"  {col}: {count:,} valores NaN")

            self.df = self.df.dropna(how='any')
            rows_after_final = len(self.df)
            rows_dropped_final = rows_before_final - rows_after_final

            if rows_dropped_final > 0:
                logger.warning(
                    f"✓ Eliminados {rows_dropped_final:,} registros con NaN "
                    f"({(rows_dropped_final / rows_before_final) * 100:.2f}%)"
                )
                self.cleaning_stats['final_records'] = rows_after_final

            self.df.to_csv(self.output_file, index=False, encoding='utf-8', na_rep='')
            file_size_mb = self.output_file.stat().st_size / 1024 / 1024

            logger.info("✓ Datos guardados exitosamente")
            logger.info(f"  Tamaño del archivo: {file_size_mb:.2f} MB")
            logger.info(f"  Registros: {len(self.df):,}")
            logger.info(f"  Columnas: {len(self.df.columns)}")

            return True

        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
            return False

    def run(self):
        """Ejecuta el proceso completo de limpieza"""
        try:
            start_time = datetime.now()
            logger.info(f"Inicio del proceso: {start_time}")

            # 1. Cargar datos
            if not self.load_data():
                return False

            # 2. Eliminar duplicados
            self.remove_duplicates()

            # 3. Manejar valores nulos
            self.handle_missing_values()

            # 4. Normalizar texto
            self.normalize_text_columns()

            # 5. Validar rangos de variables de audio
            self.validate_audio_feature_ranges()

            # 6. Crear características adicionales (incluye variable objetivo)
            self.create_additional_features()

            # 7. Optimizar tipos de datos
            self.optimize_datatypes()

            # 8. Generar reporte
            self.generate_cleaning_report()

            # 9. Guardar datos limpios
            self.save_cleaned_data()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PROCESO DE LIMPIEZA COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info(f"Tiempo total: {duration:.2f} segundos")
            logger.info(f"Log guardado en: {log_file}")

            return True

        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error("ERROR EN EL PROCESO DE LIMPIEZA")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}", exc_info=True)
            return False


def main():
    """Función principal"""
    cleaner = SpotifyDataCleaner()
    success = cleaner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
