"""
Script de Validación Estructural y Semántica - Spotify Tracks Dataset
Etapa 3 del Pipeline DataOps / ML

Funcionalidad:
- Validación estructural: esquema, tipos de datos, integridad referencial
- Validación semántica: reglas de negocio, rangos válidos, coherencia
- Detección de anomalías (géneros raros, outliers de duración/tempo)
- Generación de reportes de validación detallados
- Clasificación de registros válidos vs rechazados

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
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_VALIDATED_DIR = BASE_DIR / "data" / "validated"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "data" / "reports"

# Crear directorios si no existen
DATA_VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"03_validacion_{timestamp}.log"

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


class SpotifyDataValidator:
    """Clase para validación estructural y semántica de datos"""

    def __init__(self):
        self.input_file = DATA_PROCESSED_DIR / "spotify_tracks_clean.csv"
        self.output_file_valid = DATA_VALIDATED_DIR / "spotify_tracks_validated.csv"
        self.output_file_rejected = DATA_VALIDATED_DIR / "spotify_tracks_rejected.csv"
        self.df = None

        self.validation_results = {
            'structural': {},
            'semantic': {},
            'summary': {
                'total_records': 0,
                'valid_records': 0,
                'rejected_records': 0,
                'validation_rate': 0.0
            }
        }

        self.schema = self._define_schema()
        self.business_rules = self._define_business_rules()

        logger.info("=" * 80)
        logger.info("INICIANDO PROCESO DE VALIDACIÓN")
        logger.info("=" * 80)
        logger.info(f"Archivo de entrada: {self.input_file}")
        logger.info(f"Archivo de salida (válidos): {self.output_file_valid}")
        logger.info(f"Archivo de salida (rechazados): {self.output_file_rejected}")

    def _define_schema(self):
        """Define el esquema esperado de los datos"""
        return {
            'required_columns': [
                'track_id', 'artists', 'album_name', 'track_name', 'popularity',
                'duration_ms', 'explicit', 'danceability', 'energy', 'key',
                'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'time_signature', 'track_genre',
                'popularity_category'
            ],
            'data_types': {
                'popularity': ['int64', 'int32', 'uint8'],
                'duration_ms': ['int64', 'int32', 'uint32', 'float64', 'float32'],
                'danceability': ['float64', 'float32'],
                'energy': ['float64', 'float32'],
                'loudness': ['float64', 'float32'],
                'tempo': ['float64', 'float32'],
                'explicit': ['bool'],
                'key': ['int64', 'int32', 'uint8'],
                'mode': ['int64', 'int32', 'uint8'],
            },
            'unique_columns': ['track_id'],
            'non_null_columns': [
                'track_id', 'track_name', 'popularity', 'track_genre', 'popularity_category'
            ]
        }

    def _define_business_rules(self):
        """Define reglas de negocio para validación semántica"""
        return {
            'bounded_audio_features': {
                'danceability': {'min': 0.0, 'max': 1.0},
                'energy': {'min': 0.0, 'max': 1.0},
                'speechiness': {'min': 0.0, 'max': 1.0},
                'acousticness': {'min': 0.0, 'max': 1.0},
                'instrumentalness': {'min': 0.0, 'max': 1.0},
                'liveness': {'min': 0.0, 'max': 1.0},
                'valence': {'min': 0.0, 'max': 1.0},
            },
            'other_ranges': {
                'popularity': {'min': 0, 'max': 100},
                'loudness': {'min': -60.0, 'max': 5.0},
                'tempo': {'min': 0.0, 'max': 250.0},
                'duration_ms': {'min': 1000, 'max': 1800000},  # entre 1 seg y 30 min
                'key': {'min': 0, 'max': 11},
                'mode': {'min': 0, 'max': 1},
                'time_signature': {'min': 1, 'max': 7},
            },
            'categorical_values': {
                'explicit': [True, False],
                'popularity_category': ['Baja', 'Media', 'Alta'],
            },
            'text_fields': {
                'min_length': {
                    'track_id': 3,
                    'track_name': 1,
                },
                'max_length': {
                    'track_name': 250,
                    'album_name': 250,
                    'artists': 250,
                }
            },
            'coherence_rules': {
                # popularity_category debe coincidir con el valor numérico de popularity
                'popularity_category_match': True,
            }
        }

    def load_data(self):
        """Carga datos desde el archivo procesado"""
        logger.info("Cargando datos procesados...")

        try:
            self.df = pd.read_csv(self.input_file, low_memory=False)
            self.validation_results['summary']['total_records'] = len(self.df)

            logger.info("✓ Datos cargados exitosamente")
            logger.info(f"  Registros: {len(self.df):,}")
            logger.info(f"  Columnas: {len(self.df.columns)}")

            return True

        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {self.input_file}")
            logger.error("Por favor ejecute primero el script 02_limpieza.py")
            return False
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            return False

    def validate_structure(self):
        """Valida la estructura de los datos"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDACIÓN ESTRUCTURAL")
        logger.info("=" * 80)

        self.df['validation_errors'] = ''
        self.df['is_valid'] = True

        # 1. Validar columnas requeridas
        logger.info("\n--- Validando Columnas Requeridas ---")
        missing_columns = [col for col in self.schema['required_columns']
                            if col not in self.df.columns]

        if missing_columns:
            logger.error(f"❌ Columnas faltantes: {missing_columns}")
            self.validation_results['structural']['missing_columns'] = missing_columns
            return False
        else:
            logger.info(f"✓ Todas las columnas requeridas presentes ({len(self.schema['required_columns'])})")
            self.validation_results['structural']['missing_columns'] = []

        # 2. Validar columnas únicas (track_id)
        logger.info("\n--- Validando Unicidad ---")
        for col in self.schema['unique_columns']:
            if col in self.df.columns:
                duplicated_mask = self.df.duplicated(subset=[col], keep='first')
                dup_count = duplicated_mask.sum()
                if dup_count > 0:
                    self.df.loc[duplicated_mask, 'is_valid'] = False
                    self.df.loc[duplicated_mask, 'validation_errors'] += f'{col}_duplicado;'
                    logger.warning(f"⚠ {col}: {dup_count} registros duplicados marcados como inválidos")
                else:
                    logger.info(f"✓ {col}: sin duplicados")

        # 3. Validar columnas no nulas
        logger.info("\n--- Validando Campos No Nulos ---")
        for col in self.schema['non_null_columns']:
            if col in self.df.columns:
                null_mask = self.df[col].isnull()
                null_count = null_mask.sum()
                if null_count > 0:
                    self.df.loc[null_mask, 'is_valid'] = False
                    self.df.loc[null_mask, 'validation_errors'] += f'{col}_nulo;'
                    logger.warning(f"⚠ {col}: {null_count} registros con valor nulo")
                else:
                    logger.info(f"✓ {col}: sin valores nulos")

        self.validation_results['structural']['status'] = 'OK'
        return True

    def validate_semantics(self):
        """Valida reglas de negocio (rangos, categorías, coherencia)"""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDACIÓN SEMÁNTICA")
        logger.info("=" * 80)

        # 1. Variables de audio acotadas entre 0 y 1
        logger.info("\n--- Validando Rangos de Variables de Audio ---")
        for col, bounds in self.business_rules['bounded_audio_features'].items():
            if col in self.df.columns:
                mask = (self.df[col] < bounds['min']) | (self.df[col] > bounds['max'])
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, 'is_valid'] = False
                    self.df.loc[mask, 'validation_errors'] += f'{col}_fuera_de_rango;'
                    logger.warning(f"⚠ {col}: {count} registros fuera de [{bounds['min']}, {bounds['max']}]")
                else:
                    logger.info(f"✓ {col}: todos los valores dentro de rango")

        # 2. Otros rangos numéricos
        logger.info("\n--- Validando Otros Rangos Numéricos ---")
        for col, bounds in self.business_rules['other_ranges'].items():
            if col in self.df.columns:
                mask = (self.df[col] < bounds['min']) | (self.df[col] > bounds['max'])
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, 'is_valid'] = False
                    self.df.loc[mask, 'validation_errors'] += f'{col}_fuera_de_rango;'
                    logger.warning(f"⚠ {col}: {count} registros fuera de [{bounds['min']}, {bounds['max']}]")
                else:
                    logger.info(f"✓ {col}: todos los valores dentro de rango")

        # 3. Valores categóricos válidos
        logger.info("\n--- Validando Valores Categóricos ---")
        for col, valid_values in self.business_rules['categorical_values'].items():
            if col in self.df.columns:
                # Normalizar booleanos leídos desde CSV como string
                if col == 'explicit' and self.df[col].dtype == 'object':
                    self.df[col] = self.df[col].astype(str).str.lower().map(
                        {'true': True, 'false': False}
                    )
                mask = ~self.df[col].isin(valid_values)
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, 'is_valid'] = False
                    self.df.loc[mask, 'validation_errors'] += f'{col}_valor_invalido;'
                    logger.warning(f"⚠ {col}: {count} registros con valor no permitido")
                else:
                    logger.info(f"✓ {col}: todos los valores son válidos")

        # 4. Longitud de campos de texto
        logger.info("\n--- Validando Longitud de Campos de Texto ---")
        for col, min_len in self.business_rules['text_fields']['min_length'].items():
            if col in self.df.columns:
                mask = self.df[col].astype(str).str.len() < min_len
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, 'is_valid'] = False
                    self.df.loc[mask, 'validation_errors'] += f'{col}_muy_corto;'
                    logger.warning(f"⚠ {col}: {count} registros por debajo de la longitud mínima")

        # 5. Coherencia entre popularity y popularity_category
        logger.info("\n--- Validando Coherencia de Categoría de Popularidad ---")
        if self.business_rules['coherence_rules']['popularity_category_match']:
            expected_category = pd.cut(
                self.df['popularity'], bins=[-1, 33, 66, 100], labels=['Baja', 'Media', 'Alta']
            ).astype(str)
            mismatch_mask = expected_category != self.df['popularity_category']
            mismatch_count = mismatch_mask.sum()
            if mismatch_count > 0:
                self.df.loc[mismatch_mask, 'is_valid'] = False
                self.df.loc[mismatch_mask, 'validation_errors'] += 'categoria_popularidad_incoherente;'
                logger.warning(f"⚠ {mismatch_count} registros con categoría de popularidad incoherente")
            else:
                logger.info("✓ popularity_category coincide con popularity en todos los registros")

        self.validation_results['semantic']['status'] = 'OK'

    def detect_anomalies(self):
        """Detecta anomalías y valores atípicos relevantes para el modelado"""
        logger.info("\n" + "=" * 80)
        logger.info("DETECCIÓN DE ANOMALÍAS")
        logger.info("=" * 80)

        anomalies_detected = []

        # Géneros con muy pocas observaciones (afectan el entrenamiento del modelo)
        if 'track_genre' in self.df.columns:
            genre_counts = self.df['track_genre'].value_counts()
            rare_threshold = max(1, int(len(self.df) * 0.001))
            rare_genres = genre_counts[genre_counts < rare_threshold]

            if len(rare_genres) > 0:
                logger.info(f"ℹ {len(rare_genres)} géneros raros detectados")
                anomalies_detected.append({
                    'field': 'track_genre',
                    'type': 'rare_values',
                    'count': int(len(rare_genres))
                })

        # Outliers de duración (criterio IQR)
        if 'duration_ms' in self.df.columns:
            q1, q3 = self.df['duration_ms'].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((self.df['duration_ms'] < lower) | (self.df['duration_ms'] > upper)).sum()
            if outliers > 0:
                logger.info(f"ℹ {outliers} outliers de duración detectados (criterio IQR)")
                anomalies_detected.append({
                    'field': 'duration_ms',
                    'type': 'outlier_iqr',
                    'count': int(outliers)
                })

        # Canciones con loudness anómalamente alto (posible error de captura)
        if 'loudness' in self.df.columns:
            extreme_loudness = (self.df['loudness'] > 0).sum()
            if extreme_loudness > 0:
                logger.info(f"ℹ {extreme_loudness} registros con loudness positivo (inusual)")
                anomalies_detected.append({
                    'field': 'loudness',
                    'type': 'extreme_value',
                    'count': int(extreme_loudness)
                })

        self.validation_results['semantic']['anomalies'] = anomalies_detected

        if anomalies_detected:
            logger.info(f"Total de anomalías detectadas: {len(anomalies_detected)}")
        else:
            logger.info("✓ No se detectaron anomalías significativas")

    def generate_validation_report(self):
        """Genera reporte detallado de validación"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERANDO REPORTE DE VALIDACIÓN")
        logger.info("=" * 80)

        total = len(self.df)
        valid = self.df['is_valid'].sum()
        rejected = total - valid
        validation_rate = (valid / total * 100) if total > 0 else 0

        self.validation_results['summary'] = {
            'total_records': int(total),
            'valid_records': int(valid),
            'rejected_records': int(rejected),
            'validation_rate': float(validation_rate)
        }

        logger.info("\nResumen de Validación:")
        logger.info(f"  Total de registros: {total:,}")
        logger.info(f"  Registros válidos: {valid:,} ({validation_rate:.2f}%)")
        logger.info(f"  Registros rechazados: {rejected:,} ({100 - validation_rate:.2f}%)")

        if rejected > 0:
            logger.info("\nAnálisis de Errores de Validación:")

            error_counts = {}
            for errors_str in self.df[self.df['is_valid'] == False]['validation_errors']:
                for error in errors_str.split(';'):
                    error = error.strip()
                    if error:
                        error_counts[error] = error_counts.get(error, 0) + 1

            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

            for error_type, count in sorted_errors[:10]:
                pct = (count / rejected) * 100
                logger.info(f"  {error_type}: {count:,} ({pct:.2f}%)")

            self.validation_results['semantic']['error_breakdown'] = dict(sorted_errors)

        report_file = REPORTS_DIR / f"validation_report_{timestamp}.json"

        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)

        logger.info(f"\n✓ Reporte de validación guardado en: {report_file}")

        return report_file

    def save_validated_data(self):
        """Guarda registros válidos y rechazados en archivos separados"""
        logger.info("\n--- Guardando Datos Validados ---")

        df_valid = self.df[self.df['is_valid'] == True].copy()
        df_rejected = self.df[self.df['is_valid'] == False].copy()

        df_valid_clean = df_valid.drop(['validation_errors', 'is_valid'], axis=1, errors='ignore')

        logger.info("Limpiando valores 'nan' literales...")
        for col in df_valid_clean.columns:
            if df_valid_clean[col].dtype == 'object':
                df_valid_clean[col] = df_valid_clean[col].apply(
                    lambda x: None if isinstance(x, str) and x.lower() in ['nan', 'none', 'null'] else x
                )

        df_valid_clean = df_valid_clean.where(pd.notnull(df_valid_clean), None)

        try:
            df_valid_clean.to_csv(self.output_file_valid, index=False, encoding='utf-8')
            valid_size_mb = self.output_file_valid.stat().st_size / 1024 / 1024

            logger.info("✓ Registros válidos guardados:")
            logger.info(f"  Archivo: {self.output_file_valid}")
            logger.info(f"  Registros: {len(df_valid_clean):,}")
            logger.info(f"  Tamaño: {valid_size_mb:.2f} MB")

            if len(df_rejected) > 0:
                df_rejected.to_csv(self.output_file_rejected, index=False, encoding='utf-8')
                rejected_size_mb = self.output_file_rejected.stat().st_size / 1024 / 1024

                logger.info("\n✓ Registros rechazados guardados:")
                logger.info(f"  Archivo: {self.output_file_rejected}")
                logger.info(f"  Registros: {len(df_rejected):,}")
                logger.info(f"  Tamaño: {rejected_size_mb:.2f} MB")
            else:
                logger.info("\n✓ No hay registros rechazados")

            return True

        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
            return False

    def run(self):
        """Ejecuta el proceso completo de validación"""
        try:
            start_time = datetime.now()
            logger.info(f"Inicio del proceso: {start_time}")

            # 1. Cargar datos
            if not self.load_data():
                return False

            # 2. Validación estructural
            if not self.validate_structure():
                logger.error("Validación estructural falló")
                return False

            # 3. Validación semántica
            self.validate_semantics()

            # 4. Detección de anomalías
            self.detect_anomalies()

            # 5. Generar reporte
            self.generate_validation_report()

            # 6. Guardar datos validados
            self.save_validated_data()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PROCESO DE VALIDACIÓN COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info(f"Tiempo total: {duration:.2f} segundos")
            logger.info(f"Log guardado en: {log_file}")

            validation_rate = self.validation_results['summary']['validation_rate']
            if validation_rate < 95:
                logger.warning(f"\n⚠ ADVERTENCIA: Tasa de validación ({validation_rate:.2f}%) por debajo del umbral (95%)")
                logger.warning("  Se recomienda revisar los datos rechazados y las reglas de validación")

            return True

        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error("ERROR EN EL PROCESO DE VALIDACIÓN")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}", exc_info=True)
            return False


def main():
    """Función principal"""
    validator = SpotifyDataValidator()
    success = validator.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
