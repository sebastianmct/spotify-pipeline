"""
Script de Entrenamiento, Evaluación y Carga de Resultados - Spotify Tracks Dataset
Etapa 4 del Pipeline DataOps / ML

Funcionalidad:
- Ingeniería de características final (encoding, selección de variables)
- División de datos en entrenamiento y prueba (train/test split)
- Entrenamiento de un modelo de clasificación supervisada (Random Forest)
- Evaluación del modelo: accuracy, precision, recall, f1-score
- Generación y guardado de:
    * modelo entrenado (.pkl)
    * matriz de confusión (.png)
    * classification report (.txt y .json)
    * métricas principales (.json)
    * dataset procesado para el dashboard (.csv)

Autor: Equipo DataOps Spotify
Fecha: Junio 2026
"""

import os
import sys
import logging
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica, apto para ejecución por consola
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Configuración de paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_VALIDATED_DIR = BASE_DIR / "data" / "validated"
DATA_DASHBOARD_DIR = BASE_DIR / "data" / "dashboard"
MODEL_DIR = BASE_DIR / "data" / "model"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "data" / "reports"

# Crear directorios si no existen
DATA_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"04_carga_{timestamp}.log"

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
    logger = logging.getLogger(__name__)
else:
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
logger = logging.getLogger(__name__)


class SpotifyModelTrainer:
    """Clase para entrenamiento, evaluación y carga de artefactos del modelo"""

    # Variables numéricas de audio usadas como features del modelo
    NUMERIC_FEATURES = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_min',
        'key', 'mode', 'time_signature'
    ]
    CATEGORICAL_FEATURES = ['explicit', 'track_genre_grouped']
    TARGET_COLUMN = 'popularity_category'

    def __init__(self):
        self.input_file = DATA_VALIDATED_DIR / "spotify_tracks_validated.csv"
        self.model_file = MODEL_DIR / "spotify_popularity_classifier.pkl"
        self.confusion_matrix_file = REPORTS_DIR / f"confusion_matrix_{timestamp}.png"
        self.classification_report_txt = REPORTS_DIR / f"classification_report_{timestamp}.txt"
        self.classification_report_json = REPORTS_DIR / f"classification_report_{timestamp}.json"
        self.metrics_file = REPORTS_DIR / f"metrics_{timestamp}.json"
        self.dashboard_file = DATA_DASHBOARD_DIR / "spotify_dashboard_data.csv"

        self.test_size = float(os.getenv('TEST_SIZE', 0.2))
        self.random_state = int(os.getenv('RANDOM_STATE', 42))
        self.n_estimators = int(os.getenv('N_ESTIMATORS', 200))

        self.df = None
        self.label_encoders = {}
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.y_pred = None

        self.training_stats = {
            'total_records': 0,
            'train_records': 0,
            'test_records': 0,
            'features_used': [],
            'training_time_seconds': 0
        }

        logger.info("=" * 80)
        logger.info("INICIANDO ENTRENAMIENTO Y CARGA DE RESULTADOS")
        logger.info("=" * 80)
        logger.info(f"Archivo de entrada: {self.input_file}")
        logger.info(f"Variable objetivo: {self.TARGET_COLUMN}")
        logger.info(f"Proporción de prueba (test_size): {self.test_size}")

    def load_data(self):
        """Carga datos validados"""
        logger.info("\nCargando datos validados...")

        try:
            self.df = pd.read_csv(self.input_file, low_memory=False)
            self.training_stats['total_records'] = len(self.df)

            logger.info("✓ Datos cargados exitosamente")
            logger.info(f"  Registros: {len(self.df):,}")
            logger.info(f"  Columnas: {len(self.df.columns)}")

            return True

        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {self.input_file}")
            logger.error("Por favor ejecute primero el script 03_validacion.py")
            return False
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            return False

    def engineer_features(self):
        """Codifica variables categóricas y arma la matriz de features (X) y el target (y)"""
        logger.info("\n--- Ingeniería de Características Final ---")

        # Codificar variables categóricas con LabelEncoder (se guardan para uso futuro/dashboard)
        for col in self.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                encoder = LabelEncoder()
                self.df[f'{col}_encoded'] = encoder.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = encoder
                logger.info(f"  ✓ {col} codificada ({len(encoder.classes_)} categorías)")

        feature_columns = self.NUMERIC_FEATURES + [f'{c}_encoded' for c in self.CATEGORICAL_FEATURES
                                                     if c in self.df.columns]
        feature_columns = [c for c in feature_columns if c in self.df.columns]

        self.training_stats['features_used'] = feature_columns

        logger.info(f"  Total de features seleccionadas: {len(feature_columns)}")
        logger.info(f"  Features: {feature_columns}")

        X = self.df[feature_columns].copy()
        y = self.df[self.TARGET_COLUMN].copy()

        return X, y

    def split_data(self, X, y):
        """Divide los datos en conjuntos de entrenamiento y prueba"""
        logger.info("\n--- Dividiendo Datos (Train/Test Split) ---")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        self.training_stats['train_records'] = len(self.X_train)
        self.training_stats['test_records'] = len(self.X_test)

        logger.info(f"  Registros de entrenamiento: {len(self.X_train):,}")
        logger.info(f"  Registros de prueba: {len(self.X_test):,}")
        logger.info("\n  Distribución de clases en entrenamiento:")
        for cls, count in self.y_train.value_counts().items():
            logger.info(f"    {cls}: {count:,} ({count / len(self.y_train) * 100:.2f}%)")

    def train_model(self):
        """Entrena el modelo de clasificación supervisada"""
        logger.info("\n--- Entrenando Modelo (Random Forest Classifier) ---")

        start_time = datetime.now()

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=None,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)

        duration = (datetime.now() - start_time).total_seconds()
        self.training_stats['training_time_seconds'] = round(duration, 2)

        logger.info(f"✓ Modelo entrenado en {duration:.2f} segundos")
        logger.info(f"  Estimadores: {self.n_estimators}")

        # Importancia de características
        importances = pd.Series(
            self.model.feature_importances_, index=self.X_train.columns
        ).sort_values(ascending=False)

        logger.info("\n  Top 5 características más importantes:")
        for feat, importance in importances.head(5).items():
            logger.info(f"    {feat}: {importance:.4f}")

    def evaluate_model(self):
        """Evalúa el modelo entrenado sobre el conjunto de prueba"""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUACIÓN DEL MODELO")
        logger.info("=" * 80)

        self.y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average='weighted', zero_division=0)

        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")

        report_dict = classification_report(
            self.y_test, self.y_pred, output_dict=True, zero_division=0
        )
        report_text = classification_report(self.y_test, self.y_pred, zero_division=0)

        logger.info("\nClassification Report:\n" + report_text)

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'train_records': self.training_stats['train_records'],
            'test_records': self.training_stats['test_records'],
            'features_used': self.training_stats['features_used'],
            'model_params': {
                'n_estimators': self.n_estimators,
                'test_size': self.test_size,
                'random_state': self.random_state
            }
        }

        return metrics, report_dict, report_text

    def save_confusion_matrix(self):
        """Genera y guarda la matriz de confusión como imagen"""
        logger.info("\n--- Generando Matriz de Confusión ---")

        labels = sorted(self.y_test.unique())
        cm = confusion_matrix(self.y_test, self.y_pred, labels=labels)

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusión - Clasificación de Popularidad Spotify')
        plt.tight_layout()
        plt.savefig(self.confusion_matrix_file, dpi=150)
        plt.close()

        logger.info(f"✓ Matriz de confusión guardada en: {self.confusion_matrix_file}")

    def save_artifacts(self, metrics, report_dict, report_text):
        """Guarda modelo, classification report y métricas en disco"""
        logger.info("\n--- Guardando Artefactos del Modelo ---")

        # 1. Modelo entrenado
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.training_stats['features_used']
        }, self.model_file)
        logger.info(f"✓ Modelo guardado en: {self.model_file}")

        # 2. Classification report (texto)
        with open(self.classification_report_txt, 'w') as f:
            f.write(report_text)
        logger.info(f"✓ Classification report (texto) guardado en: {self.classification_report_txt}")

        # 3. Classification report (JSON)
        with open(self.classification_report_json, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info(f"✓ Classification report (JSON) guardado en: {self.classification_report_json}")

        # 4. Métricas principales
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"✓ Métricas guardadas en: {self.metrics_file}")

    def save_dashboard_data(self):
        """Genera el dataset final que consumirá el dashboard"""
        logger.info("\n--- Generando Datos para el Dashboard ---")

        dashboard_df = self.df.copy()

        # Agregar predicciones del modelo sobre todo el dataset (para exploración en el dashboard)
        feature_columns = self.training_stats['features_used']
        dashboard_df['predicted_popularity_category'] = self.model.predict(dashboard_df[feature_columns])

        dashboard_df.to_csv(self.dashboard_file, index=False, encoding='utf-8')
        file_size_mb = self.dashboard_file.stat().st_size / 1024 / 1024

        logger.info(f"✓ Datos del dashboard guardados en: {self.dashboard_file}")
        logger.info(f"  Registros: {len(dashboard_df):,}")
        logger.info(f"  Tamaño: {file_size_mb:.2f} MB")

        return True

    def load_to_mysql(self, metrics):
        """Carga los resultados del pipeline en la base de datos MySQL"""
        logger.info("\n--- Cargando Resultados en MySQL ---")

        db_host = os.getenv('DB_HOST', 'mysql')
        db_port = os.getenv('DB_PORT', '3306')
        db_user = os.getenv('DB_USER', 'pipeline_user')
        db_password = os.getenv('DB_PASSWORD', 'pipelinepass')
        db_name = os.getenv('DB_NAME', 'spotify_ml_db')

        try:
            engine = create_engine(
                f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
                connect_args={'connect_timeout': 10}
            )

            # 1. Insertar tracks con predicciones
            logger.info("  Insertando tracks en spotify_tracks...")
            track_cols = [
                'track_id', 'artists', 'album_name', 'track_name', 'popularity',
                'duration_ms', 'duration_min', 'explicit', 'danceability', 'energy',
                'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
                'track_genre', 'track_genre_grouped', 'popularity_category'
            ]
            tracks_df = self.df[track_cols].copy()
            tracks_df['explicit'] = tracks_df['explicit'].astype(int)
            tracks_df = tracks_df.where(pd.notna(tracks_df), None)
            tracks_df.to_sql('spotify_tracks', engine, if_exists='append', index=False, method='multi')
            logger.info(f"  ✓ {len(tracks_df)} tracks insertados")

            # 2. Insertar métricas del modelo
            logger.info("  Insertando métricas en model_metrics...")
            metrics_row = pd.DataFrame([{
                'execution_timestamp': datetime.now(),
                'model_type': 'RandomForestClassifier',
                'test_size': self.test_size,
                'random_state': self.random_state,
                'n_estimators': self.n_estimators,
                'accuracy': metrics['accuracy'],
                'precision_weighted': metrics['precision_weighted'],
                'recall_weighted': metrics['recall_weighted'],
                'f1_weighted': metrics['f1_weighted'],
                'train_records': self.training_stats['train_records'],
                'test_records': self.training_stats['test_records'],
                'features_used': json.dumps(self.training_stats['features_used']),
                'training_duration_s': self.training_stats['training_time_seconds']
            }])
            metrics_row.to_sql('model_metrics', engine, if_exists='append', index=False)
            logger.info("  ✓ Métricas insertadas")

            # Obtener el metric_id insertado
            with engine.connect() as conn:
                result = conn.execute(text("SELECT LAST_INSERT_ID()"))
                metric_id = result.scalar()

            # 3. Insertar predicciones del test set
            logger.info("  Insertando predicciones en predictions...")
            test_indices = self.X_test.index
            probas = self.model.predict_proba(self.X_test)
            confidence = probas.max(axis=1)

            preds_df = pd.DataFrame({
                'track_id': self.df.loc[test_indices, 'track_id'].values,
                'execution_timestamp': datetime.now(),
                'actual_category': self.y_test.values,
                'predicted_category': self.y_pred,
                'confidence_score': confidence,
                'metric_id': metric_id
            })
            preds_df.to_sql('predictions', engine, if_exists='append', index=False, method='multi')
            logger.info(f"  ✓ {len(preds_df)} predicciones insertadas")

            engine.dispose()
            return True

        except Exception as e:
            logger.warning(f"  ⚠ No se pudieron cargar datos en MySQL: {e}")
            logger.warning("  Los resultados se guardaron en archivos locales.")
            return False

    def run(self):
        """Ejecuta el proceso completo de entrenamiento, evaluación y carga"""
        try:
            start_time = datetime.now()
            logger.info(f"Inicio del proceso: {start_time}")

            # 1. Cargar datos validados
            if not self.load_data():
                return False

            # 2. Ingeniería de características final
            X, y = self.engineer_features()

            # 3. División entrenamiento/prueba
            self.split_data(X, y)

            # 4. Entrenamiento del modelo
            self.train_model()

            # 5. Evaluación del modelo
            metrics, report_dict, report_text = self.evaluate_model()

            # 6. Matriz de confusión
            self.save_confusion_matrix()

            # 7. Guardar modelo, reporte y métricas
            self.save_artifacts(metrics, report_dict, report_text)

            # 8. Generar datos para el dashboard
            self.save_dashboard_data()

            # 9. Cargar resultados en MySQL
            self.load_to_mysql(metrics)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PROCESO DE ENTRENAMIENTO Y CARGA COMPLETADO EXITOSAMENTE")
            logger.info("=" * 80)
            logger.info(f"Tiempo total: {duration:.2f} segundos")
            logger.info(f"Log guardado en: {log_file}")

            logger.info("\nResumen Final:")
            logger.info(f"  Accuracy del modelo: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score (weighted): {metrics['f1_weighted']:.4f}")

            if metrics['accuracy'] < 0.6:
                logger.warning(f"\n⚠ ADVERTENCIA: Accuracy ({metrics['accuracy']:.2f}) por debajo del umbral esperado (0.60)")

            return True

        except Exception as e:
            logger.error("\n" + "=" * 80)
            logger.error("ERROR EN EL PROCESO DE ENTRENAMIENTO Y CARGA")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}", exc_info=True)
            return False


def main():
    """Función principal"""
    trainer = SpotifyModelTrainer()
    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
