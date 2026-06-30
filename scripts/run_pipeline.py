"""
Script Orquestador del Pipeline Completo - Spotify Tracks Dataset
Ejecuta todas las etapas del pipeline en secuencia

Etapas:
  1. Ingesta de datos
  2. Limpieza, validación de rangos e ingeniería de características
  3. Validación estructural y semántica
  4. Entrenamiento, evaluación y carga de resultados para el dashboard

Autor: Equipo DataOps Spotify
Fecha: Junio 2026
"""

import sys
import json
import time
import logging
import importlib.util
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOGS_DIR / f"pipeline_completo_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def _import_stage_module(module_filename: str, module_name: str):
    """
    Importa dinámicamente un script de etapa cuyo nombre de archivo
    comienza con un número (ej. '01_ingesta.py'), lo cual no es un
    identificador válido para una sentencia 'import' estándar.
    """
    module_path = SCRIPTS_DIR / module_filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Cargar las clases de cada etapa del pipeline
ingesta_module = _import_stage_module("01_ingesta.py", "ingesta_stage")
limpieza_module = _import_stage_module("02_limpieza.py", "limpieza_stage")
validacion_module = _import_stage_module("03_validacion.py", "validacion_stage")
carga_module = _import_stage_module("04_carga.py", "carga_stage")

SpotifyDataIngestion = ingesta_module.SpotifyDataIngestion
SpotifyDataCleaner = limpieza_module.SpotifyDataCleaner
SpotifyDataValidator = validacion_module.SpotifyDataValidator
SpotifyModelTrainer = carga_module.SpotifyModelTrainer


def run_pipeline():
    """Ejecuta el pipeline completo"""
    pipeline_start = datetime.now()
    stage_results = {}

    banner = """
╔══════════════════════════════════════════════════════════════╗
║        PIPELINE DE MACHINE LEARNING - SPOTIFY TRACKS         ║
║        Clasificación Supervisada de Popularidad               ║
║        Equipo: DataOps Spotify                                ║
╚══════════════════════════════════════════════════════════════╝
"""
    logger.info(banner)
    logger.info(f"Inicio del pipeline: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    # ──────────────────────────────────────────────
    # ETAPA 1: Ingesta
    # ──────────────────────────────────────────────
    logger.info("\n▶ ETAPA 1/4 — INGESTA DE DATOS")
    t0 = time.time()
    try:
        ingestion = SpotifyDataIngestion()
        ok = ingestion.run()
        stage_results['ingesta'] = {
            'status': 'SUCCESS' if ok else 'FAILED',
            'duration_seconds': round(time.time() - t0, 2)
        }
        if not ok:
            logger.error("❌ Etapa 1 falló. Pipeline detenido.")
            _save_summary(stage_results, pipeline_start)
            return False
        logger.info(f"✓ Etapa 1 completada en {stage_results['ingesta']['duration_seconds']}s")
    except Exception as e:
        logger.error(f"❌ Error inesperado en Etapa 1: {e}", exc_info=True)
        stage_results['ingesta'] = {'status': 'ERROR', 'error': str(e)}
        _save_summary(stage_results, pipeline_start)
        return False

    # ──────────────────────────────────────────────
    # ETAPA 2: Limpieza y Transformación
    # ──────────────────────────────────────────────
    logger.info("\n▶ ETAPA 2/4 — LIMPIEZA, TRANSFORMACIÓN E INGENIERÍA DE CARACTERÍSTICAS")
    t0 = time.time()
    try:
        cleaner = SpotifyDataCleaner()
        ok = cleaner.run()
        stage_results['limpieza'] = {
            'status': 'SUCCESS' if ok else 'FAILED',
            'duration_seconds': round(time.time() - t0, 2)
        }
        if not ok:
            logger.error("❌ Etapa 2 falló. Pipeline detenido.")
            _save_summary(stage_results, pipeline_start)
            return False
        logger.info(f"✓ Etapa 2 completada en {stage_results['limpieza']['duration_seconds']}s")
    except Exception as e:
        logger.error(f"❌ Error inesperado en Etapa 2: {e}", exc_info=True)
        stage_results['limpieza'] = {'status': 'ERROR', 'error': str(e)}
        _save_summary(stage_results, pipeline_start)
        return False

    # ──────────────────────────────────────────────
    # ETAPA 3: Validación
    # ──────────────────────────────────────────────
    logger.info("\n▶ ETAPA 3/4 — VALIDACIÓN ESTRUCTURAL Y SEMÁNTICA")
    t0 = time.time()
    try:
        validator = SpotifyDataValidator()
        ok = validator.run()
        stage_results['validacion'] = {
            'status': 'SUCCESS' if ok else 'FAILED',
            'duration_seconds': round(time.time() - t0, 2)
        }
        if not ok:
            logger.error("❌ Etapa 3 falló. Pipeline detenido.")
            _save_summary(stage_results, pipeline_start)
            return False
        logger.info(f"✓ Etapa 3 completada en {stage_results['validacion']['duration_seconds']}s")
    except Exception as e:
        logger.error(f"❌ Error inesperado en Etapa 3: {e}", exc_info=True)
        stage_results['validacion'] = {'status': 'ERROR', 'error': str(e)}
        _save_summary(stage_results, pipeline_start)
        return False

    # ──────────────────────────────────────────────
    # ETAPA 4: Entrenamiento, Evaluación y Carga de Resultados
    # ──────────────────────────────────────────────
    logger.info("\n▶ ETAPA 4/4 — ENTRENAMIENTO, EVALUACIÓN Y CARGA DE RESULTADOS")
    t0 = time.time()
    try:
        trainer = SpotifyModelTrainer()
        ok = trainer.run()
        stage_results['carga'] = {
            'status': 'SUCCESS' if ok else 'FAILED',
            'duration_seconds': round(time.time() - t0, 2)
        }
        if not ok:
            logger.error("❌ Etapa 4 falló.")
            _save_summary(stage_results, pipeline_start)
            return False
        logger.info(f"✓ Etapa 4 completada en {stage_results['carga']['duration_seconds']}s")
    except Exception as e:
        logger.error(f"❌ Error inesperado en Etapa 4: {e}", exc_info=True)
        stage_results['carga'] = {'status': 'ERROR', 'error': str(e)}
        _save_summary(stage_results, pipeline_start)
        return False

    # ──────────────────────────────────────────────
    # RESUMEN FINAL
    # ──────────────────────────────────────────────
    _save_summary(stage_results, pipeline_start)
    return True


def _save_summary(stage_results: dict, pipeline_start: datetime):
    """Imprime y guarda el resumen final del pipeline"""
    total_duration = (datetime.now() - pipeline_start).total_seconds()

    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN DEL PIPELINE")
    logger.info("=" * 70)

    all_ok = True
    for stage, result in stage_results.items():
        icon = "✓" if result['status'] == 'SUCCESS' else "❌"
        dur = result.get('duration_seconds', 'N/A')
        logger.info(f"  {icon} {stage.upper():<15} → {result['status']:<10} ({dur}s)")
        if result['status'] != 'SUCCESS':
            all_ok = False

    logger.info(f"\n  Duración total: {total_duration:.2f} segundos")
    logger.info(f"  Estado final  : {'✓ PIPELINE EXITOSO' if all_ok else '❌ PIPELINE CON ERRORES'}")
    logger.info("=" * 70)

    summary_file = LOGS_DIR / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        'pipeline_start': pipeline_start.isoformat(),
        'pipeline_end': datetime.now().isoformat(),
        'total_duration_seconds': round(total_duration, 2),
        'overall_status': 'SUCCESS' if all_ok else 'FAILED',
        'stages': stage_results
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Resumen guardado en: {summary_file}")


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
