CREATE DATABASE IF NOT EXISTS spotify_ml_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE spotify_ml_db;

CREATE TABLE IF NOT EXISTS spotify_tracks (
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

) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Canciones de Spotify procesadas por el pipeline ML';

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id             INT            AUTO_INCREMENT PRIMARY KEY,
    execution_timestamp   DATETIME       NOT NULL,
    model_type            VARCHAR(100)   NOT NULL,
    test_size             FLOAT,
    random_state          INT,
    n_estimators          INT,
    accuracy              FLOAT,
    precision_weighted    FLOAT,
    recall_weighted       FLOAT,
    f1_weighted           FLOAT,
    train_records         INT,
    test_records          INT,
    features_used         JSON,
    training_duration_s   FLOAT,
    created_at            DATETIME       DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_timestamp   (execution_timestamp),
    INDEX idx_model_type  (model_type)

) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Métricas de rendimiento de modelos entrenados';

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         INT            AUTO_INCREMENT PRIMARY KEY,
    track_id              VARCHAR(50),
    execution_timestamp   DATETIME       NOT NULL,
    actual_category       VARCHAR(10),
    predicted_category    VARCHAR(10),
    confidence_score      FLOAT          COMMENT 'Probabilidad de la predicción (opcional)',
    metric_id             INT,

    INDEX idx_track       (track_id),
    INDEX idx_prediction  (predicted_category),
    INDEX idx_ts          (execution_timestamp),
    CONSTRAINT fk_metric
        FOREIGN KEY (metric_id)
        REFERENCES model_metrics(metric_id)
        ON DELETE SET NULL

) ENGINE=InnoDB
  DEFAULT CHARSET=utf8mb4
  COLLATE=utf8mb4_unicode_ci
  COMMENT='Predicciones del modelo sobre el dataset';

SELECT
    TABLE_NAME       AS `Tabla`,
    TABLE_ROWS       AS `Filas aprox.`,
    TABLE_COMMENT    AS `Descripción`
FROM information_schema.TABLES
WHERE TABLE_SCHEMA = 'spotify_ml_db'
ORDER BY TABLE_NAME;
