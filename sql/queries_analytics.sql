USE spotify_ml_db;

SELECT
    COUNT(*)                          AS total_canciones,
    COUNT(DISTINCT track_genre)       AS generos_distintos,
    ROUND(AVG(popularity), 2)         AS popularidad_promedio,
    ROUND(AVG(danceability), 3)       AS bailabilidad_promedio,
    ROUND(AVG(energy), 3)             AS energia_promedio,
    ROUND(AVG(valence), 3)            AS valencia_promedio
FROM spotify_tracks;

SELECT
    popularity_category,
    COUNT(*)                               AS total,
    ROUND(COUNT(*) / (SELECT COUNT(*) FROM spotify_tracks) * 100, 2) AS pct_total,
    ROUND(AVG(danceability), 3)            AS avg_danceability,
    ROUND(AVG(energy), 3)                  AS avg_energy,
    ROUND(AVG(valence), 3)                 AS avg_valence,
    ROUND(AVG(tempo), 2)                   AS avg_tempo
FROM spotify_tracks
GROUP BY popularity_category
ORDER BY FIELD(popularity_category, 'Baja', 'Media', 'Alta');

SELECT
    track_genre AS genero,
    COUNT(*)                                AS total_canciones,
    ROUND(AVG(popularity), 2)               AS popularidad_promedio,
    ROUND(AVG(danceability), 3)             AS bailabilidad,
    ROUND(AVG(energy), 3)                   AS energia,
    ROUND(AVG(valence), 3)                  AS valencia
FROM spotify_tracks
GROUP BY track_genre
ORDER BY total_canciones DESC
LIMIT 15;

SELECT
    explicit,
    COUNT(*)                               AS total,
    ROUND(AVG(popularity), 2)              AS popularidad_promedio,
    ROUND(AVG(danceability), 3)            AS bailabilidad,
    ROUND(AVG(energy), 3)                  AS energia
FROM spotify_tracks
GROUP BY explicit;

SELECT
    CASE
        WHEN duration_min < 2 THEN '< 2 min'
        WHEN duration_min BETWEEN 2 AND 3 THEN '2-3 min'
        WHEN duration_min BETWEEN 3 AND 4 THEN '3-4 min'
        WHEN duration_min BETWEEN 4 AND 5 THEN '4-5 min'
        ELSE '> 5 min'
    END AS duracion_rango,
    COUNT(*) AS total,
    ROUND(AVG(popularity), 2) AS popularidad_promedio
FROM spotify_tracks
GROUP BY duracion_rango
ORDER BY FIELD(duracion_rango, '< 2 min', '2-3 min', '3-4 min', '4-5 min', '> 5 min');

SELECT
    ROUND(AVG(popularity), 2) AS popularidad_promedio,
    ROUND(AVG(danceability), 3) AS bailabilidad_promedio,
    ROUND(AVG(energy), 3) AS energia_promedio,
    ROUND(AVG(speechiness), 4) AS speechiness_promedio,
    ROUND(AVG(acousticness), 4) AS acousticness_promedio,
    ROUND(AVG(instrumentalness), 4) AS instrumentalness_promedio,
    ROUND(AVG(liveness), 4) AS liveness_promedio,
    ROUND(AVG(valence), 3) AS valence_promedio,
    ROUND(AVG(tempo), 2) AS tempo_promedio
FROM spotify_tracks;

SELECT
    t.track_genre_grouped,
    COUNT(*) AS total,
    ROUND(AVG(m.accuracy), 4) AS avg_accuracy,
    ROUND(AVG(m.f1_weighted), 4) AS avg_f1
FROM spotify_tracks t
CROSS JOIN (SELECT accuracy, f1_weighted FROM model_metrics ORDER BY metric_id DESC LIMIT 1) m
GROUP BY t.track_genre_grouped
ORDER BY total DESC;

SELECT
    execution_timestamp,
    model_type,
    ROUND(accuracy, 4) AS accuracy,
    ROUND(f1_weighted, 4) AS f1_score,
    train_records,
    test_records,
    ROUND(training_duration_s, 2) AS duracion_seg
FROM model_metrics
ORDER BY execution_timestamp DESC;

SELECT
    actual_category,
    predicted_category,
    COUNT(*) AS total
FROM predictions
GROUP BY actual_category, predicted_category
ORDER BY actual_category, predicted_category;
