# Modelo de predicción de inasistencias

Modelo LightGBM mejorado para predecir `No-show` (`True` = el paciente no asistió).
Fue entrenado con `data/citas_medicas_preprocesadas.csv` y exportado por [05_Modelamiento.ipynb](../notebooks/05_Modelamiento.ipynb). El conjunto contiene 110.521 citas, 62.298 pacientes y 22.314 inasistencias.

## Archivos

| Archivo | Contenido |
|---|---|
| `mejor_modelo_lgbm.joblib` | Diccionario con el modelo, features, umbrales y metadatos. |
| `metricas.json` | Métricas reproducibles y comparación contra el baseline del notebook 05. |
| `mejores_hiperparametros.json` | Hiperparámetros elegidos por Optuna. |

## Resultados

Validación principal: `GroupKFold(5)` agrupado por `PatientId` (pacientes de
validación no vistos en entrenamiento). El umbral se eligió con las predicciones
OOF, no con las predicciones de entrenamiento.

| Métrica OOF | Baseline formal | Modelo final | Diferencia |
| --- | ---: | ---: | ---: |
| ROC-AUC | 0.7389 | 0.7535 | +0.0146 |
| PR-AUC (precisión promedio) | 0.3841 | 0.4103 | +0.0262 |
| F1 | 0.4499 | 0.4628 | +0.0129 |
| Recall | 0.7602 | 0.7401 | -0.0200 |
| Precisión | 0.3195 | 0.3366 | +0.0171 |
| Lift@20 | 1.9331 | 2.0140 | +0.0809 |
| Recall@20 | 0.3866 | 0.4028 | +0.0162 |

Con recall mínimo de 0.65, la precisión del baseline es 0.3403 y la del modelo final es 0.3536. Lift@20 y Recall@20 usan las primeras `round(0.20 × n)` citas del ranking.

Validación temporal secundaria (entrenar con citas hasta `2016-05-25` y probar
con las posteriores):

| Métrica temporal | Baseline formal | Modelo final |
| --- | ---: | ---: |
| ROC-AUC | 0.7184 | 0.7303 |
| PR-AUC (precisión promedio) | 0.3189 | 0.3458 |
| Lift@20 | 1.7798 | 1.8876 |
| Recall@20 | 0.3560 | 0.3775 |

## Diferencias respecto del baseline formal

1. **Features temporales adicionales:** día de la semana del agendamiento, hora,
   semana de la cita, días desde el inicio del dataset, cita same-day, edad
   agrupada, adulto mayor, discapacidad y comorbilidad en adultos mayores.
2. **Historial causal del paciente:** `PriorVisits`, `PriorNoShows`,
   `PriorNoShowRate`, `HasHistory`, `DaysSinceLastVisit` y `LastVisitNoShow`.
   Los recuentos usan citas de fecha estrictamente anterior. El suavizado de la tasa individual conserva la prevalencia del CSV completo.
3. **Hiperparámetros ajustados con Optuna:** 1050 árboles, `learning_rate=0.010458503793355878`,
   118 hojas, `max_depth=12`, sin `is_unbalance` (el peso de clases empeoraba
   el ranking). La búsqueda opcional optimiza PR-AUC con un máximo de 40 ensayos y tres pliegues por paciente.
4. **Umbral elegido con OOF**, no con el entrenamiento. Se guardan dos:
   - `threshold` = 0.2219, máximo F1 (recall 0.7401, precision 0.3366).
   - `threshold_recall_065` = 0.2508, cumple recall ≥ 0.65 con mejor precision
     (recall 0.6500, precision 0.3536). **Recomendado para el caso de negocio.**

## Contrato de inferencia

| Contenido | Claves del diccionario `joblib` |
| --- | --- |
| Estimador, objetivo y clase positiva | `model`, `target`, `positive_class` |
| Variables ordenadas y niveles categóricos | `features`, `numeric_features`, `categorical_features`, `categorical_categories` |
| Historial y valores iniciales | `global_no_show_rate`, `feature_defaults`, `history_mode` |
| Políticas de contacto y resultados OOF | `threshold`, `threshold_recall_065`, `thresholds`, `metrics_oof`, `baseline_oof` |
| Entrenamiento y prueba temporal | `params`, `trained_at`, `training_rows`, `training_patients`, `lightgbm_version`, `temporal`, `nota` |

Se guardan los cortes OOF a precisión completa. `threshold` vale 0.22192507983773843 y `threshold_recall_065` vale 0.25082511960234266. La clave `metrics_oof` resume el modelo al umbral F1. `thresholds` recoge precisión, recall y F1 de cada política. En `temporal` se guardan las métricas de los dos estimadores con el corte de fecha utilizado.

## Uso

```python
import joblib
import pandas as pd

payload = joblib.load("models/mejor_modelo_lgbm.joblib")
modelo = payload["model"]
features = payload["features"]
categoricas = payload["categorical_features"]
umbral = payload["threshold_recall_065"]

# X debe tener exactamente las columnas de payload["features"].
for col in categoricas:
    X[col] = pd.Categorical(
        X[col], categories=payload["categorical_categories"][col]
    )

probabilidad = modelo.predict_proba(X)[:, 1]
priorizar = probabilidad >= umbral
```

Es importante fijar `categorical_categories`: si se usa `astype("category")` con
un subconjunto de barrios, pandas asigna códigos distintos a los del entrenamiento
y las predicciones quedan mal.

Para una cita nueva sin historial, usar estos valores por defecto:
`PriorVisits=0`, `PriorNoShows=0`, `PriorNoShowRate=payload["global_no_show_rate"]`, `HasHistory=0`,
`DaysSinceLastVisit=NaN`, `LastVisitNoShow=NaN`.

## Reproducir

Consultar los comandos de ejecución desde la raíz en la [guía de reproducción](../README.md).

La ejecución del 26 de noviembre contrastó los 1.050 árboles, el orden y niveles de las variables y las probabilidades de 1.000 citas antes de sustituir el archivo. [Evidencia de equivalencia](../docs/evidencias/refactorizacion_notebooks_2026-11-26.json)

## Limitaciones

La reproducción completa del 23/nov/2026 se ejecutó con Python 3.12.4 en una salida aislada. Incluyó baseline y modelo mejorado con cinco pliegues agrupados, control temporal y entrenamiento final. Todas las métricas coincidieron dentro de 1e-9 y los artefactos vigentes conservaron sus huellas. [Evidencia y versiones](../docs/evidencias/reproduccion_entrenamiento_2026-11-23.json).

- El historial se calcula con citas anteriores a la fecha de la cita (modo `cita`).
  Si el scoring se hace al momento de agendar, usar `modo_historial="agendamiento"`;
  ese escenario también mejora al baseline, pero dispone de menos historial.
- `PatientId` identifica el historial y los grupos de validación. El baseline formal de 17 variables y el modelo final de 38 lo excluyen de sus matrices de entrada.
- El dataset es de Brasil (2016) y no representa necesariamente el contexto chileno.
- `SMS_received` es una intervención previa a la cita. Si el scoring se hace antes
  de enviar recordatorios, esa variable no estaría disponible y habría que
  reentrenar sin ella.
- El modelo prioriza pacientes para revisión humana; no cancela ni reasigna citas.
- Lift@20 y recall se calculan sobre el conjunto de validación; no prueban que
  contactar a los pacientes reduzca la inasistencia.
