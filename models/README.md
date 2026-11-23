# Modelo de predicción de inasistencias

Modelo LightGBM mejorado para predecir `No-show` (`True` = el paciente no asistió).
Fue entrenado con `data/citas_medicas_preprocesadas.csv` y exportado con
`scripts/entrenar_modelo.py`.

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

| Métrica | Baseline notebook 05 | Modelo mejorado | Diferencia |
|---|---:|---:|---:|
| ROC-AUC | 0.7364 | **0.7535** | +0.0171 |
| PR-AUC | 0.3797 | **0.4103** | +0.0306 |
| F1 | 0.4490 | **0.4628** | +0.0137 |
| Recall | 0.7565 | 0.7401 | -0.0164 |
| Precision | 0.3193 | 0.3366 | +0.0173 |
| Lift@20 | 1.9311 | **2.0140** | +0.0829 |
| Recall@20 | 0.3862 | **0.4028** | +0.0166 |

Comparación al mismo recall de 0.65: la precision sube de 0.3397 (baseline) a
0.3536 (mejorado), es decir, se priorizan menos falsos positivos manteniendo la
cobertura exigida por el KPI.

Validación temporal secundaria (entrenar con citas hasta `2016-05-25` y probar
con las posteriores):

| Métrica | Baseline | Mejorado |
|---|---:|---:|
| ROC-AUC | 0.7254 | **0.7303** |
| PR-AUC | 0.3362 | **0.3458** |
| Lift@20 | 1.8417 | **1.8876** |

## Qué cambió respecto del notebook 05

1. **Features temporales adicionales:** día de la semana del agendamiento, hora,
   semana de la cita, días desde el inicio del dataset, cita same-day, edad
   agrupada, adulto mayor, discapacidad y comorbilidad en adultos mayores.
2. **Historial causal del paciente:** `PriorVisits`, `PriorNoShows`,
   `PriorNoShowRate`, `HasHistory`, `DaysSinceLastVisit` y `LastVisitNoShow`.
   Solo usan citas cuya fecha de atención es anterior a la cita actual; nunca
   usan información futura.
3. **Hiperparámetros ajustados con Optuna:** 1050 árboles, `learning_rate=0.0105`,
   118 hojas, `max_depth=12`, sin `is_unbalance` (el peso de clases empeoraba
   el ranking). La búsqueda optimizó PR-AUC.
4. **Umbral elegido con OOF**, no con el entrenamiento. Se guardan dos:
   - `threshold` = 0.2219, máximo F1 (recall 0.7401, precision 0.3366).
   - `threshold_recall_065` = 0.2508, cumple recall ≥ 0.65 con mejor precision
     (recall 0.6500, precision 0.3536). **Recomendado para el caso de negocio.**

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
`PriorVisits=0`, `PriorNoShows=0`, `PriorNoShowRate=0.2019`, `HasHistory=0`,
`DaysSinceLastVisit=NaN`, `LastVisitNoShow=NaN`.

## Reproducir

```powershell
.venv\Scripts\python.exe scripts\entrenar_modelo.py
```

## Limitaciones

La reproducción completa del 23/nov/2026 se ejecutó con Python 3.12.4 en una salida aislada. Incluyó baseline y modelo mejorado con cinco pliegues agrupados, control temporal y entrenamiento final. Todas las métricas coincidieron dentro de 1e-9 y los artefactos vigentes conservaron sus huellas. [Evidencia y versiones](../docs/evidencias/reproduccion_entrenamiento_2026-11-23.json).

- El historial se calcula con citas anteriores a la fecha de la cita (modo `cita`).
  Si el scoring se hace al momento de agendar, usar `modo_historial="agendamiento"`;
  ese escenario también mejora al baseline, pero dispone de menos historial.
- `PatientId` no se usa como feature: solo agrupa `GroupKFold` para evitar fuga
  entre citas del mismo paciente.
- El dataset es de Brasil (2016) y no representa necesariamente el contexto chileno.
- `SMS_received` es una intervención previa a la cita. Si el scoring se hace antes
  de enviar recordatorios, esa variable no estaría disponible y habría que
  reentrenar sin ella.
- El modelo prioriza pacientes para revisión humana; no cancela ni reasigna citas.
- Lift@20 y recall se calculan sobre el conjunto de validación; no prueban que
  contactar a los pacientes reduzca la inasistencia.
