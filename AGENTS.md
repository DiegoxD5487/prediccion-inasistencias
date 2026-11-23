# AGENTS.md

## Contexto del proyecto
- Prototipo académico de predicción de inasistencias a citas médicas, con dataset brasileño de Kaggle: `data/KaggleV2-May-2016.csv` (110.527 × 14).
- Objetivo `No-show`: `Yes`/`True` significa que el paciente **no asistió**; es la clase positiva (1). Desbalance ≈ 79.8% / 20.2%.
- `data/citas_medicas_preprocesadas.csv` es el dataset limpio (110.521 × 16, 62.299 pacientes) y la entrada real de 03, 04 y 05. No se regenera automáticamente.
- Pipeline secuencial: `01_Perfilamiento` (calidad) → `02_Preprocesamiento` (limpieza, features, anonimización) → `03_EDA` → `04_Baselines` → `05_Modelamiento`.
- El artefacto vigente está en `models/mejor_modelo_lgbm.joblib`; se reproduce con `scripts/entrenar_modelo.py`. `scripts/experimento_modelo.py` concentra la ingeniería de features y la evaluación reutilizada por los demás scripts.
- `T01`–`T04` documentan negocio, datos y KPIs. `T02` fija los umbrales que importan: recall ≥ 0.65 y Lift@20 ≥ 1.50. `INFORME_MODELO_ML.md` describe detalladamente el proceso técnico que llevó a obtener el modelo y contiene al final instrucciones preliminares que podrían servir para el próximo paso de integración en el prototipo.
- Mantener comentarios, títulos y prosa de los notebooks en español.

## Cómo trabajar con los notebooks
- No usar `read`/`edit`/`write` nativos sobre `.ipynb`; usar la skill `gestor-ipynb`:
  ```powershell
  python .opencode/skills/gestor-ipynb/ipynb_tool.py read "notebooks/05_Modelamiento.ipynb"
  ```
- Para `ipynb_tool.py`, `python` del PATH funciona porque el script solo usa la librería estándar. No usar `py`: el lanzador no encuentra ninguna instalación. Para datos/modelos, usar siempre `.venv\Scripts\python.exe`.
- Ejecutar siempre desde la raíz del repo: todas las cargas usan `Path("data")` y otro cwd rompe los `assert`. `ipynb_tool.py run` fija internamente el directorio de trabajo en la raíz (`path='.'`).
- Después de `edit`/`add`, borrar `_temp_cell.txt` aunque el comando falle.

## Entorno de ejecución
- Usar `.venv\Scripts\python.exe` para datos y modelos. Ese venv (Python 3.12.4, ruta reparada en este equipo) ya tiene pandas, numpy, scikit-learn, lightgbm, optuna, catboost, joblib, seaborn, nbconvert e ipykernel.
- Usar siempre `.venv\Scripts\python.exe` para entrenar. El entorno copiado tenía una ruta a otro equipo y fue reparado; la reproducción completa quedó registrada en `docs/evidencias/reproduccion_entrenamiento_2026-11-23.json`.
- `requirements.txt` fija las versiones usadas.
- `ipynb_tool.py run` ejecuta los notebooks localmente usando el kernel de `.venv` y genera todas sus salidas y gráficos de forma reproducible.

## Trampas conocidas de los notebooks
- `02_Preprocesamiento` termina con celdas exclusivas de Colab (`from google.colab import files`, `files.download(...)`) y guarda el CSV en el cwd, no en `data/`. Re-ejecutarlo tal cual crea un CSV suelto en la raíz y luego falla. Para regenerar, usar `df.to_csv(Path("data") / "citas_medicas_preprocesadas.csv", index=False)` y eliminar la celda de descarga.
- `04_Baselines` ya tiene corregidas las importaciones de `plt` y `sns` en su celda 1, ejecutando sin errores de forma aislada.
- `05_Modelamiento` depende de `lightgbm`, `catboost`, `optuna` y `joblib`, y contiene `!pip install`; el venv ya los tiene, pero no asumir que están en un entorno nuevo.
- El CSV no conserva dtypes: 03, 04 y 05 vuelven a convertir `ScheduledDay`/`AppointmentDay` con `pd.to_datetime`.
- Los acentos se ven como mojibake en las salidas guardadas; es codificación de la salida, no de los datos.

## Modelo y evaluación
- El modelo vigente es `models/mejor_modelo_lgbm.joblib`, generado por `.venv\Scripts\python.exe scripts\entrenar_modelo.py`. Sus métricas y umbrales están en `models/metricas.json` y `models/README.md`.
- Protocolo principal: `GroupKFold(5)` agrupado por `PatientId` (pacientes de validación no vistos). No reemplazarlo por `train_test_split`/`KFold` sin justificarlo.
- `04` es solo baseline: split aleatorio estratificado, sin `PatientId`, `Neighbourhood` ni fechas. Sus métricas no son comparables con `05`.
- Baseline notebook 05 reproducido en este entorno (incluye `PatientId`): ROC-AUC 0.7364, PR-AUC 0.3797, F1 0.4490, Lift@20 1.9311.
- Modelo mejorado OOF: ROC-AUC **0.7535**, PR-AUC **0.4103**, F1 0.4628, recall 0.7401, Lift@20 **2.0140**, recall@20 0.4028.
- Validación temporal secundaria (corte `2016-05-25`): ROC-AUC 0.7303 vs 0.7254; Lift@20 1.8876 vs 1.8417.
- La mejora viene de features temporales, historial causal del paciente (solo citas anteriores) y Optuna. `scripts/experimento_modelo.py` contiene la ingeniería y `scripts/optimizar_modelo.py` la búsqueda.
- `is_unbalance=True` empeoraba el ranking en este dataset; no volver a activarlo sin medir.
- Umbrales: 0.2219 para máximo F1 y 0.2508 para recall ≥ 0.65. Se eligen con predicciones OOF, nunca con entrenamiento. El modelo final se reentrena con el 100% de los datos.
- Al predecir, fijar las categorías con `payload["categorical_categories"]`; un `astype("category")` sobre un subconjunto de barrios cambia los códigos y produce predicciones incorrectas.
- El notebook `05` refleja ahora el modelo mejorado: las curvas ROC y PR y los umbrales se calculan sobre predicciones Out-Of-Fold (OOF), descartando la antigua métrica sobreentrenada (0.817 / 0.512) y exportando el artefacto a `models/mejor_modelo_lgbm.joblib`.
- `Lift@20` ya se calcula en `scripts/`. No optimizar accuracy: la clase mayoritaria da 79.8% sin detectar inasistencias.
- Bajo `GroupKFold`, las categorías de `PatientId` no aparecen en validación; el historial del paciente debe construirse solo con citas anteriores; nunca usar información futura.
- `SMS_received` es una intervención previa a la cita. Si el scoring ocurre antes de enviar recordatorios, reentrenar sin esa variable para evitar fuga.
