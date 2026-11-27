# AGENTS.md

## Datos y fuentes

El proyecto prioriza el contacto con pacientes con riesgo de faltar a una cita. La clase positiva `No-show=True` representa una inasistencia. El archivo crudo tiene 110.527 filas y 14 columnas. El CSV limpio tiene 110.521 filas, 16 columnas, 62.298 pacientes y 22.314 inasistencias.

| Etapa | Fuente y responsabilidad |
| --- | --- |
| 01 Perfilamiento | Audita calidad, fechas y edades del CSV crudo. |
| 02 Preprocesamiento | Limpia y anonimiza los identificadores de forma determinista. Valida el resultado y escribe `data/citas_medicas_preprocesadas.csv`. |
| 03 EDA | Describe patrones en el conjunto limpio. |
| 04 Baselines | Compara Dummy y regresión logística con una partición aleatoria estratificada. Sus resultados son exploratorios. |
| 05 Modelamiento | Define el baseline formal, las 38 variables, la búsqueda Optuna, la evaluación, los umbrales y la exportación. Alimenta `models/mejor_modelo_lgbm.joblib` y `models/metricas.json`. |

Mantener el desarrollo del modelo dentro del notebook 05. Escribir el código explicativo, los comentarios y los textos de los notebooks en español.

## Entorno y ejecución

Usar Python 3.12 y las versiones de `requirements.txt`. Comprobar si existe `.venv\Scripts\python.exe`. Si falta el entorno, crearlo desde la raíz del repositorio e instalar las dependencias.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Seguir los comandos de reproducción en [README.md](README.md) desde la raíz del repositorio. Mantener 02 ejecutable en Windows, sin descargas de Colab. El CSV pierde los tipos de fecha al serializarse. Convertir `ScheduledDay` y `AppointmentDay` tras leerlo.

## Modelo y criterios de revisión

- `PatientId` agrupa los cinco pliegues y permite construir los antecedentes de cada paciente. Se excluye de las entradas a LightGBM tanto en el baseline formal (14 variables numéricas y 3 categóricas) como en el mejorado (33 y 5).
- El historial en modo `cita` usa atenciones estrictamente anteriores y excluye las del mismo día. El suavizado conserva la tasa global del CSV del modelo publicado. Calcularla dentro de cada pliegue sería una revisión del modelo.
- El entrenamiento de referencia reutiliza `models/mejores_hiperparametros.json` con `EJECUTAR_OPTUNA = False`. El modelo usa 1.050 árboles, 118 hojas, profundidad 12, tasa de aprendizaje 0.010458503793355878 y `is_unbalance=False`. La opción de búsqueda usa TPE con semilla 42, tres pliegues por paciente y hasta 40 ensayos limitados por tiempo.
- Comparar los modelos con predicciones fuera de pliegue (OOF). Para Lift@20 y Recall@20 usar las primeras `max(1, round(0.20 * n))` citas ordenadas por riesgo. El control temporal entrena con citas hasta el 25 de mayo de 2016 inclusive y evalúa las posteriores.
- Seleccionar ambos umbrales con las predicciones OOF y guardar su valor completo. Usar `threshold` para máximo F1 y `threshold_recall_065` para máxima precisión entre los cortes con recall ≥ 0.65. La meta M2 exige Lift@20 ≥ 1.50. Para inferencia, respetar el orden de `features` y aplicar `categorical_categories` a cada columna categórica del lote.
- `SMS_received` describe una intervención previa. Puntuar antes de enviar recordatorios requeriría reentrenar y evaluar sin esa variable. La decisión de contactar corresponde al equipo humano.

Los resultados con su precisión completa están en [metricas.json](models/metricas.json). La ficha del artefacto incluye el contrato de inferencia y las comparaciones OOF y temporal. [Ficha técnica](models/README.md)
