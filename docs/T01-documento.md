# Objetivo y alcance:
Creación de una demo/prototipo basada en la utilización de un modelo de ML que se debe crear, el cual predice inasistencias a citas médicas.
* El prototipo no debe costar dinero de hostear o testear.
* El prototipo debe permitir que el usuario ingrese datos.
* El modelo debe devolver una predicción de en base a los datos ingresados.
* El modelo de ML será exportado desde un archivo .ipynb
* El dataset es de Kaggle, y por ello representa un ejemplo de aplicación de cómo se desarrollaría el proyecto real.

# Integrantes:
* Alfredo Álvarez: encargado de los datos
* Diego Bustamante: encargado del modelo

# Supuestos, riesgos y restricciones:

### **1. Supuestos del proyecto**

| ID | Supuesto                | Descripción                                                                                                            |
| -- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| S1 | Disponibilidad de datos | El dataset público Medical Appointment No Shows de Kaggle será suficiente para desarrollar una demostración académica. |
| S2 | Calidad de datos        | Los registros podrán limpiarse y prepararse para entrenar modelos predictivos.                                         |
| S3 | Capacidad predictiva    | Modelos como Regresión Logística y Random Forest podrán identificar patrones asociados a las inasistencias.            |
| S4 | Priorización            | Se seleccionará el 20% de las citas con mayor riesgo para priorizar los recordatorios.                                 |
| S5 | Efectividad de contacto | Los SMS y el apoyo humano podrían contribuir a reducir las inasistencias, sujeto a validación.                         |
| S6 | Recursos disponibles    | Se contará con herramientas, tiempo y conocimientos técnicos suficientes para desarrollar el prototipo.                |

### **2. Riesgos del proyecto**

| ID | Riesgo                              | Impacto potencial                                                   | Medida de mitigación                                          |
| -- | ----------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- |
| R1 | Datos incompletos o incorrectos     | Predicciones poco confiables.                                       | Limpieza y validación de datos.                               |
| R2 | Sesgos algorítmicos                 | Resultados desiguales entre grupos de pacientes.                    | Evaluar métricas por grupos y revisar sesgos.                 |
| R3 | Fuga de información                 | Métricas artificialmente elevadas y sobreajuste.                    | Separar correctamente entrenamiento y prueba.                 |
| R4 | Vulneración de privacidad           | Exposición de información sensible de pacientes.                    | Control de accesos, cifrado y protección de datos.            |
| R5 | Fallas en SMS                       | Recordatorios no entregados.                                        | Monitorear entregabilidad y disponer de contacto alternativo. |
| R6 | Sobrecostos                         | Superación del presupuesto estimado.                                | Seguimiento financiero y revisión de costos.                  |
| R7 | Transferencia de datos Brasil-Chile | El modelo podría no funcionar adecuadamente en el contexto chileno. | Validar posteriormente con datos locales.                     |
| R8 | Mal uso de las predicciones         | Decisiones injustas sobre las citas médicas.                        | Mantener supervisión y decisiones humanas.                    |

### **3. Restricciones del proyecto**

| ID | Restricción         | Descripción                                                                                                  |
| -- | ------------------- | ------------------------------------------------------------------------------------------------------------ |
| C1 | Fuente de datos     | Se utilizará el dataset público de Kaggle, correspondiente a registros médicos de Brasil.                    |
| C2 | Alcance académico   | El prototipo no representa un sistema hospitalario productivo ni un piloto real autorizado.                  |
| C3 | Tiempo              | El prototipo académico se organiza en un cronograma de 6 días.                                               |
| C4 | Alcance funcional   | El sistema se limita a predecir riesgos y generar un ranking de citas.                                       |
| C5 | Intervención humana | No podrá cancelar, rechazar ni reasignar automáticamente horas médicas.                                      |
| C6 | Privacidad          | No se podrán utilizar datos reales de pacientes sin autorización institucional.                              |
| C7 | Integración         | Los envíos reales de SMS y la conexión con sistemas hospitalarios quedan fuera de la demostración académica. |
| C8 | Validación          | No se podrá afirmar que el modelo reduce las inasistencias sin pruebas y evaluación comparativa.             |
| C9 | Recursos            | El presupuesto y los costos operacionales son estimaciones, no gastos ejecutados.                            |

# Definición de terminado
Se define como terminado aquella tarea que finaliza su desarrollo.

# Tablero Kanban

| Clave | Tarea | Épica | Días planificados | Vencimiento | Responsable sugerido | Trabajo a realizar | Entregable | Criterio de aceptación | Hito de cierre | Depende de |
|---|---|---|---|---|---|---|---|---|---|---|
| KAN-11 | T01 · Inicio y alcance | E1 Inicio y comprensión del negocio | Día 1 | 16/nov/26 | Jefatura de proyecto / equipo | Acordar objetivo, alcance del prototipo, integrantes/roles, riesgos, restricciones y definición de terminado. | Acta breve del sprint y tablero de trabajo | El equipo comparte objetivo, responsables, prioridades y límites. | Alcance aprobado por el equipo | Ninguna |
| KAN-12 | T02 · Comprensión del negocio | E1 Inicio y comprensión del negocio | Día 1 | 16/nov/26 | Equipo + responsable funcional | Explicar el problema de inasistencia, usuarios, valor esperado, etiqueta objetivo y decisión que apoyará el modelo. | Ficha de problema y criterios de éxito | Se distingue predicción de impacto: el modelo no prueba que SMS reduzca ausencias. | Problema y criterios documentados | T01 |
| KAN-13 | T03 · Fuente y diccionario | E2 Datos: fuente, calidad y preparación | Día 1–2 | 17/nov/26 | Data steward / analista de datos | Registrar Kaggle, versión/fecha de descarga, filas y columnas; describir variables y su significado. | Inventario y diccionario de datos básico | Se identifica No-show como etiqueta y se advierte que la fuente corresponde a Brasil. | Dataset y variables comprendidos | T02 |
| KAN-14 | T04 · Perfilamiento de calidad | E2 Datos: fuente, calidad y preparación | Día 2–3 | 18/nov/26 | Analista de datos | Revisar tipos, faltantes, duplicados, edades, fechas incoherentes y codificación de la etiqueta. | Reporte inicial de calidad | Los problemas y decisiones quedan registrados, sin ocultar exclusiones. | Calidad revisada y excepciones documentadas | T03 |
| KAN-15 | T05 · Preparación de datos | E2 Datos: fuente, calidad y preparación | Día 2–3 | 18/nov/26 | Analista de datos / ML | Convertir fechas, crear días de anticipación/día de semana, limpiar y construir pipeline reproducible. | Notebook de preparación y dataset procesado | Las transformaciones se pueden volver a ejecutar; no se usan datos posteriores a la cita. | Pipeline de preparación ejecutable | T04 |
| KAN-16 | T06 · EDA | E3 Análisis exploratorio (EDA) | Día 3–4 | 19/nov/26 | Analista de datos | Explorar balance de clases y distribuciones; generar visualizaciones legibles y comentar hallazgos observados. | Sección EDA en notebook y gráficos | Cada hallazgo se apoya en cifras/gráficos ejecutados; asociación no se presenta como causalidad. | EDA incorporado al notebook | T04 a T05 |
| KAN-17 | T07 · Baseline y partición | E4 Modelado y evaluación | Día 3–4 | 19/nov/26 | Analista ML | Definir train/test, preferir corte temporal cuando sea viable, fijar semilla y construir baseline. | Partición documentada y baseline | Se evita contaminación entre entrenamiento y prueba y se explica el criterio de separación. | Baseline ejecutado y registrado | T05 |
| KAN-18 | T08 · Modelamiento | E4 Modelado y evaluación | Día 4–5 | 20/nov/26 | Analista ML | Entrenar regresión logística y, si el tiempo/recursos lo permiten, comparar con Random Forest u otro modelo simple. | Modelos guardados o reproducibles y tabla comparativa | Se registran parámetros y se puede reproducir la ejecución. | Modelos ejecutados y comparación disponible | T07 |
| KAN-19 | T09 · Evaluación | E4 Modelado y evaluación | Día 4–6 | 21/nov/26 | Analista ML + revisor | Calcular matriz de confusión, precisión, recall y F1; revisar falsos positivos/negativos y diferencias por grupos si hay muestra suficiente. | Informe breve de evaluación y limitaciones | Las métricas provienen de test no usado para ajustar; se informa el tamaño de muestra. | Métricas reproducibles y limitaciones declaradas | T08 |
| KAN-20 | T10 · Priorización top 20% | E5 Ranking y prototipo visual | Día 5–6 | 21/nov/26 | Analista ML / desarrollo | Ordenar las citas de prueba/demostración por score y seleccionar el 20% de mayor riesgo. | Tabla de ranking simulado top 20% | Se documenta fecha/corte, universo y regla de selección; no se generan acciones reales. | Ranking generado y explicado | T09 |
| KAN-21 | T11 · Prototipo visual | E5 Ranking y prototipo visual | Día 5–6 | 21/nov/26 | Desarrollo / equipo | Construir una vista sencilla: resumen, métricas, ranking y flujo sugerido de recordatorio con aprobación humana. | Dashboard, interfaz simple o reporte navegable | La demostración conecta datos → modelo → ranking → revisión humana. | Flujo demostrable de extremo a extremo | T08 a T10 |
| KAN-22 | T12 · Pruebas y consolidación | E6 Pruebas, documentación y cierre | Día 5–6 | 21/nov/26 | Equipo completo | Ejecutar desde cero, corregir errores, ordenar archivos, preparar README, guion y diapositivas. | Notebook final, instrucciones, evidencias y presentación | Otra persona puede seguir la demo con las instrucciones disponibles. | Demo ensayada y artefactos consolidados | T05 a T11 |
| KAN-23 | T13 · Presentación final | E6 Pruebas, documentación y cierre | Día 6 | 21/nov/26 | Equipo completo | Exponer problema, datos, método, demo, métricas, riesgos, límites y hoja de ruta. | Presentación y demostración del prototipo | Se distingue lo implementado, lo simulado y lo pendiente para un piloto real. | Prototipo presentado y cierre del sprint | T12 |
| KAN-24 | T14 · Documentación continua (README y diapositivas) | E7 Transversal (documentación y riesgos) | Día 3–5 | 20/nov/26 | — | Ir armando README, guion y diapositivas en paralelo para no concentrarlo en el día 6. | — | Al iniciar el día 6 existe un borrador completo de README y diapositivas. | — | — |
| KAN-25 | T15 · Registro de riesgos e impedimentos (revisión diaria) | E7 Transversal (documentación y riesgos) | Día 1–6 | 21/nov/26 | — | Revisión de 10 minutos al cierre de cada día para actualizar avance, bloqueos y cambios de alcance. | — | Tablero actualizado y registro de impedimentos al día. | — | — |