# Informe General de Desarrollo y Gestión del Proyecto
## Sistema Predictivo de Inasistencias a Citas Médicas · Hospital Aurora

**Proyecto:** Prototipo Predictivo de Inasistencias a Citas Médicas  
**Código del Proyecto (Jira):** `KAN` (Prediccion inasistencias)  
**Integrantes:** 
- **Alfredo Javier Álvarez Reyes** — Analista de Datos / Data Engineer
- **Diego Alonso Bustamante Belmar** — Analista de Machine Learning / Desarrollador de Prototipo / Coordinador  
**Periodo del Sprint:** 16 al 21 de noviembre de 2026 (Consolidación y Cierre: 23 de noviembre de 2026)  
**Dataset Base:** *Medical Appointment No Shows* (Kaggle V5, 110.527 registros brutos)  
**Artefactos Finales:** `models/mejor_modelo_lgbm.joblib` y `index.html` (Prototipo Web Autónomo)  

---

## 1. Resumen Ejecutivo

### Actualización posterior del prototipo

El `index.html` vigente incorpora la versión EV3 de la agenda y mensajería simulada, recibida después del corte documentado a continuación. Incluye plantillas, comprobantes, estados de entrega y reintentos de SMS ficticios. No envía mensajes reales ni establece una conexión con Twilio. La definición de entregabilidad y el recorrido se detallan en [README-de-demo.md](README-de-demo.md). Los informes PDF, la presentación, las capturas y las huellas del corte anterior conservan su carácter histórico y pueden mostrar la interfaz previa. El objeto del modelo integrado conserva los mismos 1.050 árboles y 38 variables.

### Estado consolidado tras actualización de Jira

Al cierre documental del 23 de noviembre de 2026, Jira registra **14 de 15 tareas finalizadas**. Solo **T13 (presentación final)** permanece en curso porque el equipo confirmó que los materiales están terminados y la exposición aún no se realiza. Las épicas E1–E5 y E7 están finalizadas y E6 continúa por T13. Las quince tareas tienen enlaces al paquete de entregables y fechas de ejecución declaradas por el equipo.

El equipo confirmó ejecución durante el cronograma, con la desviación de T06/T07 el 19 de noviembre y recuperación técnica el 20 de noviembre. El registro administrativo posterior no se interpreta como fecha del trabajo técnico. T15 contiene seis entradas retrospectivas por jornada, el incidente y las acciones correctivas. Los vencimientos y las fechas automáticas del sistema se conservan.

El entorno local fue reparado para usar Python 3.12.4. Se ejecutó el entrenamiento completo en una carpeta aislada: las métricas coinciden con las vigentes con tolerancia absoluta de 1e-9. Las huellas del modelo, metricas.json e index.html se conservaron. La evidencia está en `docs/evidencias/reproduccion_entrenamiento_2026-11-23.json`.

Los valores históricos 0.8170 y 0.5123 procedían del entrenamiento. Las métricas OOF son la referencia vigente y participan en la selección, por lo que un piloto exige evaluación prospectiva independiente. El [informe principal actualizado](../output/pdf/Informe_Principal_EP5_EP6_Actualizado.pdf) y el [registro actualizado de desviaciones](../output/pdf/Registro_Control_Desviaciones_Actualizado.pdf) reflejan el cierre actual.

El presente informe documenta de forma exhaustiva la génesis, planificación, desarrollo técnico, gestión ágil y despliegue del **Sistema Predictivo de Inasistencias a Citas Médicas (Hospital Aurora)**. El proyecto aborda la problemática crítica de las inasistencias no notificadas (*No-shows*) en la atención médica ambulatoria, fenómeno que en el conjunto de datos de referencia afecta al **20.2%** de las consultas, ocasionando pérdidas de capacidad de atención especializada, encarecimiento de costos operativos, postergación diagnóstica y sobrecarga en los servicios de urgencia.

A través de un ciclo de desarrollo disciplinado de seis días, el equipo diseñó, entrenó, optimizó y empaquetó un modelo de aprendizaje automático de alto desempeño (**LightGBM** con 1.050 árboles de decisión), sustentado en una estricta ingeniería de variables causales y validación cruzada por paciente (`GroupKFold`). El modelo demostró una capacidad de discriminación sobresaliente con un **ROC-AUC de 0.7535**, un **PR-AUC de 0.4103** y un **Lift@20 de 2.0140**, superando con holgura los criterios de éxito definidos por el negocio. 

El proyecto culminó exitosamente con la creación de una **demostración interactiva autónoma en navegador ([index.html](../index.html))**, que transpila e integra los árboles del modelo directamente en JavaScript del lado del cliente. La aplicación opera sin costos, sin servidores, sin dependencias externas y en modalidad 100% *offline*, incorporando explicabilidad local basada en valores SHAP, simulación de escenarios hipotéticos ("what-if") y un estricto protocolo ético de apoyo a la toma de decisiones clínicas con aprobación humana (*Human-in-the-Loop*).

---

## 2. Marco Metodológico Aplicado

La ejecución del proyecto articuló cuatro marcos metodológicos complementarios, garantizando rigor científico, trazabilidad técnica, agilidad operativa y gobernanza de datos:

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                             MARCO METODOLÓGICO INTEGRADO                                  │
├────────────────────────┬────────────────────────┬─────────────────────────────────────────┤
│       CRISP-DM         │     Agile / Kanban     │          PMBOK (PMI) & Gobernanza       │
│ (Ciclo de Vida de Datos)│  (Gestión del Flujo)   │      (Líneas Base y Puertas de Control)  │
├────────────────────────┼────────────────────────┼─────────────────────────────────────────┤
│ 1. Business Understand.│ • Tablero Jira (KAN)   │ • Gestión de Integración (Charter T01)  │
│ 2. Data Understanding  │ • 7 Épicas (E1 a E7)   │ • Gestión del Alcance (WBS / EDT)       │
│ 3. Data Preparation    │ • 15 Tareas (T01-T15)  │ • Gestión de Riesgos (Matriz R1-R8)     │
│ 4. Modeling            │ • Standups diarios     │ • Gobernanza Stage-Gate (Puertas A, B, C)│
│ 5. Evaluation          │ • Control cruzado      │ • Gobernanza de Datos (DAMA-DMBOK)      │
│ 6. Deployment          │ • Definition of Done   │ • Ética IA en Salud (Human-in-the-Loop) │
└────────────────────────┴────────────────────────┴─────────────────────────────────────────┘
```

### 2.1. CRISP-DM (*Cross-Industry Standard Process for Data Mining*)
Estructuró las fases técnicas del ciclo de vida del dato:
1. **Comprensión del Negocio (*Business Understanding*):** Formalización de objetivos, límites causales, costos operacionales y matriz de KPIs en [docs/T02-problema_user_y_kpis.md](../docs/T02-problema_user_y_kpis.md).
2. **Comprensión de los Datos (*Data Understanding*):** Catalogación de la fuente Kaggle, diccionario de atributos en [docs/T03-diccionario-datos.md](../docs/T03-diccionario-datos.md), auditoría de calidad en [docs/T04-calidad-datos.md](../docs/T04-calidad-datos.md) y análisis exploratorio en `notebooks/03_EDA.ipynb`.
3. **Preparación de Datos (*Data Preparation*):** Depuración de registros anómalos, imputación lógica, cálculo de brechas temporales (`WaitingDays`) y construcción de historiales causales libres de fuga en `notebooks/02_Preprocesamiento.ipynb`.
4. **Modelado (*Modeling*):** Experimentación con modelos base (Dummy, Regresión Logística), árboles de decisión gradient boosting (LightGBM, CatBoost) y búsqueda bayesiana de hiperparámetros con Optuna en `scripts/optimizar_modelo.py`.
5. **Evaluación (*Evaluation*):** Validación fuera de pliegue (*Out-Of-Fold*, OOF) y validación temporal (*backtesting*), confrontando métricas frente a los KPIs de negocio.
6. **Despliegue (*Deployment*):** Construcción y validación del prototipo ejecutable standalone [index.html](../index.html).

### 2.2. Gestión Ágil: Kanban y Control Cruzado
Se implementó un flujo visual sobre Jira (proyecto clave `KAN`), estructurado en **7 épicas** (`KAN-4` a `KAN-10`) y **15 tareas** (`KAN-11` a `KAN-25`).
* **Límites de Trabajo en Curso (WIP):** Se restringió el avance paralelo a un máximo de dos tareas activas para evitar dispersión.
* **Control Cruzado de Pares (*Peer Review*):** Cada producto técnico desarrollado por un integrante requirió la revisión, ejecución independiente y aprobación del otro integrante antes de transicionar a estado *Finalizado*.
* **Definición de Terminado (*Definition of Done - DoD*):** Una tarjeta solo se dio por cerrada si contaba con su código ejecutable desde cero sin errores, artefacto enlazado, métricas trazables y documentación actualizada.
* **Revisiones Diarias de Impedimentos (T15):** Reuniones breves de sincronización diaria para registrar bloqueos y ajustar el plan.

### 2.3. DataOps y MLOps
* **Trazabilidad y Reproducibilidad:** Semillas aleatorias fijadas (`random_state=42`), entorno virtual controlado (`requirements.txt`) y ejecución estandarizada con `gestor-ipynb`.
* **Prevención de Contaminación Temporal (*Zero Leakage*):** Separación estricta entre pasado y futuro al calcular estadísticas de pacientes, empleando algoritmos de búsqueda binaria sobre series temporales ordenadas.
* **Gobernanza del Modelo:** Empaquetado de metadatos, listas de variables numéricas/categóricas y categorías fijadas dentro del payload serializado (`models/mejor_modelo_lgbm.joblib`) para asegurar inferencias consistentes.

### 2.4. Estándares PMBOK (PMI) y Enfoque de Gestión Híbrida
El proyecto incorporó las mejores prácticas del estándar **PMBOK (Project Management Body of Knowledge)** del PMI, adoptando una **gestión híbrida** que combina la disciplina estructurada de las líneas base predictivas con la adaptabilidad ágil del ciclo de desarrollo:

1. **Gestión de la Integración:**
   - **Acta de Constitución (*Project Charter*):** Formalizada en [docs/T01-documento.md](../docs/T01-documento.md), definiendo el propósito del proyecto, asignación de roles entre Diego y Alfredo, supuestos, restricciones y autoridad técnica.
   - **Control Integrado de Cambios:** Cualquier modificación sobre el alcance (por ejemplo, el rediseño del historial causal del Día 4) se evaluó en función de su impacto sobre el cronograma y los entregables comprometidos.
2. **Gestión del Alcance:**
   - **Línea Base del Alcance y WBS/EDT:** Estructuración jerárquica del trabajo en 6 paquetes y 15 actividades clave (A01 a A15), explicitando con precisión los límites del proyecto: uso exclusivo de datos públicos de Kaggle, simulación del ranking top 20% y exclusión total de envíos reales de SMS o interacción con sistemas clínicos en producción.
   - **Verificación del Alcance:** Cada entregable debió satisfacer criterios verificables independientes antes de considerarse aceptado formalmente.
3. **Gestión del Cronograma:**
   - **Secuenciamiento y Dependencias:** Definición explícita de la red de precedencias de tareas (T01 $\rightarrow$ T02 $\rightarrow$ T03 $\rightarrow$ T04 $\rightarrow$ T05 $\rightarrow$ T06/T07 $\rightarrow$ T08 $\rightarrow$ T09 $\rightarrow$ T10 $\rightarrow$ T11 $\rightarrow$ T12 $\rightarrow$ T13).
   - **Gestión de Variaciones y Holguras:** Ante el incidente del Día 4, el equipo reporta reorganización de tareas, programación en parejas y automatización del ajuste. No se dispone de una medición de ruta crítica ni de recursos adicionales que permita acreditar formalmente fast-tracking o crashing.
4. **Gestión de Costos y Valor de Negocio:**
   - Estimación y modelado financiero en [docs/T02-problema_user_y_kpis.md](../docs/T02-problema_user_y_kpis.md): escenario simulado de 10.000 citas elegibles, 2.000 contactos priorizados, presupuesto referencial de \$11.004.330 CLP y umbral de costo por cita recuperada atribuible $\le \$60.000$ CLP (KPI I3 y G2).
5. **Gestión de la Calidad:**
   - Planificación y aseguramiento de calidad con métricas cuantificables: completitud de atributos $\ge 98\%$ (KPI D1), consistencia de etiquetas $\ge 99\%$ (KPI D2), y validación de reproducibilidad técnica en máquinas limpias.
6. **Gestión de Riesgos:**
   - Matriz integral de riesgos cualitativos (R1 a R8) categorizados por probabilidad e impacto: desde fuga de información (R3) y sesgos algorítmicos (R2) hasta transferencia contextual de datos Brasil-Chile (R7).
   - Plan de respuesta a riesgos y seguimiento diario en el registro de impedimentos (T15).
7. **Gestión de Interesados (*Stakeholders*):**
   - Mapeo y análisis de necesidades de los usuarios del sistema: equipos de coordinación médica, directores hospitalarios, administrativos de agendamiento y pacientes citados.

### 2.5. Metodologías y Marcos de Gobernanza

La gobernanza del proyecto garantizó que el desarrollo no fuera un mero ejercicio algorítmico aislado, sino una solución sujeta a controles normativos, organizacionales y éticos:

#### A. Gobernanza de Proyectos y Puertas de Decisión (*Stage-Gate / Phase-Gate*)
El proyecto estructuró su evolución a través de **Puertas de Decisión (*Stage Gates*)** que actúan como puntos de control *Go/No-Go*:
* **Puerta de Decisión A (Fin Paquete 2 - Datos y Privacidad):** Validación de la licitud de los datos, anonimización y controles de seguridad. *Criterio de aprobación:* Verificación de que el dataset Kaggle estuviera libre de identificadores personales sensibles y con diccionario completo antes de iniciar cualquier entrenamiento.
* **Puerta de Decisión B (Fin Paquete 3 - Modelo Predictivo):** Verificación técnica de desempeño y equidad. *Criterio de aprobación:* Superar los umbrales mínimos de Recall $\ge 0.65$ y Lift@20 $\ge 1.50$, demostrando ausencia total de fuga temporal (*zero leakage*) mediante validación cruzada por paciente (`GroupKFold`).
* **Puerta de Decisión C (Fin Paquete 4 - Integración y Pruebas):** Control de calidad de la interfaz y seguridad operativa (*Go/No-Go* de demostración). *Criterio de aprobación:* Prueba de extremo a extremo que confirme que la interfaz [index.html](../index.html) no realiza conexiones no autorizadas y presenta advertencias visibles de simulación académica.

#### B. Gobernanza de Datos (Alineada con DAMA-DMBOK)
* **Gestión de la Calidad del Dato:** Auditoría bajo dimensiones formales: completitud (100% en el dataset limpio), validez temporal (corrección de citas previas a agendamientos), consistencia de tipos y unicidad de identificadores (`AppointmentID`).
* **Metadatos y Trazabilidad:** Diccionario estructurado ([docs/T03-diccionario-datos.md](../docs/T03-diccionario-datos.md)) y linaje claro de transformaciones desde el CSV original hasta las variables analíticas finales.
* **Seguridad y Privacidad:** Tratamiento de identificadores mediante anonimización de `PatientId` y preservación de confidencialidad (KPI G4: 0 incidentes de seguridad).

#### C. Gobernanza Ética de Inteligencia Artificial en Salud (OMS / IEEE / Principios de IA Confiable)
* **Principio de Supervisión Humana (*Human-in-the-Loop*):** El modelo está concebido estrictamente como un Sistema de Soporte a la Decisión Clínica (*Clinical Decision Support System - CDSS*). El score predictivo **nunca** cancela, rechaza ni reasigna citas automáticamente (KPI G3: 0 casos de intervención destructiva).
* **Principio de Transparencia y Explicabilidad (XAI):** La interfaz [index.html](../index.html) descompone el puntaje en valores SHAP en escala log-odds, permitiendo al personal de salud comprender qué factores explican la estimación de riesgo de cada paciente.
* **Principio de Equidad y No Discriminación (*Fairness*):** Monitoreo estricto de la brecha máxima de recall entre grupos demográficos y socioeconómicos (KPI M3: diferencia $\le 10$ puntos porcentuales).
* **Trazabilidad y Auditoría Asistencial:** La aplicación web registra una bitácora local persistente, modificable o eliminable desde el navegador de cada contacto simulado, SMS enviado o gestión administrativa, asegurando responsabilidad sobre las acciones realizadas.

---

## 3. Bitácora del Equipo por Jornada y Contraste con el Cronograma

A continuación se presenta el relato cronológico del equipo. Esta bitácora describe la ejecución reportada, mientras que el estado y las fechas de cierre administrativo se contrastan con Jira. No equivale a una medición de avance aceptado en cada jornada.

```
    Lunes 16/Nov       Martes 17/Nov      Miércoles 18/Nov    Jueves 19/Nov       Viernes 20/Nov      Sábado 21/Nov
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │    DÍA 1     │   │    DÍA 2     │   │    DÍA 3     │   │    DÍA 4     │   │    DÍA 5     │   │    DÍA 6     │
   ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤   ├──────────────┤
   │ • T01 Inicio │   │ • T03 Fuente │   │ • T04 Calidad│   │ • INCIDENTE  │   │ • T08 Modelo │   │ • T12 Pruebas│
   │ • T02 Negocio│   │   y Diccion. │   │   (Cierre)   │   │   Técnico y  │   │   LightGBM   │   │   Consolidac.│
   │ • Setup Jira │   │ • T04 Calidad│   │ • T05 Preproc│   │   Bloqueo    │   │ • T09 Eval.  │   │ • T13 Cierre │
   │   (KAN-4)    │   │   (Inicio)   │   │   Dataset CSV│   │ • Pair Prog. │   │ • T10 Ranking│   │   y Ensayo   │
   │              │   │              │   │ • T14 Doc.   │   │ • Rediseño   │   │ • T11 Demo   │   │ • T15 Cierre │
   │              │   │              │   │              │   │   GroupKFold │   │   index.html │   │   Riesgos    │
   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

### Día 1 — Lunes 16 de Noviembre de 2026: Inicio, Gobierno y Comprensión del Negocio
* **Tareas Jira:** `KAN-11` (T01 · Inicio y alcance) y `KAN-12` (T02 · Comprensión del negocio).
* **Actividades Ejecutadas:**
  - Diego y Alfredo realizaron la sesión de arranque para formalizar los objetivos y límites del proyecto. Se redactó y aprobó el acta de inicio ([docs/T01-documento.md](../docs/T01-documento.md)).
  - Se configuró el tablero Jira con sus 7 épicas y 15 tareas, estableciendo la política de revisión cruzada: Alfredo como líder de datos y revisor de métricas/demo; Diego como líder de modelamiento/desarrollo y coordinador del tablero.
  - Se formuló la ficha de comprensión del negocio ([docs/T02-problema_user_y_kpis.md](../docs/T02-problema_user_y_kpis.md)), clarificando que la etiqueta `No-show = Yes` corresponde a la inasistencia (clase positiva minoritaria).
  - Se definieron los KPIs cuantitativos irrenunciables: **Recall $\ge 0.65$**, **Lift@20 $\ge 1.50$**, **Completitud $\ge 98\%$** y **0 cancelaciones automáticas** (Gobernanza G3).
* **Entregables Logrados:** Acta de inicio, ficha de KPIs y tablero Kanban operativo.
* **Control Diario (T15):** Sin impedimentos. Trabajo al día.

### Día 2 — Martes 17 de Noviembre de 2026: Ingesta de Datos y Diccionario
* **Tareas Jira:** `KAN-13` (T03 · Fuente y diccionario) y arranque de `KAN-14` (T04 · Perfilamiento de calidad).
* **Actividades Ejecutadas:**
  - Alfredo descargó y verificó la integridad del dataset oficial de Kaggle (`data/KaggleV2-May-2016.csv`), constatando 110.527 filas y 14 columnas brutas.
  - Se construyó el diccionario de datos ([docs/T03-diccionario-datos.md](../docs/T03-diccionario-datos.md)), detallando tipos de datos esperados, semántica y restricciones.
  - Diego efectuó la revisión cruzada, confirmando que las variables capturan el contexto de Vitória (Brasil) y validando la advertencia de límite de alcance sobre transferibilidad directa a centros locales.
* **Entregables Logrados:** Diccionario técnico y dataset versionado en el entorno local.
* **Control Diario (T15):** Se detectó que las fechas venían formateadas como texto plano (`object`), requiriendo tipificación estricta en el pipeline.

### Día 3 — Miércoles 18 de Noviembre de 2026: Auditoría de Calidad y Preprocesamiento
* **Tareas Jira:** Cierre de `KAN-14` (T04 · Perfilamiento de calidad) y `KAN-15` (T05 · Preparación de datos).
* **Actividades Ejecutadas:**
  - Alfredo completó el perfilamiento de datos ([docs/T04-calidad-datos.md](../docs/T04-calidad-datos.md)) en `notebooks/01_Perfilamiento.ipynb`. Se confirmó 0% de valores nulos y 0 duplicados exactos, pero se descubrieron anomalías críticas: **6 casos de edades inconsistentes** (-1 y 115 años) y **5 casos de fechas incoherentes** donde la cita ocurría antes de la fecha de agendamiento.
  - Se implementó el pipeline reproducible en `notebooks/02_Preprocesamiento.ipynb`. Se eliminaron las 6 observaciones corruptas, se normalizaron los timestamps y se calculó la brecha de anticipación `WaitingDays = AppointmentDay - ScheduledDay`.
  - Se generó el dataset limpio de referencia: `data/citas_medicas_preprocesadas.csv` (**110.521 filas, 16 columnas y 62.298 pacientes únicos**).
* **Entregables Logrados:** Reporte de calidad, notebook de preparación reproducible y dataset limpio exportado.
* **Control Diario (T15):** Avance según cronograma. Se inició la documentación continua (T14).

---

### Día 4 — Jueves 19 de Noviembre de 2026: El Incidente Técnico y Contingencia Operativa

Durante la jornada del Día 4, planificada para ejecutar el Análisis Exploratorio (`KAN-16`, T06) y establecer la Partición y Baselines (`KAN-17`, T07), el proyecto enfrentó una **contingencia operativa y técnica crítica** que paralizó temporalmente el avance:

#### 1. Origen del Problema
* **Falla de Infraestructura y Conectividad:** Durante el Día 4, el sector de residencia de Alfredo sufrió un corte eléctrico imprevisto acompañado de una caída en la red de internet, imposibilitando la entrega y consolidación de los gráficos de distribución y asociaciones del EDA.
* **Revisión de la evaluación inicial:** El informe técnico `docs/INFORME_MODELO_ML.md` aclara que los valores **ROC-AUC de 0.8170** y **PR-AUC de 0.5123** procedían del conjunto de entrenamiento. Se descartaron como evidencia de generalización. Separadamente, el equipo reforzó los controles de fuga por paciente y disponibilidad temporal de las variables. 
* La evaluación vigente agrupa por paciente y construye historial solo con citas anteriores. El valor histórico de Lift@20 = 1.18 citado en el relato inicial no aparece en `models/metricas.json` y no se utiliza como línea base verificable. La referencia reproducible del notebook 05 es **Lift@20 = 1.9311**.
* **Situación al Cierre de la Jornada:** El relato del equipo registra un retraso de T06 y un bloqueo técnico de T07. La desviación se clasifica cualitativamente como relevante por su efecto sobre la evaluación y la integración posteriores.

#### 2. Protocolo de Contingencia y Recuperación (Sprint Recovery)
Tras recuperar la conectividad, el equipo reporta la activación de un protocolo de contingencia. La descripción se consolida como evidencia retrospectiva en [Registro de control de desviaciones](../output/pdf/Registro_Control_Desviaciones.pdf). `KAN-25` (T15) quedó consolidada con seis comentarios retrospectivos por jornada y el registro del incidente. Las entradas describen la ejecución reportada y conservan su fecha real de creación:
1. **Sesión Intensiva de Programación en Parejas (*Pair Programming*):** Ambos integrantes trabajaron de forma colaborativa mediante llamada y pantalla compartida. Alfredo estructuró la lógica matemática de un **historial causal retrospectivo** para prevenir la fuga temporal: cada cita solo podría consultar citas anteriores de atención mediante búsqueda binaria (`np.searchsorted` con `side='left'`), aplicando suavizado bayesiano ($\alpha=20.0$) para evitar distorsiones en pacientes nuevos.
2. **Reingeniería de la Validación:** Diego reconfiguró la partición a un esquema estricto de **`GroupKFold(5)` agrupado por `PatientId`**, garantizando que ningún paciente del conjunto de prueba hubiera sido visto durante el entrenamiento. Asimismo, se tomó la decisión técnica de **eliminar `is_unbalance=True`**, comprobando que el desbalanceo se debía corregir mediante la calibración posterior del umbral de decisión y no alterando los pesos de los árboles.
3. **Automatización del ajuste:** Diego dejó ejecutándose en segundo plano el script `scripts/optimizar_modelo.py` con **Optuna**, explorando 50 ensayos sobre el espacio de hiperparámetros (profundidad, número de hojas, tasa de aprendizaje y submuestreo) optimizando directamente sobre PR-AUC.

#### 3. Resultado de la Recuperación
Según la bitácora del equipo, durante el Día 5 concluyó el ajuste y se retomó el trabajo de evaluación y prototipo. Los scripts y métricas disponibles respaldan la corrección técnica. La fecha exacta de recuperación del cronograma no se acredita con el tablero: varios cierres administrativos se registran entre el 21 y el 23 de noviembre.

---

### Día 5 — Viernes 20 de Noviembre de 2026: Modelamiento Avanzado, Evaluación y Prototipado
* **Tareas Jira:** `KAN-18` (T08 · Modelamiento), `KAN-19` (T09 · Evaluación), `KAN-20` (T10 · Priorización top 20%) y `KAN-21` (T11 · Prototipo visual).
* **Actividades Ejecutadas:**
  - Se formalizó el modelo definitivo en `scripts/entrenar_modelo.py` utilizando los mejores hiperparámetros de Optuna: **LightGBM con 1.050 árboles, `learning_rate=0.0105`, `num_leaves=118` y `max_depth=12`**.
  - Se calcularon las métricas fuera de pliegue (OOF) sobre las 110.521 observaciones: **ROC-AUC 0.7535**, **PR-AUC 0.4103**, **F1 0.4628** y **Lift@20 de 2.0140**, validando además la robustez temporal en backtesting (*corte 25 de mayo* con ROC-AUC 0.7303 y Lift@20 1.8876).
  - Se definieron los dos umbrales operativos: **0.2219 (F1 Óptimo)** y **0.2508 (KPI M1 Recall $\ge 65\%$ con precisión 35.36%)**.
  - Se exportó el artefacto final a `models/mejor_modelo_lgbm.joblib` y las métricas a `models/metricas.json`.
  - **Desarrollo del Prototipo Visual:** Diego implementó la interfaz gráfica autónoma en `index.html`. Transpiló los 1.050 árboles de decisión del modelo a un motor de ejecución en JavaScript cliente, incorporó la agenda simulada de 12 pacientes ("Hospital Aurora"), el módulo de explicabilidad SHAP en log-odds, la simulación "what-if" de tiempos de espera y el selector dinámico de umbrales.
  - Alfredo realizó la validación funcional de extremo a extremo, corroborando que las probabilidades calculadas en el navegador fueran idénticas a las emitidas por LightGBM en Python.
* **Entregables Logrados:** Artefacto serializado `mejor_modelo_lgbm.joblib`, informe técnico [docs/INFORME_MODELO_ML.md](../docs/INFORME_MODELO_ML.md), ranking simulado y prototipo [index.html](../index.html).
* **Control Diario (T15):** La bitácora reporta recuperación técnica. El estado Finalizado de T08 a T11 se observa en Jira al corte del 23 de noviembre, no como una prueba de cierre administrativo en el Día 5.

### Día 6 — Sábado 21 de Noviembre de 2026: Pruebas, Consolidación y Ensayo Final
* **Tareas Jira:** `KAN-22` (T12 · Pruebas y consolidación), cierre de `KAN-24` (T14 · Documentación) y preparación de `KAN-23` (T13 · Presentación final).
* **Actividades Ejecutadas:**
  - La bitácora del equipo declara una prueba integral en una máquina limpia. Como verificación adicional, el 23/nov se reparó la ruta del entorno de esta copia y se ejecutó el entrenamiento completo en una salida aislada. Las métricas coincidieron con las vigentes; evidencia: [reproducción del entrenamiento](evidencias/reproduccion_entrenamiento_2026-11-23.json).
  - Se estructuró y redactó el [README.md](../README.md) principal en la raíz, integrando la guía operativa, arquitectura técnica y el guion de demostración de `README-de-demo.md`.
  - Se organizaron todos los documentos conceptuales y reportes en la carpeta consolidada `docs/`.
  - Se ensayó la presentación final y la navegación del prototipo [index.html](../index.html), revisando el flujo de demostración y validando el mensaje ético central: apoyo a la decisión humana sin descarte automático de cupos médicos.
* **Entregables Logrados:** Repositorio limpio, documentación consolidada y demostración lista para presentación.
* **Control Diario (T15):** El relato del equipo sitúa aquí la consolidación técnica. En la consolidación actual, T14 y T15 quedan cerradas. La exposición de T13 permanece pendiente según confirmación del equipo.

---

## 4. Desarrollo Técnico Fase por Fase y Metodologías Reflejadas

Esta sección desglosa las soluciones de ingeniería adoptadas y cómo reflejaron la aplicación práctica de los marcos de trabajo:

### 4.1. Fase de Comprensión del Negocio (CRISP-DM + Agile)
* **Reflejo Metodológico:** Se evitó el sesgo común de optimizar ciegamente la exactitud (*accuracy*), dado que un clasificador ingenuo alcanza 79.8% prediciendo siempre asistencia pero es inútil para la clínica.
* **Decisión de Diseño:** En [docs/T02-problema_user_y_kpis.md](../docs/T02-problema_user_y_kpis.md) se definieron como métricas reinas el **Recall** (capturar la mayor cantidad de pacientes que faltarán) y el **Lift@20** (eficiencia en priorizar recursos en el quintil más riesgoso).
* **Principio de Gobernanza:** Se estipuló el KPI **G3 (0 cancelaciones automáticas)**, garantizando que el modelo sea un instrumento de contacto preventivo y no de exclusión asistencial.

### 4.2. Fase de Calidad y Preprocesamiento (CRISP-DM + DataOps)
* **Reflejo Metodológico:** Aplicación de controles de calidad auditables con criterios de aceptación documentados.
* **Tratamiento de Anomalías:**
  - En `notebooks/01_Perfilamiento.ipynb` se registraron 6 registros con edades fuera de norma (-1 y 115) y 5 registros con inconsistencia temporal. Fueron excluidos de forma justificada en `notebooks/02_Preprocesamiento.ipynb`.
  - Generación de `WaitingDays`: variable clave que mide el tiempo de anticipación entre el agendamiento y la consulta.
* **Resultado:** Reducción controlada de 110.527 a **110.521 filas**, preservando el 99.99% de los datos con 100% de integridad en atributos críticos (KPI D1 cumplido).

### 4.3. Fase de Ingeniería de Características (CRISP-DM + MLOps)
* **Reflejo Metodológico:** Enfoque *Data-Centric AI* con prevención matemática de fuga de información (*leakage*).
* **Historial Causal Retrospectivo (`agregar_historial_causal`):**
  A diferencia de abordajes estándar que calculan promedios globales por paciente sobre todo el dataset, se diseñó una función causal estricta:
  ```python
  # Búsqueda binaria para observar únicamente citas pasadas
  idx = np.searchsorted(fechas_citas_paciente, fecha_cita_actual, side='left')
  ```
* **Variables Derivadas Causalmente:**
  - `PriorVisits`: Citas previas del paciente.
  - `PriorNoShows`: Ausencias previas acumuladas.
  - `PriorNoShowRate`: Suavizado bayesiano empírico con constante $\alpha = 20.0$:
    $$\text{PriorNoShowRate} = \frac{\text{PriorNoShows} + 20.0 \times 0.2019}{\text{PriorVisits} + 20.0}$$
    Esto evita que un paciente con una sola cita previa y una falta quede asignado espuriamente a 100% de riesgo.
  - `DaysSinceLastVisit`, `LastVisitNoShow`, `HasHistory`.
* **Variables de Calendario y Clínicas:** `WaitingDaysLog` ($\ln(1 + \text{WaitingDays})$), `IsSameDay` (cita el mismo día), `ScheduledHour`, `ScheduledDayOfWeek`, `AppointmentWeek`, `IsSenior`, `IsChild`, `ChronicAndSenior` y `HasHandcap`. Totalizando **38 variables predictivas**.

### 4.4. Fase de Modelamiento y Optimización (CRISP-DM + MLOps)
* **Reflejo Metodológico:** Comparación sistemática de hipótesis bajo idénticas condiciones experimentales.
* **Experimentación:**
  - **Baseline:** Dummy (Accuracy 79.8%, Recall 0.0%) y Regresión Logística (Recall 55.9%, Precisión 31.6%).
  - **LightGBM:** Superioridad clara sobre CatBoost en velocidad de convergencia y capacidad de ordenamiento.
  - **Optimización con Optuna:** 50 iteraciones para maximizar PR-AUC sobre validación cruzada. Se descartó `is_unbalance=True`, ya que deformaba la calibración de probabilidades relativas.
  - **Parámetros Finales:** 1.050 estimadores, tasa de aprendizaje de 0.01053, profundidad máxima 12, 118 hojas, submuestreo de 0.7238 y muestreo de columnas de 0.7412.

### 4.5. Fase de Evaluación Rigurosa (CRISP-DM)
* **Reflejo Metodológico:** Validación no contaminada por paciente (`GroupKFold(5)` por `PatientId`) y validación temporal prospectiva.
* **Resultados Consolidados:**

| Métrica Evaluada | Baseline (Notebook 05) | Modelo Mejorado (OOF) | Meta KPI | Estado KPI |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.7364 | **0.7535** | — | **Excelente** |
| **PR-AUC** | 0.3797 | **0.4103** | — | **Excelente (+3.06 pp)** |
| **F1-Score** | 0.4490 | **0.4628** | — | **Balance Óptimo** |
| **Recall (Inasistencias)** | 0.7565 | **0.7401** / **0.6500** | $\ge 0.65$ | **CUMPLE (KPI M1)** |
| **Precisión** | 0.3193 | **0.3366** / **0.3536** | — | **Mayor eficiencia en llamadas** |
| **Lift@20 (Top 20%)** | 1.9311 | **2.0140** | $\ge 1.50$ | **CUMPLE (KPI M2, +34% sobre meta)** |
| **Recall@20** | 0.3862 | **0.4028** | — | **40.28% capturado en el 20%** |

* **Calibración de Umbrales:**
  1. *Umbral F1 Óptimo (`0.2219`):* Balance balanceado para máxima detección general ($F_1 = 0.4628$).
  2. *Umbral Operativo / Negocio (`0.2508`):* Fija el Recall en 65.00% y eleva la precisión al 35.36%, reduciendo drásticamente los falsos positivos durante campañas telefónicas.

### 4.6. Fase de Despliegue y Prototipado Interactivo (CRISP-DM + Software Engineering)
* **Reflejo Metodológico:** Prototipado rápido de valor tangible para el usuario final (*Product-Led Machine Learning*).
* **Solución Técnica en [index.html](../index.html):**
  - Se embebió la estructura JSON de los 1.050 árboles de LightGBM directamente en el script cliente.
  - Implementación de un motor de inferencia en JavaScript puro que recorre las condiciones nodales (umbrales numéricos y pertenencia a subconjuntos de categorías de barrios) en tiempo real (< 10 ms por predicción).
  - Integración de explicabilidad local basada en SHAP, calculando las contribuciones aditivas en escala log-odds y agrupándolas por familias (anticipación, historial previo, factores clínicos y demográficos).
  - Simulador de escenarios ("what-if") que permite a la coordinación hospitalaria observar el impacto de acortar la espera de un paciente a 0 días.
  - Registro de auditoría y gestión de contactos persistente mediante `localStorage`.

---

## 5. Control de Gestión, Gobernanza y Estado del Tablero Jira

Estado contrastado con Jira al 23 de noviembre de 2026. Los criterios de la tabla resumen los entregables reportados y encontrados, no constituyen actas firmadas de aceptación. T14 y T15 se consolidaron y cerraron en esta actualización. T13 requiere realizar la exposición y registrar su cierre.

### 5.1. Cuadro de Tareas y Cumplimiento del Sprint (`KAN`)

| Tarea Jira | Épica Padre | Responsable | Estado | Criterio de Aceptación Verificado |
| :--- | :--- | :--- | :---: | :--- |
| **T01 · Inicio y alcance** (`KAN-11`) | E1 Inicio y negocio (`KAN-4`) | Alfredo (asig.) / Diego (coord.) | **Finalizado** | Acta de inicio guardada; tablero KAN configurado; límites éticos y supuestos acordados. |
| **T02 · Comprensión del negocio** (`KAN-12`) | E1 Inicio y negocio (`KAN-4`) | Diego (asig.) / Ambos | **Finalizado** | Ficha de problema formalizada; matriz de KPIs M1-M3, O1-O2, G1-G4 establecida. |
| **T03 · Fuente y diccionario** (`KAN-13`) | E2 Calidad y preparación (`KAN-5`) | Diego (asig.) / Alfredo | **Finalizado** | Inventario de 14 columnas Kaggle documentado; contexto geográfico Brasil explicitado. |
| **T04 · Perfilamiento de calidad** (`KAN-14`) | E2 Calidad y preparación (`KAN-5`) | Diego (asig.) / Alfredo | **Finalizado** | 6 anomalías de edad y 5 de fechas identificadas con conteos y justificación en notebook 01. |
| **T05 · Preparación de datos** (`KAN-15`) | E2 Calidad y preparación (`KAN-5`) | Diego (asig.) / Alfredo | **Finalizado** | Pipeline reproducible en notebook 02; dataset limpio derivado exportado (110.521 filas). |
| **T06 · EDA** (`KAN-16`) | E3 Análisis exploratorio (`KAN-6`) | Diego (asig.) / Alfredo | **Finalizado** | Gráficos y análisis de correlación en notebook 03; efecto same-day y reincidencia descritos. |
| **T07 · Baseline y partición** (`KAN-17`) | E4 Modelado y evaluación (`KAN-7`) | Alfredo (asig.) / Diego | **Finalizado** | Partición GroupKFold por paciente; detección y eliminación de fuga en notebook 04 y 05. |
| **T08 · Modelamiento** (`KAN-18`) | E4 Modelado y evaluación (`KAN-7`) | Alfredo (asig.) / Diego | **Finalizado** | LightGBM optimizado con Optuna (1.050 árboles); descarte justificado de is_unbalance. |
| **T09 · Evaluación** (`KAN-19`) | E4 Modelado y evaluación (`KAN-7`) | Alfredo (asig.) / Diego | **Finalizado** | Métricas OOF calculadas (ROC-AUC 0.7535, PR-AUC 0.4103); validación temporal backtesting. |
| **T10 · Priorización top 20%** (`KAN-20`) | E5 Ranking y prototipo (`KAN-8`) | Alfredo (asig.) / Diego | **Finalizado** | Lift@20 = 2.0140 calculado y verificado; regla de selección de quintil documentada. |
| **T11 · Prototipo visual** (`KAN-21`) | E5 Ranking y prototipo (`KAN-8`) | Alfredo (asig.) / Diego | **Finalizado** | Aplicación index.html con 1.050 árboles en JS, SHAP, what-if y flujo de aprobación humana. |
| **T12 · Pruebas y consolidación** (`KAN-22`) | E6 Pruebas y cierre (`KAN-9`) | Diego (asig.) / Ambos | **Finalizado** | Ejecución desde cero validada; repositorio estructurado y limpio; demo ensayada. |
| **T13 · Presentación final** (`KAN-23`) | E6 Pruebas y cierre (`KAN-9`) | Diego como coordinador; exposición de ambos | **En curso** | Materiales terminados según confirmación del equipo; exposición y evaluación del profesor pendientes. |
| **T14 · Documentación continua** (`KAN-24`) | E7 Transversal (`KAN-10`) | Diego / Alfredo | **Finalizado** | README.md integral en raíz y carpeta /docs completa con informes técnicos. |
| **T15 · Registro de riesgos** (`KAN-25`) | E7 Transversal (`KAN-10`) | Diego / Ambos | **Finalizado** | Seis registros retrospectivos por jornada, desviación y respuesta enlazadas; riesgos residuales identificados y registro consolidado. |

### 5.2. Gestión de Riesgos Mitigados durante el Proyecto

1. **Riesgo R3 (Fuga de Información Temporal y por Paciente):**  
   *Impacto Mitigado:* Falsas expectativas de desempeño causadas por partición aleatoria ingenua. Se neutralizó mediante la reingeniería a `GroupKFold(5)` por `PatientId` y el cálculo de historiales únicamente sobre citas pasadas.
2. **Riesgo R1 (Datos Anómalos):**  
   *Impacto Mitigado:* Distorsión de distribuciones por edades -1 y 115. Fueron aisladas y descartadas en la fase de perfilamiento auditado.
3. **Riesgo R7 (Transferencia Geográfica Brasil-Chile):**  
   *Impacto Mitigado:* Se mantuvo visible en todas las interfaces y reportes que los datos provienen de Vitória (Brasil), estableciendo que cualquier implantación local requerirá calibración previa.
4. **Riesgo R8 (Decisiones Injustas o Automatizadas):**  
   *Impacto Mitigado:* Implementación del principio *Human-in-the-Loop*. La aplicación [index.html](../index.html) no contiene funciones de cancelación automática; solo genera listas de trabajo para contacto asistencial.

---

## 6. Lecciones Aprendidas y Conclusiones

### 6.1. Lecciones de Gestión y Metodología
1. **La revisión cruzada salva proyectos:** Si Diego y Alfredo no hubieran contrastado las métricas del baseline durante el Día 4, el proyecto habría arrastrado un modelo sobreajustado artificialmente con `is_unbalance=True`, descubriendo la falla tarde en la presentación final.
2. **La agilidad requiere resiliencia operativa:** El bloqueo por corte de suministro eléctrico y la crisis del baseline demostraron el valor de los acuerdos de trabajo ágiles: comunicación inmediata, replanificación táctica y pair programming intensivo para absorber desvíos.
3. **Documentación viva y continua (T14):** Avanzar en la redacción de informes y README en paralelo con el código evitó cuellos de botella en el cierre, permitiendo consolidar la entrega técnica sin estrés de última hora.

### 6.2. Conclusiones Técnicas
1. **El valor del historial causal:** Incorporar la trayectoria de asistencia del paciente con suavizado bayesiano empírico fue el factor individual que mayor poder discriminante aportó al modelo (elevando el PR-AUC de 0.3797 a 0.4103).
2. **Desbalance resuelto en el umbral, no en el algoritmo:** Ponderar clases en LightGBM perjudicaba el ranking relativo. Mantener el entrenamiento con la distribución real y calibrar el umbral operativo a `0.2508` permitió satisfacer simultáneamente el Recall $\ge 65\%$ y elevar la precisión a 35.36%.
3. **Despliegue sin fricciones:** Demostrar que un modelo complejo de 1.050 árboles puede transmutarse a un archivo HTML/JS autónomo de cero dependencias ([index.html](../index.html)) elimina todas las barreras operacionales y económicas de adopción para los equipos clínicos.

---

*Informe consolidado en Santiago de Chile, a 23 de noviembre de 2026. La documentación y los riesgos quedaron consolidados. La presentación ante el profesor y su evaluación permanecen pendientes.*  
**Alfredo Javier Álvarez Reyes** (Data Engineer) & **Diego Alonso Bustamante Belmar** (ML Engineer & Coordinador).
