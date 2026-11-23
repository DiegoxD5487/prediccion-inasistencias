# Predicción de Inasistencias a Citas Médicas · Hospital Aurora

Prototipo integral de ciencia de datos y aprendizaje automático (*Machine Learning*) orientado a la gestión hospitalaria. El proyecto aborda la problemática de la inasistencia de pacientes a consultas programadas (*No-show*), abarcando desde la auditoría y depuración de datos crudos hasta el entrenamiento de un modelo **LightGBM** de alto rendimiento y su culminación en una **demostración interactiva autónoma en navegador ([index.html](index.html))**.

---

## 📌 Tabla de Contenidos

1. [Visión General del Proyecto](#-visión-general-del-proyecto)
2. [Problema de Negocio y Objetivos](#-problema-de-negocio-y-objetivos)
3. [Demostración Interactiva (`index.html`)](#-demostración-interactiva-indexhtml)
   - [Características del Prototipo](#características-del-prototipo)
   - [Guion de Demostración (3 Minutos)](#guion-de-demostración-3-minutos)
4. [Flujo de Trabajo y Pipeline de Datos](#-flujo-de-trabajo-y-pipeline-de-datos)
5. [Modelo Predictivo y Resultados](#-modelo-predictivo-y-resultados)
   - [Ingeniería de Características y Prevención de Fuga](#ingeniería-de-características-y-prevención-de-fuga)
   - [Validación Cruzada y Comparativa contra Baseline](#validación-cruzada-y-comparativa-contra-baseline)
   - [Umbrales de Decisión Operativos](#umbrales-de-decisión-operativos)
6. [Estructura del Repositorio](#-estructura-del-repositorio)
7. [Guía de Instalación y Reproducción](#-guía-de-instalación-y-reproducción)
8. [Documentación del Proyecto (`/docs`)](#-documentación-del-proyecto-docs)
9. [Supuestos, Ética y Limitaciones](#-supuestos-ética-y-limitaciones)
10. [Equipo](#-equipo)

---

## 🏥 Visión General del Proyecto

En los sistemas de salud públicos y ambulatorios, una proporción considerable de citas médicas agendadas se pierde debido a inasistencias no notificadas. Esto provoca subutilización de especialistas y pabellones, costos operacionales innecesarios, retraso de diagnósticos críticos y saturación de los servicios de urgencia.

Este proyecto desarrolla una solución analítica de extremo a extremo que:
1. **Audita y transforma** más de 110.000 citas médicas del dataset *Medical Appointment No Shows* (Kaggle).
2. **Entrena y optimiza** un modelo predictivo basado en Gradient Boosting (**LightGBM**, 1.050 árboles) calibrado para clasificar el riesgo de inasistencia mediante validación estricta por paciente (`GroupKFold`).
3. **Despliega un simulacro interactivo de gestión hospitalaria en un archivo único ([index.html](index.html))**, capaz de ejecutar el modelo completo en JavaScript del lado del cliente, calcular explicaciones locales tipo SHAP y simular escenarios "what-if" sin requerir servidores, backend ni conexión a internet.

---

## 🎯 Problema de Negocio y Objetivos

### El Reto Operativo
* **Etiqueta Objetivo:** `No-show` (`True` / `1` = el paciente **no asistió** a su cita).
* **Desbalance de Clases:** ~79.8% asistencias vs. **20.2% inasistencias**.
* **Impacto:** Pérdida de horas profesionales, medicamentos e insumos perecibles, y dilatación de listas de espera.

### Criterios de Éxito y KPIs (según [`docs/T02-problema_user_y_kpis.md`](docs/T02-problema_user_y_kpis.md))

| Categoría | KPI | Meta de Referencia | Estado del Modelo |
| :--- | :--- | :---: | :---: |
| **Desempeño** | **M1. Recall de inasistencias** | $\ge 0.65$ | **Cumple (0.6500 a 0.7401)** |
| **Priorización** | **M2. Lift@20 (Quintil superior)** | $\ge 1.50$ | **Cumple (2.0140)** |
| **Operación** | **O1. Cobertura de priorización** | $18\% - 22\%$ | **Cumple (~20%)** |
| **Gobernanza** | **G3. Intervención no destructiva** | **0 cancelaciones automáticas** | **Cumple (0)** |
| **Calidad** | **D1. Completitud de atributos** | $\ge 98\%$ | **Cumple (100%)** |

> [!IMPORTANT]
> **Human-in-the-Loop:** El modelo **nunca** cancela, rechaza ni reasigna citas automáticamente. Su única función es clasificar y ordenar las citas por riesgo para que el equipo clínico priorice llamadas de confirmación, recordatorios SMS personalizados o rescate de cupos.

---

## 💻 Demostración Interactiva (`index.html`)

El proyecto culmina en la interfaz interactiva autónoma contenida en **[index.html](index.html)**.

```
                  ┌──────────────────────────────────────────────┐
                  │          Hospital Aurora · Agenda            │
                  │   Turno Mañana · 20 de mayo de 2016          │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │  Scoring en Vivo JS    │                     │  Gestión y Contacto    │
     │  1.050 árboles LightGBM│                     │  Bitácora, SMS, Estado │
     └───────────┬────────────┘                     └───────────┬────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │  Explicabilidad (SHAP) │                     │  Simulación Hipotética │
     │  Aportes en log-odds   │                     │  Escenarios "What-If"  │
     └────────────────────────┘                     └────────────────────────┘
```

### Características del Prototipo
* **Zero-Cost & Zero-Dependencies:** No requiere instalar Node.js, Python, bases de datos ni servicios en la nube. Abre directamente con doble clic en navegadores modernos (Chrome, Edge, Firefox).
* **Inferencia Local en el Navegador:** Integra los **1.050 árboles de decisión** del modelo serializado (`mejor_modelo_lgbm.joblib`) directamente en código JavaScript optimizado, evaluando todas las ramificaciones y transformaciones numéricas/categóricas al instante.
* **Triage Clínico y Priorización:** Permite alternar entre orden cronológico y orden por puntaje de riesgo, filtrando citas que superan el umbral operativo.
* **Explicabilidad Transparente (XAI):** Muestra descomposiciones locales tipo SHAP para cada paciente en unidades log-odds, identificando qué factores aumentan o reducen la probabilidad predicha.
* **Simulación "What-If":** Permite al usuario manipular variables como días de anticipación de reserva para observar en tiempo real la variación en la probabilidad de asistencia.
* **Persistencia Local:** Las acciones registradas (llamadas, confirmaciones, notas de seguimiento) se guardan en `localStorage` del navegador.

### Guion de Demostración (3 Minutos)

1. **Presentación de la Agenda:** Abre [index.html](index.html) (presiona `F11` para pantalla completa). Se presenta la jornada del turno mañana (20 de mayo de 2016) con 12 pacientes simulados. Observa los tres indicadores superiores: total de citas, citas prioritarias sobre el umbral y pacientes contactados.
2. **Priorización y Gestión de Riesgo:**
   - Activa el filtro **Prioritarias** y ordena **Por hora**.
   - Selecciona a la paciente **Clara Azevedo** (estimación de riesgo inicial ~57.4%).
   - Presiona **Simular SMS** y revisa su bitácora de auditoría.
   - Haz clic en **Registrar contacto → Asistencia confirmada** y observa la actualización del indicador de contactos.
3. **Explicabilidad del Puntaje (SHAP):**
   - En la ficha de Clara, pulsa **Explicar este puntaje**.
   - Compara su probabilidad contra el umbral de corte.
   - En la sección *«¿Qué influyó más en su puntaje?»*, identifica las variables determinantes (tiempo de espera, inasistencias previas, etc.).
4. **Simulación de Escenarios ("What-If"):**
   - En la misma ventana de explicación, reduce la anticipación de la cita a 0 días (*same-day appointment*).
     - Observa el cambio en el puntaje predicho y la diferencia en puntos porcentuales. Es una comparación hipotética del modelo, no una medición del efecto causal de cambiar la espera.
5. **Selector de Criterio / Umbral:**
   - En la barra lateral o cabecera, conmuta el umbral de **Máximo F1 (0.2219)** a **Recall objetivo 65% (0.2508)**.
   - Observa cómo se recalculan las alertas de riesgo y las tarjetas de métricas históricas de validación cruzada.
6. **Reinicio del Simulacro:** Pulsa **Reiniciar simulacro** para vaciar la bitácora local y preparar otra presentación.

---

## 🔄 Flujo de Trabajo y Pipeline de Datos

El desarrollo siguió un ciclo riguroso y reproducible a través de los notebooks del repositorio:

```mermaid
flowchart TD
    A["Kaggle Raw<br>data/KaggleV2-May-2016.csv<br>(110.527 filas × 14 cols)"] --> B["01_Perfilamiento.ipynb<br>Auditoría de Calidad y Detección de Anomalías"]
    B --> C["02_Preprocesamiento.ipynb<br>Limpieza, WaitingDays, Anonimización"]
    C --> D["Dataset Limpio<br>data/citas_medicas_preprocesadas.csv<br>(110.521 filas × 16 cols)"]
    D --> E["03_EDA.ipynb<br>Análisis Exploratorio y Patrones de Inasistencia"]
    D --> F["04_Baselines.ipynb<br>Dummy Classifier y Regresión Logística"]
    D --> G["05_Modelamiento.ipynb / scripts/<br>Ingeniería Causal + LightGBM + Optuna"]
    G --> H["Artefacto Serializado<br>models/mejor_modelo_lgbm.joblib"]
    H --> I["Prototipo Interactivo en Navegador<br>index.html (Árboles JS + Interfaz)"]
```

### Resumen de Fases Analíticas

* **`notebooks/01_Perfilamiento.ipynb`:** Evaluación de completitud (0 valores nulos, 0 filas duplicadas). Detección de 6 registros con edades inconsistentes (edades negativas `-1` y valores extremos `115`) y 5 citas donde la fecha agendada era posterior a la fecha de atención.
* **`notebooks/02_Preprocesamiento.ipynb`:** Normalización temporal, cálculo de días de anticipación (`WaitingDays = AppointmentDay - ScheduledDay`), depuración de registros incoherentes y anonimización de identificadores. Dataset resultante: **110.521 filas y 62.299 pacientes**.
* **`notebooks/03_EDA.ipynb`:** Descubrimiento de correlaciones determinantes:
  - Las citas agendadas para el mismo día (*same-day*) tienen una inasistencia de apenas ~4.6%, mientras que citas con más de 15 días de anticipación superan el 32%.
  - Fuerte reincidencia: pacientes con ausencias pasadas duplican la tasa de inasistencia promedio.
* **`notebooks/04_Baselines.ipynb`:** Modelos de referencia iniciales. Un clasificador *Dummy* obtiene 79.8% de exactitud pero 0% de recall. La regresión logística balanceada alcanza recall de 55.9% con baja precisión (31.6%).
* **`notebooks/05_Modelamiento.ipynb`:** Construcción de modelos avanzados de árboles potenciados (LightGBM, CatBoost), búsqueda bayesiana de hiperparámetros con Optuna y empaquetado del artefacto productivo.

---

## 🧠 Modelo Predictivo y Resultados

El modelo final es un **`LGBMClassifier`** entrenado con 38 características estructuradas, serializado en [`models/mejor_modelo_lgbm.joblib`](models/mejor_modelo_lgbm.joblib).

### Ingeniería de Características y Prevención de Fuga

Para evitar la contaminación de información del futuro (*data leakage*) y simular con precisión la realidad hospitalaria, se implementó un **historial causal estricto** (`scripts/experimento_modelo.py`):

1. **Historial Causal por Paciente (`agregar_historial_causal`):**
   - Mediante búsqueda binaria (`np.searchsorted` con `side='left'`), cada cita solo consulta citas del mismo paciente con fecha de atención estrictamente anterior.
   - `PriorVisits`: Citas previas del paciente.
   - `PriorNoShows`: Inasistencias previas.
   - `PriorNoShowRate`: Tasa histórica con **suavizado bayesiano empírico** ($\alpha = 20.0$), evitando distorsiones en pacientes con pocas atenciones:
     $$\text{PriorNoShowRate} = \frac{\text{PriorNoShows} + \alpha \cdot \text{TasaGlobal}}{\text{PriorVisits} + \alpha}$$
   - `HasHistory`, `DaysSinceLastVisit`, `LastVisitNoShow`.
2. **Variables Temporales y de Espera:** `WaitingDaysLog` ($\ln(1 + \text{WaitingDays})$), `IsSameDay`, `ScheduledHour`, `ScheduledDayOfWeek`, `AppointmentWeek`.
3. **Variables Clínicas y Sociodemográficas:** `AgeGroup`, `IsChild`, `IsSenior`, `ChronicAndSenior` (interacción de edad con comorbilidades), `HasHandcap`.

### Validación Cruzada y Comparativa contra Baseline

La evaluación se realizó mediante **`GroupKFold(5)` agrupado por `PatientId`**, garantizando que los pacientes evaluados en validación nunca formaron parte del entrenamiento.

| Métrica de Evaluación | Baseline (Notebook 05) | Modelo Mejorado (OOF) | Mejora Absoluta | Interpretación de Impacto |
| :--- | :---: | :---: | :---: | :--- |
| **ROC-AUC** | 0.7364 | **0.7535** | **+0.0171** | Mejor discriminación global entre clases. |
| **PR-AUC (Avg Precision)** | 0.3797 | **0.4103** | **+0.0306** | Detección notablemente más precisa en la clase positiva. |
| **F1-Score** | 0.4490 | **0.4628** | **+0.0137** | Mejor equilibrio armónico entre precisión y recall. |
| **Recall (Inasistencias)** | 0.7565 | 0.7401 | -0.0164 | Cobertura amplia sobre pacientes que faltarán. |
| **Precisión** | 0.3193 | **0.3366** | **+0.0173** | Menor tasa de falsos positivos en llamadas preventivas. |
| **Lift@20** | 1.9311 | **2.0140** | **+0.0829** | **Supera ampliamente el KPI M2 ($\ge 1.50$)**. Duplica el riesgo base. |
| **Recall@20** | 0.3862 | **0.4028** | **+0.0166** | **El 40.28% de todas las inasistencias se captura contactando solo al 20% superior**. |

#### Validación Temporal Secundaria (Backtesting `2016-05-25`)
Al entrenar con citas hasta el 25 de mayo y evaluar a futuro:
* **ROC-AUC:** Sube de 0.7254 a **0.7303**.
* **Lift@20:** Sube de 1.8417 a **1.8876**.

### Umbrales de Decisión Operativos

El modelo ofrece dos puntos de corte calibrados con predicciones fuera de pliegue (OOF):

1. **Umbral F1 Óptimo (`0.2219`):** Maximiza el balance de clasificación ($F_1 = 0.4628$, Recall = 74.01%, Precisión = 33.66%).
2. **Umbral Negocio / KPI M1 (`0.2508`):** Satisface la exigencia de $\text{Recall} \ge 65\%$ (**65.00%**) incrementando la precisión a **35.36%**. Es el umbral recomendado para centros de salud que disponen de capacidad limitada de llamadas telefónicas.

---

## 📁 Estructura del Repositorio

```text
prediccion-inasistencias/
│
├── index.html                               # Demostración interactiva autónoma (1.050 árboles JS)
├── AGENTS.md                                # Instrucciones operativas para agentes y entorno
├── README.md                                # Documentación principal del proyecto
├── requirements.txt                         # Dependencias de Python fijadas
│
├── docs/                                    # Documentación técnica y de gestión del proyecto
│   ├── INFORME_GENERAL_PROYECTO.md          # Informe general de desarrollo, metodologías y cronograma
│   ├── T01-documento.md                     # Alcance, supuestos, riesgos, restricciones y Kanban
│   ├── T02-problema_user_y_kpis.md          # Definición de negocio, usuario y matriz de KPIs
│   ├── T03-diccionario-datos.md             # Diccionario y metadatos de variables crudas
│   ├── T04-calidad-datos.md                 # Informe de auditoría y perfilamiento de datos
│   ├── INFORME_MODELO_ML.md                 # Informe técnico exhaustivo de ingeniería y ML
│   └── README-de-demo.md                    # Documentación original del prototipo y guion
│
├── data/                                    # Repositorio de datos
│   ├── KaggleV2-May-2016.csv                # Dataset crudo original (110.527 filas)
│   └── citas_medicas_preprocesadas.csv      # Dataset limpio y depurado (110.521 filas)
│
├── models/                                  # Artefactos del modelo predictivo
│   ├── mejor_modelo_lgbm.joblib             # Modelo entrenado con 100% de datos (12.8 MB)
│   ├── metricas.json                        # Registro formal de métricas OOF y temporales
│   ├── mejores_hiperparametros.json         # Parámetros seleccionados por Optuna
│   └── README.md                            # Ficha técnica del modelo y código de inferencia
│
├── notebooks/                               # Cuadernos Jupyter secuenciales
│   ├── 01_Perfilamiento.ipynb               # 01: Calidad inicial y tipificación
│   ├── 02_Preprocesamiento.ipynb            # 02: Limpieza, filtros y exportación del CSV limpio
│   ├── 03_EDA.ipynb                         # 03: Análisis exploratorio de datos
│   ├── 04_Baselines.ipynb                   # 04: Modelos base de referencia
│   └── 05_Modelamiento.ipynb                # 05: Pipeline de modelamiento, OOF y exportación
│
└── scripts/                                 # Scripts modulares de Python
    ├── experimento_modelo.py                # Ingeniería de features y evaluación reutilizable
    ├── optimizar_modelo.py                  # Búsqueda bayesiana de hiperparámetros con Optuna
    └── entrenar_modelo.py                   # Entrenamiento final y serialización del joblib
```

---

## 🚀 Guía de Instalación y Reproducción

### 1. Ejecutar la Demo Interactiva (Sin Instalación)
Para explorar la demostración de gestión hospitalaria:
1. Haz doble clic en **[index.html](index.html)** para abrirlo en cualquier navegador web moderno.
2. No requiere servidor local, conexión a internet ni dependencias adicionales.

### 2. Configuración del Entorno de Machine Learning (Python)
Para reentrenar modelos o ejecutar los notebooks analíticos:

```powershell
# Clonar o situarse en la raíz del repositorio
cd prediccion-inasistencias

# Crear y activar entorno virtual con Python 3.12
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias exactas
pip install -r requirements.txt
```

### 3. Reproducción del Modelo Final
Para reentrenar el modelo LightGBM con el 100% de los datos y regenerar [`models/mejor_modelo_lgbm.joblib`](models/mejor_modelo_lgbm.joblib) y [`models/metricas.json`](models/metricas.json):

```powershell
.venv\Scripts\python.exe scripts\entrenar_modelo.py
```

### 4. Ejecución de Notebooks
El repositorio cuenta con la herramienta estandarizada `gestor-ipynb` para ejecutar los notebooks localmente desde la raíz:

```powershell
python .opencode/skills/gestor-ipynb/ipynb_tool.py run "notebooks/05_Modelamiento.ipynb"
```

---

## 📚 Documentación del Proyecto (`/docs`)

### Entrega consolidada del 23 de noviembre de 2026

- [Informe principal EP5/EP6 actualizado](output/pdf/Informe_Principal_EP5_EP6_Actualizado.pdf): ocho páginas con etapas, entregables, decisiones y contraste del plan con Jira.
- [Presentación para el cliente](output/presentacion/Defensa_Cliente_Marcos_Ajustados.pptx): diez diapositivas con A/D y notas para la defensa.
- [Registro actualizado de desviaciones](output/pdf/Registro_Control_Desviaciones_Actualizado.pdf): incidente T06/T07, respuesta, recuperación declarada y consolidación administrativa, sin horas de retraso.
- [Evidencias técnicas y de seguimiento](docs/evidencias/): capturas, métricas reproducidas y cortes de estado. Los enlaces a Jira requieren acceso al tablero.

Jira registra 14 de 15 tareas finalizadas. T13 sigue en curso porque los materiales están terminados y falta exponer. Las seis jornadas de seguimiento se documentan retrospectivamente; las fechas de ejecución declaradas se distinguen de las fechas automáticas del sistema.

El entrenamiento completo se reprodujo con Python 3.12.4 tras reparar la ruta del entorno local. Las métricas coinciden con las vigentes con tolerancia absoluta de 1e-9 y se conservaron las huellas del modelo, métricas e `index.html`. [Resultado de reproducción](docs/evidencias/reproduccion_entrenamiento_2026-11-23.json). Para repetir la comparación sin reemplazar los artefactos vigentes, desde la raíz:

```powershell
.venv\Scripts\python.exe docs\evidencias\verificar_reproduccion_2026-11-23.py
```

Toda la documentación técnica, estratégica y metodológica se encuentra centralizada en la carpeta [`docs/`](docs/):

* **[`docs/INFORME_GENERAL_PROYECTO.md`](docs/INFORME_GENERAL_PROYECTO.md):** Informe general de desarrollo, etapas, metodología, bitácora del sprint, contingencia del Día 4 y resultados.
* **[`docs/T01-documento.md`](docs/T01-documento.md):** Alcance, supuestos, matriz de riesgos (fuga de información, sesgos, transferencia de datos), restricciones operacionales y tablero Kanban del proyecto.
* **[`docs/T02-problema_user_y_kpis.md`](docs/T02-problema_user_y_kpis.md):** Caracterización del usuario, valor esperado y umbrales cuantitativos de KPIs (calidad, desempeño M1-M3, operación O1-O2, gobernanza G1-G4).
* **[`docs/T03-diccionario-datos.md`](docs/T03-diccionario-datos.md):** Ficha técnica de la fuente Kaggle y glosario de las 14 variables originales.
* **[`docs/T04-calidad-datos.md`](docs/T04-calidad-datos.md):** Reporte de auditoría de datos, tratamiento de edades atípicas y coherencia temporal de fechas.
* **[`docs/INFORME_MODELO_ML.md`](docs/INFORME_MODELO_ML.md):** Informe técnico completo del ciclo de vida del modelo ML, formulación matemática del suavizado bayesiano, curvas ROC/PR OOF e instrucciones de integración.
* **[`docs/README-de-demo.md`](docs/README-de-demo.md):** Guion y especificaciones del simulacro clínico original.

---

## ⚠️ Supuestos, Ética y Limitaciones

1. **Decisión Humana y No Cancelación:** El sistema es una herramienta de soporte a la decisión (*decision support system*). Bajo ningún motivo una predicción algorítmica debe derivar en la cancelación unilateral o reasignación de citas de un paciente.
2. **Contexto Geográfico de los Datos:** El dataset proviene de la red de salud pública de Vitória, Espírito Santo, Brasil (2016). Su aplicación en otros contextos (por ejemplo, el sistema de salud en Chile) constituye una prueba de concepto académica que requiere reentrenamiento y calibración con datos demográficos locales antes de un piloto clínico.
3. **Tratamiento de la Variable `SMS_received`:** El envío de SMS es una acción previa a la consulta. En escenarios donde el triage predictivo deba realizarse antes de programar recordatorios, el modelo debe evaluarse prescindiendo de dicha variable para evitar fugas de intervención.
4. **Predicción vs. Causalidad:** Un puntaje de alto riesgo (*scoring*) o un valor elevado de Lift@20 demuestran que el algoritmo concentra con éxito a los pacientes inasistentes; sin embargo, validar si el contacto telefónico reduce efectivamente la inasistencia requiere ensayos controlados aleatorizados (*A/B testing* en producción).

---

## 👥 Equipo

* **Alfredo Álvarez:** Ingeniería de datos, auditoría de calidad y perfilamiento exploratorio (EDA).
* **Diego Bustamante:** Modelamiento predictivo, optimización de hiperparámetros, validación cruzada y arquitectura de prototipo.
