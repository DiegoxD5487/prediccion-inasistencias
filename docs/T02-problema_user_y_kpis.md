# Problema del negocio:
La inasistencia de personas a las citas agendadas en los centros médicos públicos, lo cual genera gastos operativos básicos, interrupción de tratamientos, périda de horas profesionales e insumos perecibles, retrasa diagnósticos, agrava enfermedades crónicas, aumenta las listas de espera y esto sobrecarga los servicios de urgencia.

* Usuario: Centro médico
* Valor esperado: Saber qué cita médica tiene altas probabilidades del paciente faltar. Esto permite hacer uso eficiente del tiempo al tomar una decisión en base a esto, ahorrar gastos y prevenir complicaciones derivadas de la falta a la cita médica.
* Etiqueta objetivo: No-Show
* Decisión que apoyará el modelo: Identificar posibles inasistencias a citas lo que permite saber a quien contactar para prevenir o reagendar.

# KPIs
**Son metas de referencia, no resultados medidos ni compromisos aprobados por un hospital.** Deben validarse con el futuro mandante y con la línea base real.

## 1. Calidad de los datos

*   **D1. Completitud de atributos críticos:** registros utilizables /
    total de registros.
    -      Cumple: **≥ 98%**
    -   Alerta: **95% a \< 98%**
    -   No cumple: **\< 95%**
-   **D2. Consistencia de etiquetas:** cierres válidos / citas
    etiquetadas.
    -   Cumple: **≥ 99%**
    -   Alerta: **97% a \< 99%**
    -   No cumple: **\< 97%**

## 2. Desempeño y equidad del modelo

-   **M1. Recall de inasistencia:** TP / (TP + FN), medido sobre el
    conjunto de prueba.
    -   Cumple: **≥ 0,65**
    -   Alerta: **0,60 a \< 0,65**
    -   No cumple: **\< 0,60**
-   **M2. Lift@20:** tasa de inasistencia del 20% de citas con mayor
    riesgo / tasa global de inasistencia del conjunto de prueba.
    -   Cumple: **≥ 1,50**
    -   Alerta: **1,20 a \< 1,50**
    -   No cumple: **\< 1,20**
-   **M3. Brecha máxima de recall entre grupos comparables:**
    -   Cumple: **≤ 10 puntos porcentuales (pp)**
    -   Alerta: **\> 10 a ≤ 15 pp**
    -   No cumple: **\> 15 pp**

## 3. Indicadores operativos

-   **O1. Cobertura:** citas priorizadas / citas elegibles.
    -   Cumple: **18% a 22%**
    -   Alerta: **15% a \< 18% o \> 22% a 25%**
    -   No cumple: **\< 15% o \> 25%**
-   **O2. Entregabilidad de SMS:** SMS confirmados como entregados / SMS
    enviados.
    -   Cumple: **≥ 95%**
    -   Alerta: **90% a \< 95%**
    -   No cumple: **\< 90%**

## 4. Impacto en la asistencia y costos

-   **I1. Reducción atribuible de inasistencia:** diferencia en la tasa
    de inasistencia del grupo prioritario frente a un grupo comparador
    válido.
    -   Cumple: **≥ 10 pp**
    -   Alerta: **5 a \< 10 pp**
    -   No cumple: **\< 5 pp**
-   **I2. Citas recuperadas atribuibles:** citas adicionales
    efectivamente atendidas entre las 2.000 citas contactadas.
    -   Cumple: **≥ 200**
    -   Alerta: **100 a 199**
    -   No cumple: **\< 100**
-   **I3. Costo total por cita recuperada atribuible (CLP, seis
    meses):**
    -   Cumple: **≤ \$60.000**
    -   Alerta: **\> \$60.000 a \$80.000**
    -   No cumple: **\> \$80.000 o sin mejora**

## 5. Gestión, presupuesto y seguridad

-   **G1. Hitos aceptados con evidencia:** hitos aceptados con evidencia
    / hitos vencidos.
    -   Cumple: **≥ 90%**
    -   Alerta: **80% a \< 90%**
    -   No cumple: **\< 80%**
-   **G2. Control presupuestario:** gastos reales / presupuesto aprobado
    acumulado.
    -   Cumple: **≤ 100%**
    -   Alerta: **\> 100% a 115%**
    -   No cumple: **\> 115%**
-   **G3. Cambios de cita basados exclusivamente en el score
    predictivo:**
    -   Cumple: **0 casos**
    -   No cumple: **1 o más casos**
    -   La predicción solo prioriza la revisión/contacto; no puede
        cambiar, cancelar ni reasignar citas automáticamente.
-   **G4. Incidentes críticos de confidencialidad o seguridad:**
    -   Cumple: **0 casos**
    -   No cumple: **1 o más casos**

## Consideraciones para interpretar los KPIs

-   Los KPIs **I1, I2 e I3 requieren una intervención real y un grupo comparador válido**. Si no existe piloto, deben informarse como **"NO MEDIDO"**, no como "NO CUMPLE".

\*El modelo como tal predice, no reduce las ausencias con SMS. De eso se encarga la solución en su conjunto.