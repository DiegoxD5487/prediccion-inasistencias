# Reporte de calidad del dataset

A continuación se detallan los hallazgos principales obtenidos durante la fase de perfilamiento y análisis de calidad de datos:

* **Valores Faltantes y Duplicados:** 
  * No se detectaron valores nulos (`NaN`) en ninguna de las columnas del dataset.
  * No existen filas completamente duplicadas dentro del conjunto de datos.

* **Inconsistencias en Tipos de Datos:**
  * La columna `PatientId` está registrada como tipo flotante (`float64`), cuando lo ideal para un identificador único es que sea entero (`int64`) o de tipo texto (`str`).
  * Las columnas `ScheduledDay` y `AppointmentDay` se encontraban inicialmente como texto (`object`), fue necesario convertirlas a formato fecha (`datetime64`) para su correcto análisis de calidad.

* **Valores únicos:**
  * AppointmentID como clave primaria válida (sin duplicados).
  * Neighbourhood con 81 valores únicos
  * Scheduled vs. Appointment: Gran diferencia en valores únicos.
  * Handcap como una escala de 5 valores únicos.

* **Inconsistencia de Fechas (Coherencia Temporal):**
  * Se identificaron **5 registros** donde la fecha de la cita (`AppointmentDay`) normalizada es estrictamente anterior a la fecha de agendamiento (`ScheduledDay`) normalizada, lo cual representa una anomalía temporal.

* **Valores Atípicos en la Edad (`Age`):**
  * Se detectaron **6 registros** con edades atípicas o dudosas fuera del rango esperado (edades negativas como `-1` o personas con edades extremadamente avanzadas que requieren validación, como `115` años).

* **Distribución de la Variable Objetivo (`No-show`):**
  * El dataset presenta un desbalance de clases: aproximadamente el **79.81%** de los pacientes sí asistieron a su cita (`No`), mientras que el **20.19%** faltaron (`Yes`).