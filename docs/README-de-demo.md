# Hospital Aurora · Simulacro de gestión de citas

Extrae el ZIP y abre **Agenda-hospitalaria.html** con doble clic en Chrome, Edge o Firefox actualizado. Usa **F11** para proyectar en Windows. El HTML incluye el modelo, los casos, los estilos y el código. Una vez descargado funciona sin internet, instalación, cuenta ni servidor.

## Guion de tres minutos

1. En **Agenda hospitalaria**, presenta el turno de la mañana del 20 de mayo de 2016. Hay doce pacientes ficticios. Los indicadores muestran el número de citas, cuántas superan el umbral y cuántos pacientes tienen al menos un contacto directo registrado.
2. Filtra por **Prioritarias**, pulsa **Por hora** para ordenar por riesgo y selecciona a **Clara Azevedo**. Su estimación original es aproximadamente 57,4 %. Pulsa **Simular SMS** y revisa su bitácora. Después usa **Registrar contacto → Asistencia confirmada**. Observa el indicador de contactos. **Dar seguimiento** permite registrar una gestión pendiente.
3. Desde la ficha pulsa **Explicar este puntaje**. Lee la probabilidad de Clara y compara con el umbral para revisión. Cambia la anticipación a cero y observa el nuevo puntaje y la diferencia en puntos porcentuales. Los escenarios dejan intacta la cita original. La sección «¿Qué influyó más en su puntaje?» identifica los tres grupos de variables con mayor contribución a la predicción original.
4. Cambia el criterio de **Máximo F1** a **Recall objetivo del 65 %**. Los indicadores históricos y el umbral cambian. El puntaje original depende de los datos de la cita. Vuelve a la ficha y prueba el filtro **En seguimiento**.
5. Para preparar otra presentación usa **Reiniciar simulacro** y confirma. Las gestiones se conservan en el navegador cuando el almacenamiento local está disponible. Al bloquearlo, se mantienen durante la sesión.

## Qué representa el prototipo

La agenda, los nombres, las especialidades, las horas de atención y las gestiones son ficticios. Los botones registran acciones dentro del simulacro. Las citas siguen programadas y el equipo mantiene la decisión de contacto. El riesgo representa una estimación cercana a la cita, con el estado previo del SMS conocido. Las gestiones posteriores quedan en la bitácora. El dato de SMS previo al cálculo se conserva para poder explicar el puntaje original.

## Modelo, explicaciones y métricas

La aplicación ejecuta los 1.050 árboles del artefacto mejor_modelo_lgbm.joblib. Utiliza sus 38 variables, su orden y sus categorías. La clase positiva es inasistencia. La jornada usa el período del dataset brasileño Kaggle 2016 y las reservas se simulan a las 07:00. Los perfiles con historial sitúan la última cita nueve días antes. Las condiciones de salud no declaradas en los casos ficticios quedan en cero.

Los aportes SHAP de cada ficha se calcularon con LightGBM sobre el caso original. Se sumaron por grupos de variables y se ordenaron por valor absoluto. Se muestran en unidades log-odds. El signo indica dirección respecto del valor de base del modelo. La comparación hipotética vuelve a evaluar todos los árboles con los datos modificados, incluidas las variables derivadas del día de reserva. Un cambio observado en el puntaje describe el comportamiento del modelo. La eficacia de una intervención requiere un estudio específico.

El archivo metricas.json del proyecto contiene las métricas históricas por umbral, calculadas en validación cruzada de cinco grupos por paciente. Las tarjetas de desempeño describen esa evaluación, mientras que los contadores de la agenda describen la jornada ficticia.
