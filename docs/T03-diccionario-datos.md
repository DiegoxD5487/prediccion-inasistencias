# Fuente de los datos
* Link: https://www.kaggle.com/datasets/joniarroba/noshowappointments
* Título: Medical Appointment No Shows
* Fuente: Kaggle
* Fecha: Subida el 20 agosto del 2017
* Version: 5
* 110.527 filas, 14 columnas

# Diccionario de variables

01. PatientId           : Identificador único de cada paciente.
02. AppointmentID       : Identificador único de cada cita médica.
03. Gender              : Género del paciente ('F' / 'M'). Mayor proporción de mujeres en el dataset.
04. AppointmentDay: Fecha real en la que el paciente debe asistir a la consulta médica.
05. ScheduledDay     : Fecha en la que el paciente agendó/registró la cita (anterior a la consulta).
06. Age                 : Edad del paciente.
07. Neighbourhood       : Ubicación/Barrio donde se lleva a cabo la cita médica.
08. Scholarship         : booleano/int (0 o 1). Beneficiario del programa Bolsa Família (ver Wikipedia).
09. Hipertension        : booleano/int (0 o 1). Indica si el paciente padece hipertensión.
10. Diabetes            : booleano/int (0 o 1). Indica si el paciente padece diabetes.
11. Alcoholism          : booleano/int (0 o 1). Indica si el paciente sufre de alcoholismo.
12. Handcap             : booleano/int (0 o más). Nivel/presencia de discapacidad.
13. SMS_received        : int (0 o 1+). Cantidad de recordatorios SMS enviados al paciente.
14. No-show             : 'No' / 'Yes' (o booleano). 'Yes' indica que el paciente NO asistió a la cita.

\*Aclaración de que el dataset es de Brasil, y por ello sirve para el prototipo y no como una demostración completamente fidedigna.