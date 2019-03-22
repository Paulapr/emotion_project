
# <--> **FINAL PROJECT** <-->
============================================


## **DESCRIPCIÓN**:

-- Sistema que predice a través de un registro EEG la característica emocional o afectiva que provoca una imagen medida como arousal y valencia.

-- Entrenamiendo del modelo mediante una base de datos de imágenes, creada por la universidad de Ginebra (GAPED), 730 imágenes con una valor de arousal y valencia asociado.

file:///home/paula/Escritorio/Final_project/Eje%20arousal-valencia.jpg

-- Se entrenará al modelo con los datos EEG del registro y los valores de valencia y arousal asociados a las imágenes.

## **HERRAMIENTAS**

- Muse headband:
	- Sample rate: 256 hz
	- 4 canales: AF7, AF8, TP9, TP10 
	- Noise supression

- Libreria python 'MuseLSL':
	. Conexión Muse headset - pc via bluetooh
	- Visualización del registro en tiempo real (ver si hay buena conexión)
	- Grabación de los datos del registro EEG, se guardan en formato csv (dos canales).

- Librerias: 
	- Scipy: Filtrar y procesar la señal EEG
	- Scikit learn: machine learninig models

- Dataset imágenes:
	- The Geneva Affective Picture Database (GAPED) - 730 imágenes divididas en 6 categorías  con un valor de arousal y uno de valencia asignado


## **PASOS SEGUIDOS**

- Visualización de imágenes:
	- Imágenes aleatorias
	- 10 segundos por imagen
	- 2 segundos con un fotograma negro entre imágenes. 
	- Total --> Aproximadamente 146 minutos (dividido en 10 visionados)
	- Registro EEG durante el visionado de imágenes
	- Sincronización entre el timestamp del registro y el de inicio y fin de la imagen. 
	
- Procesado del registro EEG:
	- Filtro butter entre 1-49 Hz
	- Epochs de 2 segundos solapados 1 segundo -> 9 ventanas temporales por imagen
	- FFT: 8 bins
	- Aplicación de función de preprocesado min-max-Scaler

- El dataset final:
	- 730 imágenes* 9 ventanas temporales* 8 bins de frecuencia * 4 canales
	- Rows: 730
	- Features: 288 columnas
	- Labels: 2 (Arousal y valencia)
	- Train 80%, test 20%

- Modelos de machine learning - Multitarget Regression:
	- Entrenados en un inicio: Random Forest, Decision tree regression, SVR, Gradient Boosting Regressor.
	- Mejores resultados con: Random Forest, Decision tree regression

- Probar el modelo con nuevas imágenes:
	- 10 imágenes
	- Registro EEG
	- Aplicar el modelo y ver los valores de arousal y valencia obtenidos
