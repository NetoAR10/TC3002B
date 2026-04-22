# TC3002B - Ernesto Acosta Ruiz A01364982
## Avance 1: Generación o selección del set de datos  y preprocesado

Dataset utilizado: Smartphone Usage and Addiction Analysis Dataset

Link a dataset: https://www.kaggle.com/datasets/zahranusratt/smartphone-usage-and-addiction-analysis-dataset/data

### Descripción

Este dataset contiene información detallada sobre los patrones de uso de teléfonos inteligentes e indicadores de adicción, recopilada de 7500 personas. Incluye variables como el tiempo diario de uso de pantalla, el número de aplicaciones abiertas, el tiempo dedicado a las redes sociales, la duración del sueño, los niveles de productividad y la frecuencia de consulta del teléfono. El conjunto de datos también puede incluir atributos demográficos como la edad, el género y la ocupación, lo que permite un análisis más profundo de cómo interactúan los diferentes grupos con los teléfonos inteligentes. El dataset cuenta con la variable predictora "addicted_label" haciendo que sea un problema de clasificación (1 = adicto, 0 = no adicto).

### Análisis y procesado de los datos
El dataset cuenta con 7500 intancias con 15 columnas de features y 1 de variable predictora. La variable predictora es una booleana, haciendo este dataset un problema de clasificación. La distribución de las clases es la siguiente:
* clase 0 (no adicto) = 29.23% = 2192 
* clase 1 (adicto) = 70.77% = 5308

No tenemos valores nulos en ninguna de las columnas entonces podemos pasar a la parte del procesado.

Las columnas que no son útiles para el análisis son "transaction_id" y "user_id" ya que estas son valores únicos y solo son identificadores. Estas columnas fueron eliminadas del dataset.

Las columnas no númericas son las siguientes: "gender", "stress_level", "addiction_level" y "academic_work_impact". Para la columna de "academic_work_impact" solo se transformaron sus valores de strings a números (Ej: "YES" -> 1, "NO" -> 0). Para las demás columnas no númericas se aplicó one-hot enconding ya que estas son multiclase.

Las columnas numéricas se dejaron, por el momento, se dejaron en su estado original.

También se generó una matriz de correlación para visualizar cque variables tienen más peso sobre la variable predictora "addicted_label".

<img width="1346" height="1227" alt="image" src="https://github.com/user-attachments/assets/0dead493-9773-470d-9c2f-1e6e42693aac" />

**Al final terminamos con 20 variables y una variable predictora.**

### Separación de los datos

Tomando en consideración que el dataset tiene un total de 7500 instancias, se decidió ir por el siguiente split:
* Train = 80% = 6000 instancias
* Validation = 10% = 750 instancias
* Test = 10% = 750 instancias

Se tómo la decisión de separa el dataset de esta forma ya que se quiere dar una buena cantidad de instancias al modelo para que pueda aprender de ambas clases y dejando una cantidad significativa para las pruebas.

## Avance 2: Implementación de Modelo y Evaluación inicial del modelo

### Cambios realizados
Se eliminó la columna "addiction_level" ya que hacía que el modelo aprendiera a la perfección en train, validation y test

### Descripción
Se utilizó un modelo red neuronal feedforward (FNN) para clasificación binaria con las siguientes capas:
<img width="738" height="174" alt="image" src="https://github.com/user-attachments/assets/8ca5a72e-b4c5-4715-9cff-4f5d50757fc7" />

La selección de este modelo se tomó usando de referencia como se muestra en [1] donde se usa una Feedforward Neural Network (FNN) para el análisis de la adicción de las personas con el internet. En esta investigación también se usan múltiples modelos de machine learning, como Random Forest y XGBoost, y según los resultados la red neuronal fué la que dió los mejores resultados.

La clasificación real del modelo utilizado es un Multilayer Perceptron (MLP) debido a que se usan 3 capas densas: una capa de entrada, una capa oculta y una capa de salida.

La primera capa Dense(16) aprende patrones básicos, la siguiente capa Dense(8) combina los patrones en en relaciones más complejas y la capa Dense(1) da la predicción final.

### Parámetros
* Optimizador = Adam --> Se encarga de ajustar los pesos de la red neuronal adaptando el "learning rate" de manera automática
* Función de pérdida = Binary Crossentropy --> Calcula el error de la predicción del modelo y el valor real
* Métrica = Accuracy --> Métrica de monitoreo que no afecta el entrenamiento pero muestra si hay overfitting comparando con el accuracy de validation

### Métricas de evaluación del modelo
Las métricas que se usaron para evaluar los modelos de clasificación referenciados en [1], son Accuracy, Precision, Recall, F1-Score y AUC-ROC. Para la evaluación del modelo vamos a usar accuracy sólo para validar que no haya overfitting al comparar el accuracy de train, validation y test. Las métricas de precision, recall y f1-score son las métricas que se utilizarán para la evaluación final del desempeño del modelo.

* Precision = Indica que tan confiables son los positivos de una clase
* Recall = Indica que tantos casos detecta en una clase
* F1-score = Balance entre precision y recall

Con estas métricas podemos tener un análisis mas exacto del rendimiento del modelo. De igual manera se cuenta con una matriz de confusión para ver el desglose del rendimiento de cada clase (TP, TN, FP, FN)

### Resultados inciales

===== Resultados de la última época de entrenamiento del modelo =====
<img width="1131" height="49" alt="image" src="https://github.com/user-attachments/assets/a0cf35ac-b6b6-4053-a8f3-f995ec86d208" />


===== Accuracy de test =====
<img width="363" height="25" alt="image" src="https://github.com/user-attachments/assets/6053bd31-fdf6-4046-8e3b-5898a782bdfa" />


===== Métricas de evaluación =====
<img width="522" height="399" alt="image" src="https://github.com/user-attachments/assets/f9d297b2-8e00-406d-b04c-7160eb06a83c" />

### Conclusiones inciales

* El modelo no muestra señales de overfitting ya que el accuracy de train, validation y test es muy similar
* El modelo tiene un buen desempeño inicial al tener un f1-score de 0.81 para la clase 0 (no adicción) y un f1-score de 0.92 para la clase 1 (adicción)
* Hay un desbalance de clases entonces la clase 0 tiene potencial de mejorar su rendimiento


## Referencias

[1] Wahed, S. A., & Wahed, M. A. (2025). AI-Driven Digital Well-being: Developing Machine Learning Model to Predict and Mitigate Internet Addiction. LatIA, 3, 134. https://doi.org/10.62486/latia2025134
