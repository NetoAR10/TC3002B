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


### Separación de los datos

Tomando en consideración que el dataset tiene un total de 7500 instancias, se decidió ir por el siguiente split:
* Train = 80% = 6000 instancias
* Validation = 10% = 750 instancias
* Test = 10% = 750 instancias

Se tómo la decisión de separa el dataset de esta forma ya que se quiere dar una buena cantidad de instancias al modelo para que pueda aprender de ambas clases y dejando una cantidad significativa para las pruebas.
