# Proyecto Final — Machine Learning, Deep Learning, NLP y API

## 1. Descripción

Este repositorio integra un proyecto de análisis y predicción relacionado con uso de redes sociales, bienestar, salud mental y clasificación de texto.

La entrega contiene cuatro bloques técnicos principales:

1. **Preprocesado de datos con Scala + Apache Spark**.
2. **Modelos de Machine Learning** para bienestar/adicción y para GAD-7/PHQ-9.
3. **Modelos de Deep Learning** desarrollados en notebooks con PyTorch.
4. **NLP** para clasificación de texto y un experimento adicional de fine-tuning con Hugging Face + Unsloth y RAG.

Además, incorpora una **API FastAPI** que expone los modelos guardados en `models/` y una interfaz HTML de prueba.

> Este README se ha elaborado a partir de la estructura, código, notebooks, modelos serializados, datasets, `build.sbt`, `Dockerfile` y configuración incluida en la entrega.

---

## 2. Estructura del proyecto

```text
.
├── aplicacion/
│   ├── main.py
│   ├── index.html
│   └── requirements.txt
│
├── models/
│   ├── MA_GAD7_Classifier.pkl
│   ├── MA_GAD7_Regression.pkl
│   ├── MA_PHQ9_Classifier.pkl
│   ├── MA_PHQ9_Regression.pkl
│   ├── MWB_Behavioral_Clustering.pkl
│   ├── MWB_Classifier.pkl
│   ├── MWB_Regression.pkl
│   ├── MWB_Scaler.pkl
│   └── NLP_Classifier.pkl
│
├── Data/
│   ├── Preprocesado/
│   │   ├── dfNLP/
│   │   ├── social_media_addiction_mental_wellbeing/
│   │   └── social_media_mental_health/
│   └── Procesado/
│       ├── dfFinalMA_Test/
│       ├── dfFinalMA_Train/
│       ├── dfFinalMWB_Test/
│       ├── dfFinalMWB_Train/
│       └── dfNLP_procesado/
│
├── Preprocesado/
│   ├── build.sbt
│   ├── project/build.properties
│   └── src/main/scala/
│       ├── MentalHealth.scala
│       ├── MentalWellBeing.scala
│       └── Metodos.scala
│
├── Machine_Learning/
│   ├── ML_MA_Modelos_Escogidos/
│   ├── ML_MA_Modelos_Probados/
│   ├── ML_MWB_Modelos_Escogidos/
│   └── ML_MWB_Modelos_Probados/
│
├── Deep_Learning/
│   ├── DL_MA/
│   └── DL_MWB/
│
├── NLP/
│   ├── Preprocesamiento.ipynb
│   ├── Modelos.ipynb
│   └── Analisis.ipynb
│
├── IA/
│   └── IATFM.ipynb
│
├── Visualizaciones/
│   ├── Dashboard_Final/
│   └── Visualizaciones_Analisis_Interno/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

La entrega original también contiene artefactos generados por Git/SBT (por ejemplo `.git/` y `Preprocesado/target/`). No son dependencias de Python ni son necesarios para ejecutar la API.

---

## 3. Dependencias

### 3.1 Python

El `requirements.txt` de la raíz reúne las dependencias detectadas en todo el proyecto, no solo las necesarias para la API.

Incluye:

- FastAPI, Uvicorn y Pydantic para la API.
- NumPy, Pandas y Joblib.
- scikit-learn y XGBoost.
- Matplotlib, Seaborn y WordCloud.
- imbalanced-learn y Optuna.
- NLTK y Gensim.
- PyTorch.
- JupyterLab e IPykernel.
- Hugging Face `datasets` y `huggingface-hub`.
- TRL y Unsloth para el notebook `IA/IATFM.ipynb`.

### 3.2 Por qué `scikit-learn` está fijado a 1.9.0

Los archivos `.pkl` de `models/` contienen metadatos de serialización de scikit-learn **1.9.0**. Por ese motivo, el `requirements.txt` fija:

```text
scikit-learn==1.9.0
```

Esto es especialmente importante para evitar incompatibilidades al cargar `LogisticRegression`, `KMeans`, `SVR`, `StandardScaler`, `CountVectorizer` y el `Pipeline` de NLP.

El proyecto original tenía `scikit-learn>=1.3.0` en `aplicacion/requirements.txt`, pero para esta entrega se ha priorizado la compatibilidad con los modelos realmente incluidos.

### 3.3 Python recomendado

El `Dockerfile` incluido utiliza:

```text
python:3.11-slim
```

Por coherencia con el proyecto se recomienda **Python 3.11** para el entorno principal.

---

## 4. Instalación del entorno Python

Desde la raíz del proyecto:

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Comprueba el entorno:

```bash
python --version
pip list
```

---

## 5. Recursos de NLTK

Los notebooks de NLP utilizan al menos:

- `stopwords`
- `wordnet`

Aunque los notebooks contienen llamadas a `nltk.download(...)`, se puede dejar el entorno preparado manualmente:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

No se ha detectado uso directo de `punkt` en los notebooks incluidos.

---

## 6. Preprocesado con Scala + Spark

El preprocesado principal está implementado en:

```text
Preprocesado/src/main/scala/
```

El proyecto fija:

- **Scala 2.12.18**
- **Apache Spark 3.5.1**
- **sbt 2.0.6**
- Spark Core
- Spark SQL
- Spark MLlib
- Spark Avro
- ScalaTest 3.2.18 para tests

Estas dependencias son de Scala/JVM y **no se instalan con `pip`**.

### Requisitos adicionales

Para ejecutar esta parte hace falta:

- un JDK compatible con Spark 3.5.1;
- `sbt` 2.0.6;
- acceso a internet la primera vez para que sbt descargue dependencias.

La versión concreta de Java no está fijada en el repositorio, por lo que no se ha inventado una versión exacta en `requirements.txt`.

### Ejecutar el preprocesado

Desde la raíz:

```bash
cd Preprocesado
sbt "runMain MentalWellBeing"
sbt "runMain MentalHealth"
```

`MentalWellBeing` procesa:

```text
Data/Preprocesado/social_media_addiction_mental_wellbeing/social_media_addiction_mental_wellbeing.csv
```

y genera:

```text
Data/Procesado/dfFinalMWB_Train/MWB_Train.csv
Data/Procesado/dfFinalMWB_Test/MWB_Test.csv
```

`MentalHealth` procesa:

```text
Data/Preprocesado/social_media_mental_health/social_media_mental_health.csv
```

y genera:

```text
Data/Procesado/dfFinalMA_Train/MA_Train.csv
Data/Procesado/dfFinalMA_Test/MA_Test.csv
```

### Qué hace el preprocesado

`MentalWellBeing.scala`:

1. carga el dataset;
2. divide train/test con `0.8 / 0.2` y `seed=22`;
3. comprueba nulos;
4. imputa categóricas con `unknown`;
5. imputa variables numéricas mediante mediana;
6. convierte variables binarias a booleanos;
7. imputa valores binarios usando la moda calculada en train;
8. elimina filas restantes con nulos;
9. estandariza texto y redondea columnas decimales;
10. elimina outliers mediante IQR con factor `2`;
11. guarda train/test.

`MentalHealth.scala`:

1. carga el dataset;
2. divide train/test con `0.8 / 0.2` y `seed=22`;
3. transforma las variables booleanas;
4. normaliza texto;
5. aplica filtrado IQR con factor `1.5`;
6. guarda train/test.

---

## 7. Datasets incluidos

Los CSV principales detectados en la entrega son:

| Dataset | Filas aproximadas | Uso |
|---|---:|---|
| `social_media_addiction_mental_wellbeing.csv` | 1500 | Mental Wellbeing / Addiction |
| `social_media_mental_health.csv` | 8000 | GAD-7 / PHQ-9 |
| `sentiment_mental_health.csv` | 93661 | NLP |

También se incluyen datasets ya procesados:

```text
Data/Procesado/dfFinalMWB_Train/MWB_Train.csv
Data/Procesado/dfFinalMWB_Test/MWB_Test.csv
Data/Procesado/dfFinalMA_Train/MA_Train.csv
Data/Procesado/dfFinalMA_Test/MA_Test.csv

Data/Procesado/dfNLP_procesado/X_train.csv
Data/Procesado/dfNLP_procesado/X_test.csv
Data/Procesado/dfNLP_procesado/y_train.csv
Data/Procesado/dfNLP_procesado/y_test.csv
```

Por tanto, para usar la API no es necesario volver a generar los CSV: los resultados ya están incluidos en la entrega.

---

## 8. Machine Learning

### 8.1 Mental Wellbeing / Addiction — MWB

Los modelos finales incluidos son:

| Archivo | Tipo |
|---|---|
| `MWB_Classifier.pkl` | `LogisticRegression` |
| `MWB_Regression.pkl` | `SVR` |
| `MWB_Behavioral_Clustering.pkl` | `KMeans` |
| `MWB_Scaler.pkl` | `StandardScaler` |

Los tres primeros modelos principales reciben **31 variables** ya codificadas.

El conjunto de notebooks `ML_MWB_Modelos_Escogidos` cubre:

- regresión;
- clasificación;
- clustering.

La carpeta `ML_MWB_Modelos_Probados` incluye pruebas con KNN y XGBoost.

### 8.2 Mental Health — GAD-7

Modelo incluido:

```text
MA_GAD7_Classifier.pkl
MA_GAD7_Regression.pkl
```

Tipos detectados:

- `XGBClassifier`
- `XGBRegressor`

Ambos esperan **21 variables** codificadas.

### 8.3 Mental Health — PHQ-9

Modelos incluidos:

```text
MA_PHQ9_Classifier.pkl
MA_PHQ9_Regression.pkl
```

Tipos reales detectados en los `.pkl`:

- `LogisticRegression`
- `XGBRegressor`

El clasificador PHQ-9 **no es un XGBClassifier en el archivo serializado**, aunque el nombre/resumen del endpoint de `main.py` lo describe como XGBoost.

Ambos trabajan con **21 variables**.

---

## 9. Deep Learning

La carpeta `Deep_Learning/` contiene seis notebooks:

```text
Deep_Learning/DL_MA/DL_MA_GAD7.ipynb
Deep_Learning/DL_MA/DL_MA_GAD7_Classifier.ipynb
Deep_Learning/DL_MA/DL_MA_PHQ9.ipynb
Deep_Learning/DL_MA/DL_MA_PHQ9_Classifier.ipynb
Deep_Learning/DL_MWB/DL_MWB.ipynb
Deep_Learning/DL_MWB/DL_MWB_Classifier.ipynb
```

Usan principalmente:

- PyTorch;
- NumPy;
- Pandas;
- scikit-learn;
- Optuna;
- Matplotlib;
- Seaborn en los clasificadores.

Estos notebooks son experimentales/de entrenamiento. Los modelos que la API utiliza actualmente son los archivos `.pkl` de `models/`, no artefactos PyTorch.

---

## 10. NLP

La parte NLP está organizada en:

```text
NLP/Preprocesamiento.ipynb
NLP/Modelos.ipynb
NLP/Analisis.ipynb
```

### Flujo

1. `Preprocesamiento.ipynb`
   - limpia texto;
   - pasa a minúsculas;
   - elimina URLs;
   - elimina números;
   - elimina puntuación;
   - separa tokens;
   - elimina stopwords;
   - lematiza con WordNet;
   - genera `X_train`, `X_test`, `y_train`, `y_test`.

2. `Modelos.ipynb`
   - usa `CountVectorizer`;
   - entrena un clasificador;
   - guarda:
     ```text
     models/NLP_Classifier.pkl
     ```

3. `Analisis.ipynb`
   - analiza las categorías;
   - usa Word2Vec;
   - genera análisis y visualizaciones.

El modelo NLP serializado es un `sklearn.pipeline.Pipeline` con:

```text
CountVectorizer
    +
LogisticRegression
```

y las clases almacenadas son:

```text
Anxiety
Depression
Normal
```

El `CountVectorizer` incluido en el modelo utiliza hasta 20.000 características y n-gramas `(1, 2)`.

---

## 11. Experimento IA / Fine-tuning

`IA/IATFM.ipynb` es un bloque separado de fine-tuning con Hugging Face y Unsloth.

Utiliza:

- `unsloth`;
- `trl`;
- `datasets`;
- `huggingface_hub`;
- `torch`.

El notebook:

- solicita un token de Hugging Face;
- carga un modelo Qwen mediante Unsloth;
- trabaja con cuantización de 4 bits;
- usa LoRA/PEFT;
- entrena con `SFTTrainer`.

El código selecciona entre modelos:

```text
unsloth/Qwen3.5-2B
unsloth/Qwen3.5-9B
unsloth/Qwen3.5-27B
```

y en inferencia mueve los tensores explícitamente a:

```text
cuda
```

Por tanto, este notebook está orientado a un entorno con **GPU CUDA**. La instalación de Unsloth/PyTorch puede depender de la plataforma y del stack CUDA concreto; por eso debe considerarse un componente de entrenamiento avanzado, separado de la ejecución normal de la API.

---

## 12. API FastAPI

La API está en:

```text
aplicacion/main.py
```

El frontend está en:

```text
aplicacion/index.html
```

### Arranque correcto

Es importante ejecutar Uvicorn **desde la carpeta `aplicacion/`**, porque `main.py` utiliza rutas relativas como:

```text
../models/...
```

Comandos recomendados:

```bash
cd aplicacion
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

La interfaz HTML llama explícitamente al backend en:

```text
http://127.0.0.1:8001
```

por lo que **8001 es el puerto esperado por la interfaz actual**.

### URLs

Interfaz:

```text
http://127.0.0.1:8001/app
```

Documentación Swagger:

```text
http://127.0.0.1:8001/docs
```

OpenAPI:

```text
http://127.0.0.1:8001/openapi.json
```

Endpoint raíz:

```text
http://127.0.0.1:8001/
```

---

## 13. Endpoints de la API

| Método | Endpoint | Modelo / función |
|---|---|---|
| GET | `/` | estado de la API |
| GET | `/app` | interfaz HTML |
| POST | `/predict-class` | clasificación de adicción MWB |
| POST | `/predict-score` | regresión MWB |
| POST | `/cluster-user` | clustering KMeans MWB |
| POST | `/predict-gad7` | clasificación GAD-7 |
| POST | `/predict-gad7-score` | regresión GAD-7 |
| POST | `/predict-phq9` | clasificación PHQ-9 |
| POST | `/predict-phq9-score` | regresión PHQ-9 |
| POST | `/predict-nlp` | clasificación de texto |

### Entrada MWB

El esquema usa 18 campos de entrada:

```text
Age
Gender
Occupation
Relationship_Status
Primary_Platform
Daily_Usage_Hours
Platforms_Used_Count
Posts_Per_Week
Late_Night_Usage
First_Check_Morning
Notifications_Per_Day
Scroll_Without_Purpose
Tried_To_Cut_Back
Failed_To_Cut_Back
Sleep_Hours
Offline_Relationship_Quality
Physical_Activity_Hrs_Week
Screen_Free_Time_Hrs
```

### Entrada GAD-7 / PHQ-9

El esquema usa 10 campos:

```text
Age
Gender
User_Archetype
Primary_Platform
Daily_Screen_Time_Hours
Dominant_Content_Type
Activity_Type
Late_Night_Usage
Social_Comparison_Trigger
Sleep_Duration_Hours
```

### Entrada NLP

```json
{
  "text": "I feel pressure in my chest and I struggle to sleep."
}
```

---

## 14. Frontend

`aplicacion/index.html` es una interfaz HTML/JavaScript que usa:

- Bootstrap 5.3.2 desde CDN;
- jQuery 3.7.1 desde CDN.

No hay un proyecto Node/React/Vue en la entrega, por lo que **no se necesita `npm install`**.

Para que la interfaz se vea correctamente debe existir acceso a Internet para cargar estos recursos CDN.

---

## 15. Docker

Existe un `Dockerfile`, pero **no coincide exactamente con la estructura actual de esta entrega**.

El Dockerfile intenta copiar:

```text
app/requirements.txt
app/
models/
```

mientras que el proyecto contiene:

```text
aplicacion/requirements.txt
aplicacion/
models/
```

Por tanto, **el Dockerfile incluido no es directamente reproducible desde esta estructura sin ajustar las rutas**.

Además, `main.py` depende de rutas relativas, por lo que cualquier imagen Docker debe conservar una estructura equivalente a:

```text
/app/main.py
/models/*.pkl
```

o adaptar las rutas del código.

### Lo que sí se puede tomar del Dockerfile

El proyecto está planteado para:

```text
Python 3.11
```

y ejecutar:

```text
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
```

Si se utiliza un puerto distinto de `8001`, habrá que actualizar también `API_BASE_URL` en `aplicacion/index.html`.

---

## 16. Modelos y compatibilidad

Los modelos actuales son artefactos serializados con Joblib/Pickle.

Inventario:

```text
models/MA_GAD7_Classifier.pkl
models/MA_GAD7_Regression.pkl
models/MA_PHQ9_Classifier.pkl
models/MA_PHQ9_Regression.pkl
models/MWB_Behavioral_Clustering.pkl
models/MWB_Classifier.pkl
models/MWB_Regression.pkl
models/MWB_Scaler.pkl
models/NLP_Classifier.pkl
```

No se deben borrar ni mover si se quiere ejecutar la API sin modificar las rutas de `main.py`.

---

## 17. Observaciones técnicas importantes detectadas

### 17.1 El scaler guardado no se utiliza actualmente en la API

Existe:

```text
models/MWB_Scaler.pkl
```

pero `main.py` crea un nuevo:

```python
StandardScaler()
```

y ejecuta:

```python
fit_transform(...)
```

sobre una única petición.

Eso significa que cada petición recalcula el escalado con esa única fila. En un `StandardScaler`, una sola observación produce valores estandarizados de referencia cero para las columnas numéricas.

Por tanto, **el comportamiento de inferencia de la API no coincide con el uso normal de un scaler ajustado sobre el dataset de entrenamiento**. Esto está documentado aquí como una observación del código incluido, no como una modificación aplicada.

Lo mismo ocurre en `preprocesar_y_escalar_datos_21()` para GAD-7/PHQ-9.

### 17.2 Rutas relativas de los modelos

El código utiliza:

```text
../models/...
```

por lo que el directorio de trabajo importa. Ejecutar desde `aplicacion/` es la forma prevista por el código actual.

### 17.3 Descripción de PHQ-9 frente al modelo real

El endpoint `/predict-phq9` se etiqueta en `main.py` como XGBoost, pero el `.pkl` entregado es un:

```text
sklearn.linear_model.LogisticRegression
```

La documentación de este README refleja el tipo real del artefacto serializado.

### 17.4 CORS

La API configura CORS con:

```text
allow_origins=["*"]
allow_credentials=True
```

Esto facilita pruebas locales, pero debe revisarse antes de exponer la aplicación públicamente.

### 17.5 Datos sensibles / salud mental

El proyecto procesa variables y texto relacionados con salud mental. Antes de desplegarlo en un entorno real se debería revisar privacidad, protección de datos, almacenamiento de entradas, trazabilidad y uso clínico. Las salidas del modelo no deberían interpretarse automáticamente como diagnóstico médico sin la validación y supervisión correspondientes.

---

## 18. Orden recomendado para reproducir el proyecto

### Solo ejecutar la aplicación

No es necesario reentrenar ni reprocesar:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd aplicacion
uvicorn main:app --host 127.0.0.1 --port 8001
```

Abrir:

```text
http://127.0.0.1:8001/app
```

### Reproducir todo el pipeline

1. Instalar Python y `requirements.txt`.
2. Instalar Java/JDK + sbt.
3. Ejecutar `Preprocesado`.
4. Ejecutar `NLP/Preprocesamiento.ipynb`.
5. Ejecutar los notebooks de ML.
6. Ejecutar los notebooks de Deep Learning cuando corresponda.
7. Ejecutar `NLP/Modelos.ipynb`.
8. Verificar/actualizar los `.pkl` de `models/`.
9. Ejecutar la API desde `aplicacion/`.
10. Ejecutar las visualizaciones y el dashboard Power BI.

---

## 19. Power BI

La entrega contiene:

```text
Visualizaciones/Dashboard_Final/Dashboards_Mental_Healt.pbix
```

`requirements.txt` no puede instalar Power BI Desktop.

Este artefacto debe abrirse con una instalación compatible de **Microsoft Power BI Desktop** en el entorno correspondiente.

---

## 20. Resumen de dependencias no Python

| Componente | Tecnología | Configuración detectada |
|---|---|---|
| Preprocesado | Scala | 2.12.18 |
| Preprocesado | Apache Spark | 3.5.1 |
| Build Scala | sbt | 2.0.6 |
| Dashboard | Power BI | `.pbix`, instalación externa |
| Frontend | Bootstrap | 5.3.2 vía CDN |
| Frontend | jQuery | 3.7.1 vía CDN |

---

## 21. Comprobación rápida

Una vez instalada la aplicación:

```bash
cd aplicacion
uvicorn main:app --host 127.0.0.1 --port 8001
```

Comprueba:

```text
GET http://127.0.0.1:8001/
GET http://127.0.0.1:8001/docs
GET http://127.0.0.1:8001/app
```

En `/docs` se pueden probar los ocho endpoints de predicción/segmentación sin utilizar el frontend.

---

## 22. Archivos generados para esta entrega

En la raíz deben quedar:

```text
requirements.txt
README.md
```

El `requirements.txt` de la raíz es el fichero unificado para el proyecto completo. El `aplicacion/requirements.txt` original puede conservarse como referencia específica de la API, pero no cubre las dependencias de los notebooks de ML, DL, NLP y fine-tuning.
