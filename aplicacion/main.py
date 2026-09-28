from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Ocultar advertencias de versiones
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

tags_metadata = [
    {"name": "ml_predictions", "description": "Endpoints de predicción con Machine Learning (Scikit-Learn y XGBoost)."},
    {"name": "ml_clustering", "description": "Endpoints de aprendizaje no supervisado (Segmentación/Clustering)."},
    {"name": "nlp_predictions", "description": "Endpoints de clasificación de texto con procesamiento de lenguaje natural (NLP)."},
]

# Instancia global principal
app = FastAPI(
    title="API Unificada de Diagnóstico, Salud Mental y Segmentación",
    description="Backend FastAPI con escalado de características integrado para clasificadores, regresores, clustering y NLP.",
    version="3.6.0",
    openapi_tags=tags_metadata
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de los modelos PKL dentro de la carpeta models/
PATH_CLASSIFIER = "../models/MWB_Classifier.pkl"
PATH_REGRESSOR = "../models/MWB_Regression.pkl"
PATH_GAD7_XGB = "../models/MA_GAD7_Classifier.pkl"
PATH_GAD7_REG = "../models/MA_GAD7_Regression.pkl"
PATH_PHQ9_XGB = "../models/MA_PHQ9_Classifier.pkl"
PATH_PHQ9_REG = "../models/MA_PHQ9_Regression.pkl"
PATH_KMEANS = "../models/MWB_Behavioral_Clustering.pkl"
PATH_NLP = "../models/NLP_Classifier.pkl"
PATH_MWB_SCALER = "../models/Scalers/MWB_Scaler.pkl"
PATH_MA_SCALER = "../models/Scalers/MA_Scaler.pkl"

def cargar_modelo(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"Error cargando {path}: {e}")
        return None

# Carga global de modelos
modelo_clasificador = cargar_modelo(PATH_CLASSIFIER)
modelo_regresor = cargar_modelo(PATH_REGRESSOR)
modelo_gad7 = cargar_modelo(PATH_GAD7_XGB)
modelo_gad7_reg = cargar_modelo(PATH_GAD7_REG)
modelo_phq9 = cargar_modelo(PATH_PHQ9_XGB)
modelo_phq9_reg = cargar_modelo(PATH_PHQ9_REG)
modelo_kmeans = cargar_modelo(PATH_KMEANS)
modelo_nlp = cargar_modelo(PATH_NLP)
scaler_mwb = cargar_modelo(PATH_MWB_SCALER)
scaler_ma = cargar_modelo(PATH_MA_SCALER)

CLASES_MAPA = {0: "High", 1: "Low", 2: "Moderate", 3: "Severe"}

PHQ9_CLASES_MAPA = {
    0: "NONE-MINIMAL",
    1: "MILD",
    2: "MODERATE",
    3: "MODERATELY SEVERE",
    4: "SEVERE"
}
# ==========================================
# DICCIONARIO DE DESCRIPCIÓN DE CLUSTERS
# ==========================================

DESCRIPCIONES_CLUSTERS = {
    0: {
        "nombre": "El Multi-Plataforma / Omnívoro Digital",
        "perfil": "Diversificado con Buen Descanso",
        "patron": "Uso del doble de plataformas que el promedio (5.47 redes frente a 2.5–3.2).",
        "impacto": "Uso diario moderado (4.85 h) y el mejor nivel de descanso (7.32 h de sueño)."
    },
    1: {
        "nombre": "El Conectado Social Intencional",
        "perfil": "Uso Activo y Funcional",
        "patron": "Alto volumen de notificaciones (78.22), pero con el menor tiempo de scroll pasivo (3.90).",
        "impacto": "Mantiene la mayor calidad de relaciones en el mundo real (7.45)."
    },
    2: {
        "nombre": "El Pasivo / Desconectado",
        "perfil": "Mayor Edad con Bajo Engagement Digital",
        "patron": "Grupo de mayor edad (35 años) con mínimas notificaciones (38.26) y baja publicación (3.48/semana).",
        "impacto": "Déficit en el descanso (5.96 h de sueño) y poca actividad física no vinculada al uso de redes."
    },
    3: {
        "nombre": "El Joven Activo / Equilibrado",
        "perfil": "Hábitos Saludables y Bienestar",
        "patron": "Grupo más joven (23 años) que compensa el consumo digital con hábitos saludables.",
        "impacto": "Líder en actividad física semanal (3.89 h) y tiempo libre de pantallas (4.07 h)."
    },
    4: {
        "nombre": "El Usuario en Riesgo / Compulsivo",
        "perfil": "Alto Riesgo de Adicción",
        "patron": "Máximo uso diario (5.17 h), alto scroll sin propósito (6.62) y bombardeo de notificaciones (91.31).",
        "impacto": "Peor calidad en relaciones offline (4.20); perfil objetivo para el predictor de adicción."
    }
}
@app.on_event("startup")
def cargar_sistema_rag():
    global cadena_rag
    try:
        if not os.path.exists("../models/IA/chroma_db"):
            print(f"⚠️ Alerta: No se encontró la carpeta '{"../models/IA/chroma_db"}'. RAG no estará disponible.")
            return

        print("1. Cargando embeddings y Chroma DB desde disco...")
        modelo_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vectorstore = Chroma(
            persist_directory="../models/IA/chroma_db",
            embedding_function=modelo_embeddings
        )

        print("2. Configurando Retriever y Reranker...")
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        model_rerank = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model_rerank, top_n=3)

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

        print("3. Conectando con Ollama (llama3.1)...")
        llm = ChatOllama(model="llama3.1", temperature=0)

        cadena_rag = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        print("✅ Sistema RAG listo para recibir consultas.")

    except Exception as e:
        print(f"❌ Error al inicializar el RAG: {e}")
# ==========================================
# ESQUEMAS PYDANTIC (VALIDACIÓN Y DOCS)
# ==========================================

# Esquema 1: Para modelos de 18 entradas (MWB Classifier, SVR y KMeans)
class InputData(BaseModel):
    Age: float = Field(..., example=25.0)
    Gender: Literal["MALE", "FEMALE", "NON-BINARY", "UNKNOWN"] = Field(..., example="MALE")
    Occupation: Literal["STUDENT", "UNEMPLOYED", "WORKING PROFESSIONAL", "UNKNOWN"] = Field(..., example="STUDENT")
    Relationship_Status: Literal["MARRIED", "SINGLE", "UNKNOWN"] = Field(..., example="SINGLE")
    Primary_Platform: Literal["INSTAGRAM", "SNAPCHAT", "TIKTOK", "TWITTER/X", "YOUTUBE", "UNKNOWN"] = Field(..., example="INSTAGRAM")
    Daily_Usage_Hours: float = Field(..., example=4.5)
    Platforms_Used_Count: int = Field(..., example=3)
    Posts_Per_Week: int = Field(..., example=5)
    Late_Night_Usage: bool = Field(..., description="True para Sí, False para No", example=True)
    First_Check_Morning: Literal["AFTER 30 MIN", "WITHIN 5 MIN", "OTHER"] = Field(..., example="WITHIN 5 MIN")
    Notifications_Per_Day: int = Field(..., example=50)
    Scroll_Without_Purpose: float = Field(..., description="", example=5.0)
    Tried_To_Cut_Back: bool = Field(..., description="True para Sí, False para No", example=True)
    Failed_To_Cut_Back: bool = Field(..., description="True para Sí, False para No", example=True)
    Sleep_Hours: float = Field(..., example=7.0)
    Offline_Relationship_Quality: float = Field(..., example=8.0)
    Physical_Activity_Hrs_Week: float = Field(..., example=3.0)
    Screen_Free_Time_Hrs: float = Field(..., example=2.0)


# Esquema 2: Para modelos de 10 entradas (GAD-7 Classifier/Regressor y PHQ-9 Classifier/Regressor)
class GAD7InputData(BaseModel):
    Age: float = Field(..., example=22.0, description="Edad del usuario")
    Gender: Literal["MALE", "FEMALE", "NON-BINARY", "UNKNOWN"] = Field(..., example="MALE")
    User_Archetype: Literal["DIGITAL MINIMALIST", "HYPER-CONNECTED", "PASSIVE SCROLLER", "OTHER"] = Field(..., example="HYPER-CONNECTED")
    Primary_Platform: Literal["INSTAGRAM", "LINKEDIN", "SNAPCHAT", "TIKTOK", "TWITTER/X", "YOUTUBE", "UNKNOWN"] = Field(..., example="INSTAGRAM")
    Daily_Screen_Time_Hours: float = Field(..., example=5.5, description="Horas de pantalla diarias")
    Dominant_Content_Type: Literal["ENTERTAINMENT/COMEDY", "GAMING", "LIFESTYLE/FASHION", "NEWS/POLITICS", "SELF-HELP/MOTIVATION", "OTHER"] = Field(..., example="ENTERTAINMENT/COMEDY")
    Activity_Type: Literal["PASSIVE", "ACTIVE", "OTHER"] = Field(..., example="PASSIVE")
    Late_Night_Usage: bool = Field(..., description="True para Sí, False para No", example=True)
    Social_Comparison_Trigger: bool = Field(..., description="True para Sí, False para No", example=True)
    Sleep_Duration_Hours: float = Field(..., example=6.0, description="Horas de sueño diarias")


# Esquema 3: Para la predicción con Texto NLP
class NLPInputData(BaseModel):
    text: str = Field(..., example="I feel pressure in my chest and I struggle to sleep.")

#Esquema 4: Para el RAG
class RAGInputData(BaseModel):
    question: str = Field(
        ..., 
        description="Pregunta o texto enviado al RAG", 
        example="¿Cuáles son los síntomas del uso compulsivo de redes sociales?"
    )

# ==========================================
# FUNCIONES DE PREPROCESAMIENTO Y ESCALADO
# ==========================================

MWB_NUMERIC_COLUMNS = [
    "Age",
    "Daily_Usage_Hours",
    "Platforms_Used_Count",
    "Posts_Per_Week",
    "Notifications_Per_Day",
    "Scroll_Without_Purpose",
    "Sleep_Hours",
    "Offline_Relationship_Quality",
    "Physical_Activity_Hrs_Week",
    "Screen_Free_Time_Hrs"
]

MA_NUMERIC_COLUMNS = [
    "Age",
    "Daily_Screen_Time_Hours",
    "Sleep_Duration_Hours"
]


# ==========================================================
# PREPROCESADO MWB - 18 ENTRADAS
# ==========================================================

def preprocesar_y_escalar_datos_18(datos: InputData) -> pd.DataFrame:

    if scaler_mwb is None:
        raise RuntimeError(
            f"No se encontró el scaler entrenado: {PATH_MWB_SCALER}"
        )

    # Convertimos los datos recibidos a DataFrame
    df = pd.DataFrame([datos.model_dump()])

    # ------------------------------------------------------
    # 1. Asegurar que las variables numéricas sean numéricas
    # ------------------------------------------------------

    for col in MWB_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------
    # 2. Aplicar el scaler ENTRENADO
    #
    # IMPORTANTE:
    # NO usar fit()
    # NO usar fit_transform()
    #
    # Solo transform()
    # ------------------------------------------------------

    df[MWB_NUMERIC_COLUMNS] = scaler_mwb.transform(
        df[MWB_NUMERIC_COLUMNS]
    )

    # ------------------------------------------------------
    # 3. One-Hot Encoding
    # ------------------------------------------------------

    df_encoded = pd.get_dummies(
        df,
        columns=[
            "Gender",
            "Occupation",
            "Relationship_Status",
            "Primary_Platform",
            "First_Check_Morning"
        ]
    )

    # ------------------------------------------------------
    # 4. Columnas EXACTAS que espera el modelo
    # ------------------------------------------------------

    columnas_esperadas = [
        "Age",
        "Daily_Usage_Hours",
        "Platforms_Used_Count",
        "Posts_Per_Week",
        "Late_Night_Usage",
        "Notifications_Per_Day",
        "Scroll_Without_Purpose",
        "Tried_To_Cut_Back",
        "Failed_To_Cut_Back",
        "Sleep_Hours",
        "Offline_Relationship_Quality",
        "Physical_Activity_Hrs_Week",
        "Screen_Free_Time_Hrs",

        "Gender_MALE",
        "Gender_NON-BINARY",
        "Gender_UNKNOWN",

        "Occupation_STUDENT",
        "Occupation_UNEMPLOYED",
        "Occupation_UNKNOWN",
        "Occupation_WORKING PROFESSIONAL",

        "Relationship_Status_MARRIED",
        "Relationship_Status_SINGLE",
        "Relationship_Status_UNKNOWN",

        "Primary_Platform_INSTAGRAM",
        "Primary_Platform_SNAPCHAT",
        "Primary_Platform_TIKTOK",
        "Primary_Platform_TWITTER/X",
        "Primary_Platform_UNKNOWN",
        "Primary_Platform_YOUTUBE",

        "First_Check_Morning_AFTER 30 MIN",
        "First_Check_Morning_WITHIN 5 MIN"
    ]

    # ------------------------------------------------------
    # 5. Asegurar exactamente las columnas del entrenamiento
    # ------------------------------------------------------

    df_encoded = df_encoded.reindex(
        columns=columnas_esperadas,
        fill_value=0
    )

    return df_encoded


# ==========================================================
# PREPROCESADO GAD-7 / PHQ-9 - 10 ENTRADAS
# ==========================================================

def preprocesar_y_escalar_datos_21(datos: GAD7InputData) -> pd.DataFrame:

    if scaler_ma is None:
        raise RuntimeError(
            f"No se encontró el scaler entrenado: {PATH_MA_SCALER}"
        )

    # Convertimos los datos recibidos a DataFrame
    df = pd.DataFrame([datos.model_dump()])

    # ------------------------------------------------------
    # 1. Asegurar que las variables numéricas sean numéricas
    # ------------------------------------------------------

    for col in MA_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------
    # 2. Aplicar el scaler ENTRENADO
    # ------------------------------------------------------

    df[MA_NUMERIC_COLUMNS] = scaler_ma.transform(
        df[MA_NUMERIC_COLUMNS]
    )

    # ------------------------------------------------------
    # 3. One-Hot Encoding
    # ------------------------------------------------------

    df_encoded = pd.get_dummies(
        df,
        columns=[
            "Gender",
            "User_Archetype",
            "Primary_Platform",
            "Dominant_Content_Type",
            "Activity_Type"
        ]
    )

    # ------------------------------------------------------
    # 4. Columnas EXACTAS del entrenamiento
    # ------------------------------------------------------

    columnas_esperadas_21 = [
        "Age",
        "Daily_Screen_Time_Hours",
        "Late_Night_Usage",
        "Social_Comparison_Trigger",
        "Sleep_Duration_Hours",

        "Gender_MALE",

        "User_Archetype_DIGITAL MINIMALIST",
        "User_Archetype_HYPER-CONNECTED",
        "User_Archetype_PASSIVE SCROLLER",

        "Primary_Platform_INSTAGRAM",
        "Primary_Platform_LINKEDIN",
        "Primary_Platform_SNAPCHAT",
        "Primary_Platform_TIKTOK",
        "Primary_Platform_TWITTER/X",
        "Primary_Platform_YOUTUBE",

        "Dominant_Content_Type_ENTERTAINMENT/COMEDY",
        "Dominant_Content_Type_GAMING",
        "Dominant_Content_Type_LIFESTYLE/FASHION",
        "Dominant_Content_Type_NEWS/POLITICS",
        "Dominant_Content_Type_SELF-HELP/MOTIVATION",

        "Activity_Type_PASSIVE"
    ]

    # ------------------------------------------------------
    # 5. Asegurar exactamente las columnas esperadas
    # ------------------------------------------------------

    df_encoded = df_encoded.reindex(
        columns=columnas_esperadas_21,
        fill_value=0
    )

    return df_encoded


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"mensaje": "API de Machine Learning y NLP activa. Revisa /docs"}


@app.get("/app", response_class=FileResponse)
def read_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="No se encontró el archivo index.html")
    return FileResponse("index.html")


# 1. MWB Clasificador
@app.post("/predict-class", tags=["ml_predictions"], summary="Predecir Nivel de Adicción (Clasificador)")
def predict_class(data: InputData):
    if modelo_clasificador is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_CLASSIFIER}'.")
    try:
        df_input = preprocesar_y_escalar_datos_18(data)
        prediccion_num = int(modelo_clasificador.predict(df_input)[0])
        
        probabilidades = None
        if hasattr(modelo_clasificador, "predict_proba"):
            probs = modelo_clasificador.predict_proba(df_input).tolist()[0]
            if hasattr(modelo_clasificador, "classes_"):
                probabilidades = {
                    CLASES_MAPA.get(int(clase), str(clase)): round(prob, 4) 
                    for clase, prob in zip(modelo_clasificador.classes_, probs)
                }
            else:
                probabilidades = {
                    CLASES_MAPA.get(i, f"Clase_{i}"): round(prob, 4) 
                    for i, prob in enumerate(probs)
                }

        return {
            "status": "success",
            "prediction_code": prediccion_num,
            "addiction_level": CLASES_MAPA.get(prediccion_num, "Desconocido"),
            "probabilities": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# 2. MWB Regresor
@app.post("/predict-score", tags=["ml_predictions"], summary="Predecir Puntuación Continua (SVR)")
def predict_score(data: InputData):
    if modelo_regresor is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_REGRESSOR}'.")
    try:
        df_input = preprocesar_y_escalar_datos_18(data)
        prediccion_valor = float(modelo_regresor.predict(df_input)[0])
        return {"status": "success", "predicted_score": round(prediccion_valor, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción SVR: {str(e)}")


# 3. GAD-7 Clasificador
@app.post("/predict-gad7", tags=["ml_predictions"], summary="Predecir Nivel GAD-7 (XGBoost Classifier)")
def predict_gad7(data: GAD7InputData):
    if modelo_gad7 is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_GAD7_XGB}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_num = int(modelo_gad7.predict(df_input)[0])
        
        probabilidades = None
        if hasattr(modelo_gad7, "predict_proba"):
            probs = modelo_gad7.predict_proba(df_input).tolist()[0]
            probabilidades = {
                CLASES_MAPA.get(i, f"Clase_{i}"): round(prob, 4) 
                for i, prob in enumerate(probs)
            }

        return {
            "status": "success",
            "prediction_code": prediccion_num,
            "anxiety_level": CLASES_MAPA.get(prediccion_num, "Desconocido"),
            "probabilities": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción GAD-7 XGBoost: {str(e)}")


# 4. GAD-7 Regresor
@app.post("/predict-gad7-score", tags=["ml_predictions"], summary="Predecir Puntuación GAD-7 (XGBoost Regressor)")
def predict_gad7_score(data: GAD7InputData):
    if modelo_gad7_reg is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_GAD7_REG}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_valor = float(modelo_gad7_reg.predict(df_input)[0])
        return {
            "status": "success",
            "predicted_anxiety_score": round(prediccion_valor, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción GAD-7 Regressor: {str(e)}")


# 5. PHQ-9 Clasificador
@app.post("/predict-phq9", tags=["ml_predictions"], summary="Predecir Diagnóstico PHQ-9 (XGBoost Classifier)")
def predict_phq9(data: GAD7InputData):
    if modelo_phq9 is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_PHQ9_XGB}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_code = int(modelo_phq9.predict(df_input)[0])
        
        probabilidades = None
        if hasattr(modelo_phq9, "predict_proba"):
            probs = modelo_phq9.predict_proba(df_input).tolist()[0]
            probabilidades = {
                PHQ9_CLASES_MAPA.get(i, f"Clase_{i}"): round(prob, 4) 
                for i, prob in enumerate(probs)
            }

        return {
            "status": "success",
            "prediction_code": prediccion_code,
            "depression_level": PHQ9_CLASES_MAPA.get(prediccion_code, "Desconocido"),
            "probabilities": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción PHQ-9 XGBoost: {str(e)}")


# 6. PHQ-9 Regresor
@app.post("/predict-phq9-score", tags=["ml_predictions"], summary="Predecir Puntuación PHQ-9 (XGBoost Regressor)")
def predict_phq9_score(data: GAD7InputData):
    if modelo_phq9_reg is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_PHQ9_REG}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_valor = float(modelo_phq9_reg.predict(df_input)[0])
        return {
            "status": "success",
            "predicted_depression_score": round(prediccion_valor, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción PHQ-9 Regressor: {str(e)}")


# 7. Endpoint KMeans (Segmentación no supervisada)
@app.post("/cluster-user", tags=["ml_clustering"], summary="Segmentar Usuario en Cluster (KMeans)")
def cluster_user(data: InputData):
    if modelo_kmeans is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_KMEANS}'.")
    try:
        df_input = preprocesar_y_escalar_datos_18(data)
        cluster_id = int(modelo_kmeans.predict(df_input)[0])
        distancias = modelo_kmeans.transform(df_input).tolist()[0]
        
        info_cluster = DESCRIPCIONES_CLUSTERS.get(cluster_id, {
            "nombre": "Cluster Desconocido",
            "perfil": "No definido",
            "patron": "Sin datos de patrón disponibles.",
            "impacto": "Sin datos de impacto disponibles."
        })
        
        return {
            "status": "success",
            "assigned_cluster": cluster_id,
            "cluster_info": info_cluster,
            "centroid_distances": [round(d, 4) for d in distancias]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la segmentación KMeans: {str(e)}")


# 8. Endpoint NLP (Clasificación de Texto: Ansiedad, Depresión o Normal)
@app.post("/predict-nlp", tags=["nlp_predictions"], summary="Clasificar Texto (Ansiedad, Depresión o Normal)")
def predict_nlp(data: NLPInputData):
    if modelo_nlp is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_NLP}'.")
    try:
        # Se asume que modelo_nlp es un Pipeline de Scikit-Learn que acepta un iterable de textos
        prediccion = modelo_nlp.predict([data.text])[0]
        
        probabilidades = None
        if hasattr(modelo_nlp, "predict_proba"):
            probs = modelo_nlp.predict_proba([data.text]).tolist()[0]
            if hasattr(modelo_nlp, "classes_"):
                probabilidades = {
                    str(clase): round(prob, 4) 
                    for clase, prob in zip(modelo_nlp.classes_, probs)
                }

        return {
            "status": "success",
            "text": data.text,
            "prediction": str(prediccion),
            "probabilities": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción NLP: {str(e)}")
    from pydantic import BaseModel, Field



# 2. Endpoint RAG
@app.post("/ask-rag", tags=["rag_system"], summary="Consultar al sistema RAG")
def ask_rag(data: RAGInputData):
    if cadena_rag is None:
        raise HTTPException(status_code=500, detail="El sistema RAG no se encuentra inicializado.")
    try:
        # Ejecutamos la cadena RAG con la pregunta recibida
        respuesta = cadena_rag.invoke({"query": data.question})
        
        # Extraemos el texto de la respuesta (cadena_rag devuelve un dict con 'result')
        texto_respuesta = respuesta.get("result", "No se obtuvo respuesta del modelo.")

        return {
            "status": "success",
            "question": data.question,
            "answer": texto_respuesta
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la consulta RAG: {str(e)}")