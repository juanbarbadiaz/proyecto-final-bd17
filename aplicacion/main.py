from typing import Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

tags_metadata = [
    {"name": "social_media_dataset", "description": "Consulta de datos sobre redes sociales y bienestar."},
    {"name": "ml_predictions", "description": "Endpoints de predicción con Machine Learning (Scikit-Learn y XGBoost)."},
    {"name": "ml_clustering", "description": "Endpoints de aprendizaje no supervisado (Segmentación/Clustering)."},
]

app = FastAPI(
    title="API Unificada de Diagnóstico, Salud Mental y Segmentación",
    description="Backend FastAPI con escalado de características integrado para clasificadores, regresores y clustering.",
    version="3.5.0",
    openapi_tags=tags_metadata
)

# Configuración global de CORS para consumo desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de los 7 modelos PKL
PATH_CLASSIFIER = "model/MWB_Classifier.pkl"
PATH_REGRESSOR = "model/MWB_Regressor.pkl"
PATH_GAD7_XGB = "model/MA_GAD7_Classifier.pkl"
PATH_GAD7_REG = "model/MA_GAD7_Regressor.pkl"
PATH_PHQ9_XGB = "model/MA_PHQ9_Classifier.pkl"
PATH_PHQ9_REG = "model/MA_PHQ9_Regressor.pkl"
PATH_KMEANS = "model/MWB_KMeans.pkl"

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

CLASES_MAPA = {0: "High", 1: "Low", 2: "Moderate", 3: "Severe"}

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
    Scroll_Without_Purpose: bool = Field(..., description="True para Sí, False para No", example=True)
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


# ==========================================
# FUNCIONES DE PREPROCESAMIENTO Y ESCALADO
# ==========================================

def preprocesar_y_escalar_datos_18(datos: InputData) -> pd.DataFrame:
    """Preprocesa y escala las variables continuas para modelos basados en 18 entradas (31 columnas final)."""
    df = pd.DataFrame([datos.model_dump()])
    
    columnas_numericas = [
        "Age", "Daily_Usage_Hours", "Platforms_Used_Count", "Posts_Per_Week",
        "Notifications_Per_Day", "Sleep_Hours", "Offline_Relationship_Quality",
        "Physical_Activity_Hrs_Week", "Screen_Free_Time_Hrs"
    ]
    
    scaler = StandardScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    df_encoded = pd.get_dummies(df, columns=[
        "Gender", "Occupation", "Relationship_Status", 
        "Primary_Platform", "First_Check_Morning"
    ])

    columnas_esperadas = [
        "Age", "Daily_Usage_Hours", "Platforms_Used_Count", "Posts_Per_Week",
        "Late_Night_Usage", "Notifications_Per_Day", "Scroll_Without_Purpose",
        "Tried_To_Cut_Back", "Failed_To_Cut_Back", "Sleep_Hours",
        "Offline_Relationship_Quality", "Physical_Activity_Hrs_Week", "Screen_Free_Time_Hrs",
        "Gender_MALE", "Gender_NON-BINARY", "Gender_UNKNOWN",
        "Occupation_STUDENT", "Occupation_UNEMPLOYED", "Occupation_UNKNOWN", "Occupation_WORKING PROFESSIONAL",
        "Relationship_Status_MARRIED", "Relationship_Status_SINGLE", "Relationship_Status_UNKNOWN",
        "Primary_Platform_INSTAGRAM", "Primary_Platform_SNAPCHAT", "Primary_Platform_TIKTOK",
        "Primary_Platform_TWITTER/X", "Primary_Platform_UNKNOWN", "Primary_Platform_YOUTUBE",
        "First_Check_Morning_AFTER 30 MIN", "First_Check_Morning_WITHIN 5 MIN"
    ]

    return df_encoded.reindex(columns=columnas_esperadas, fill_value=0)


def preprocesar_y_escalar_datos_21(datos: GAD7InputData) -> pd.DataFrame:
    """Preprocesa y escala las variables continuas para modelos basados en 10 entradas (21 columnas final)."""
    df = pd.DataFrame([datos.model_dump()])

    columnas_numericas = ["Age", "Daily_Screen_Time_Hours", "Sleep_Duration_Hours"]
    
    scaler = StandardScaler()
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    df_encoded = pd.get_dummies(df, columns=[
        "Gender", "User_Archetype", "Primary_Platform", 
        "Dominant_Content_Type", "Activity_Type"
    ])

    columnas_esperadas_21 = [
        "Age", "Daily_Screen_Time_Hours", "Late_Night_Usage", "Social_Comparison_Trigger", "Sleep_Duration_Hours",
        "Gender_MALE", "User_Archetype_DIGITAL MINIMALIST", "User_Archetype_HYPER-CONNECTED", "User_Archetype_PASSIVE SCROLLER",
        "Primary_Platform_INSTAGRAM", "Primary_Platform_LINKEDIN", "Primary_Platform_SNAPCHAT", "Primary_Platform_TIKTOK",
        "Primary_Platform_TWITTER/X", "Primary_Platform_YOUTUBE", "Dominant_Content_Type_ENTERTAINMENT/COMEDY",
        "Dominant_Content_Type_GAMING", "Dominant_Content_Type_LIFESTYLE/FASHION", "Dominant_Content_Type_NEWS/POLITICS",
        "Dominant_Content_Type_SELF-HELP/MOTIVATION", "Activity_Type_PASSIVE"
    ]
    return df_encoded.reindex(columns=columnas_esperadas_21, fill_value=0)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def home():
    return {"mensaje": "API de Machine Learning activa. Revisa /docs"}


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


# 2. MWB Regresor (SVR)
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


# 3. GAD-7 Clasificador (XGBoost)
@app.post("/predict-gad7", tags=["ml_predictions"], summary="Predecir Nivel GAD-7 (XGBoost Classifier)")
def predict_gad7(data: GAD7InputData):
    if modelo_gad7 is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_GAD7_XGB}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_num = int(modelo_gad7.predict(df_input)[0])
        probs_list = modelo_gad7.predict_proba(df_input).tolist()[0] if hasattr(modelo_gad7, "predict_proba") else None
        return {
            "status": "success",
            "prediction_code": prediccion_num,
            "addiction_level": CLASES_MAPA.get(prediccion_num, "Desconocido"),
            "probabilities": probs_list
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción GAD-7 XGBoost: {str(e)}")


# 4. GAD-7 Regresor (XGBoost)
@app.post("/predict-gad7-score", tags=["ml_predictions"], summary="Predecir Puntuación GAD-7 (XGBoost Regressor)")
def predict_gad7_score(data: GAD7InputData):
    if modelo_gad7_reg is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_GAD7_REG}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_valor = float(modelo_gad7_reg.predict(df_input)[0])
        return {
            "status": "success",
            "predicted_gad7_score": round(prediccion_valor, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción GAD-7 Regressor: {str(e)}")


# 5. PHQ-9 Clasificador (XGBoost Classifier) -> Entrada: GAD7InputData (10 variables / 21 OHE)
@app.post("/predict-phq9", tags=["ml_predictions"], summary="Predecir Diagnóstico PHQ-9 (XGBoost Classifier)")
def predict_phq9(data: GAD7InputData):
    if modelo_phq9 is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_PHQ9_XGB}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_clase = str(modelo_phq9.predict(df_input)[0])
        
        probabilidades = None
        if hasattr(modelo_phq9, "predict_proba"):
            probs = modelo_phq9.predict_proba(df_input).tolist()[0]
            if hasattr(modelo_phq9, "classes_"):
                probabilidades = {str(clase): round(prob, 4) for clase, prob in zip(modelo_phq9.classes_, probs)}
            else:
                probabilidades = [round(prob, 4) for prob in probs]

        return {
            "status": "success",
            "phq9_severity": prediccion_clase,
            "probabilities": probabilidades
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción PHQ-9 XGBoost: {str(e)}")


# 6. PHQ-9 Regresor (XGBoost Regressor)
@app.post("/predict-phq9-score", tags=["ml_predictions"], summary="Predecir Puntuación PHQ-9 (XGBoost Regressor)")
def predict_phq9_score(data: GAD7InputData):
    if modelo_phq9_reg is None:
        raise HTTPException(status_code=500, detail=f"No se encontró el archivo '{PATH_PHQ9_REG}'.")
    try:
        df_input = preprocesar_y_escalar_datos_21(data)
        prediccion_valor = float(modelo_phq9_reg.predict(df_input)[0])
        return {
            "status": "success",
            "predicted_phq9_score": round(prediccion_valor, 2)
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
        
        return {
            "status": "success",
            "assigned_cluster": cluster_id,
            "centroid_distances": [round(d, 4) for d in distancias]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la segmentación KMeans: {str(e)}")