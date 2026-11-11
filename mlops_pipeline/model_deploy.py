"""
model_deploy.py

Servicio de predicción con FastAPI. Expone el modelo guardado por model_training_evaluation.py
(mejor_modelo.joblib) para predecir la probabilidad de impago de uno o varios créditos.

Endpoints
---------
GET  /health        Estado del servicio y del modelo cargado.
POST /predict       Predicción por lotes con entrada JSON: {"registros": [{...}, {...}]}.
POST /predict/csv   Predicción por lotes subiendo un archivo CSV con las columnas de
                    Base_de_datos.csv (la columna Pago_atiempo es opcional y se ignora).

Los registros se envían crudos, con las mismas columnas de la fuente. El modelo guardado incluye
generar_features() de ft_engineering.py, así que aplica la misma limpieza y los mismos atributos
derivados que en el entrenamiento.

Ejecución local (desde la raíz del repositorio):
    uvicorn model_deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
Documentación:     http://localhost:8000/docs
"""

import io
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict, Field

# Importar ft_engineering también registra las funciones y clases que usa el modelo serializado
# (generar_features, RecortePercentiles); sin ese módulo, joblib no puede reconstruir el pipeline.
from ft_engineering import COLUMNAS_ENTRADA, CONFIG, RAIZ

MAX_REGISTROS = CONFIG["despliegue"]["max_registros_por_lote"]
RUTA_MODELO = RAIZ / CONFIG["salidas"]["modelo"]



def cargar_artefacto(ruta=RUTA_MODELO) -> dict:
    """Carga el modelo y sus metadatos (umbral, métricas, versión)."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}. Ejecute primero: python model_training_evaluation.py")
    return joblib.load(ruta)


ARTEFACTO = cargar_artefacto()


# ---------------------------------------------------------------------------------------------
# Esquemas de entrada y salida
# ---------------------------------------------------------------------------------------------

class RegistroCredito(BaseModel):
    """Un crédito con las columnas de la fuente. Los campos opcionales admiten nulos porque
    vienen vacíos en la fuente (información de la central de riesgo)."""

    tipo_credito: int
    fecha_prestamo: str = Field(examples=["2025-01-07 14:40:36"])
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str = Field(examples=["Empleado"])
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    # El modelo no usa puntaje (sospecha de fuga de información), pero se acepta para que el
    # esquema coincida con la fuente.
    puntaje: Optional[float] = None
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = Field(default=None, examples=["Creciente"])

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "tipo_credito": 4, "fecha_prestamo": "2025-01-07 14:40:36", "capital_prestado": 1852560,
        "plazo_meses": 12, "edad_cliente": 32, "tipo_laboral": "Empleado",
        "salario_cliente": 3500000, "total_otros_prestamos": 1000000, "cuota_pactada": 128650,
        "puntaje": 95.23, "puntaje_datacredito": 795, "cant_creditosvigentes": 0,
        "huella_consulta": 2, "saldo_mora": 0, "saldo_total": 0, "saldo_principal": None,
        "saldo_mora_codeudor": None, "creditos_sectorFinanciero": 0,
        "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0,
        "promedio_ingresos_datacredito": 916148, "tendencia_ingresos": "Creciente",
    }]})


class PeticionPrediccion(BaseModel):
    registros: List[RegistroCredito] = Field(min_length=1, max_length=MAX_REGISTROS)


class Prediccion(BaseModel):
    indice: int
    probabilidad_impago: float
    prediccion_impago: int
    resultado: str


class RespuestaPrediccion(BaseModel):
    modelo: str
    umbral: float
    n_registros: int
    n_impagos_predichos: int
    predicciones: List[Prediccion]


# ---------------------------------------------------------------------------------------------
# Lógica de predicción (separada de los endpoints para poder probarla sin servidor)
# ---------------------------------------------------------------------------------------------

def predecir(df: pd.DataFrame, artefacto: dict = ARTEFACTO) -> dict:
    """Predice un lote de registros crudos y devuelve la respuesta estructurada."""
    faltantes = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    probabilidades = artefacto["modelo"].predict_proba(df[COLUMNAS_ENTRADA])[:, 1]
    umbral = artefacto["umbral"]
    predicciones = [
        {
            "indice": i,
            "probabilidad_impago": round(float(p), 4),
            "prediccion_impago": int(p >= umbral),
            "resultado": "Riesgo de impago" if p >= umbral else "Pago a tiempo esperado",
        }
        for i, p in enumerate(probabilidades)
    ]
    return {
        "modelo": artefacto["nombre_modelo"],
        "umbral": round(umbral, 4),
        "n_registros": len(predicciones),
        "n_impagos_predichos": sum(p["prediccion_impago"] for p in predicciones),
        "predicciones": predicciones,
    }


# ---------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------

app = FastAPI(
    title="API de riesgo de impago",
    description="Predice la probabilidad de que un crédito no se pague a tiempo.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "estado": "ok",
        "modelo": ARTEFACTO["nombre_modelo"],
        "fecha_entrenamiento": ARTEFACTO["fecha_entrenamiento"],
        "version_sklearn": ARTEFACTO["version_sklearn"],
    }


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(peticion: PeticionPrediccion) -> dict:
    """Predicción por lotes con entrada JSON. FastAPI valida tipos y campos obligatorios y
    responde 422 con el detalle si un registro no cumple el esquema."""
    df = pd.DataFrame([r.model_dump() for r in peticion.registros])
    return predecir(df)


@app.post("/predict/csv", response_model=RespuestaPrediccion)
async def predict_csv(archivo: UploadFile = File(...)) -> dict:
    """Predicción por lotes subiendo un CSV con las columnas de Base_de_datos.csv."""
    if not archivo.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .csv")
    try:
        df = pd.read_csv(io.BytesIO(await archivo.read()))
    except Exception as error:  # el detalle del error de lectura ayuda a corregir el archivo
        raise HTTPException(status_code=400, detail=f"No se pudo leer el CSV: {error}") from error

    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV no tiene registros")
    if len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=413, detail=f"Máximo {MAX_REGISTROS} registros por lote")
    try:
        return predecir(df)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
