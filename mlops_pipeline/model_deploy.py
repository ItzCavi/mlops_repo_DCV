"""
model_deploy.py

Servicio de predicción con FastAPI. Expone el modelo guardado por model_training_evaluation.py
(mejor_modelo.joblib) para predecir la probabilidad de impago de uno o varios créditos.

Endpoints
---------
GET  /health        Estado del servicio y del modelo cargado.
POST /predict       Predicción por lotes. Acepta, según el encabezado Content-Type:
                      - application/json:     {"registros": [{...}, {...}]}
                      - text/csv:             el contenido del CSV en el cuerpo
                      - multipart/form-data:  un archivo CSV en el campo "archivo"
                    El CSV lleva las columnas de Base_de_datos.csv (Pago_atiempo es opcional
                    y se ignora).
POST /predict/csv   Atajo para subir un archivo CSV desde /docs (mismo resultado).

Los registros se envían crudos, con las mismas columnas de la fuente. El modelo guardado incluye
generar_features() de ft_engineering.py, así que aplica la misma limpieza y los mismos atributos
derivados que en el entrenamiento.

Ejecución local (desde la raíz del repositorio):
    uvicorn model_deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
Documentación:     http://localhost:8000/docs
"""

import io
import logging
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Importar ft_engineering también registra las funciones y clases que usa el modelo serializado
# (generar_features, RecortePercentiles); sin ese módulo, joblib no puede reconstruir el pipeline.
from ft_engineering import COLUMNAS_ENTRADA, CONFIG, RAIZ

MAX_REGISTROS = CONFIG["despliegue"]["max_registros_por_lote"]
# Límite del tamaño del archivo recibido: evita que una petición enorme agote la memoria.
MAX_BYTES_CSV = 20 * 1024 * 1024
LOGGER = logging.getLogger("model_deploy")
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
    puntaje: float | None = None
    puntaje_datacredito: float | None = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float | None = None
    saldo_total: float | None = None
    saldo_principal: float | None = None
    saldo_mora_codeudor: float | None = None
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float | None = None
    tendencia_ingresos: str | None = Field(default=None, examples=["Creciente"])

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
    registros: list[RegistroCredito] = Field(min_length=1, max_length=MAX_REGISTROS)


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
    predicciones: list[Prediccion]


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


def leer_csv(contenido: bytes) -> pd.DataFrame:
    """Convierte el contenido de un CSV en DataFrame validando tamaño y número de registros."""
    if len(contenido) > MAX_BYTES_CSV:
        raise HTTPException(status_code=413, detail="El archivo supera el tamaño máximo de 20 MB")
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        LOGGER.warning("CSV no válido: %s", error)
        raise HTTPException(status_code=400, detail="No se pudo leer el CSV") from error
    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV no tiene registros")
    if len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=413, detail=f"Máximo {MAX_REGISTROS} registros por lote")
    return df


def predecir_o_422(df: pd.DataFrame) -> dict:
    try:
        return predecir(df)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


# Documentación de las tres formas de entrada de /predict en /docs (OpenAPI).
DOC_PREDICT = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": PeticionPrediccion.model_json_schema()},
            "text/csv": {"schema": {"type": "string"},
                         "example": ",".join(COLUMNAS_ENTRADA) + "\n4,2025-01-07 14:40:36,..."},
            "multipart/form-data": {"schema": {
                "type": "object", "required": ["archivo"],
                "properties": {"archivo": {"type": "string", "format": "binary"}}}},
        },
    }
}


@app.post("/predict", response_model=RespuestaPrediccion, openapi_extra=DOC_PREDICT)
async def predict(request: Request) -> dict:
    """Predicción por lotes. El formato de entrada se elige con el encabezado Content-Type:
    JSON ({"registros": [...]}), CSV en el cuerpo (text/csv) o archivo CSV (multipart)."""
    tipo = request.headers.get("content-type", "").split(";")[0].strip().lower()

    if tipo == "application/json":
        try:
            peticion = PeticionPrediccion.model_validate_json(await request.body())
        except ValidationError as error:
            # Mismo formato de error que FastAPI usa para los esquemas inválidos.
            raise HTTPException(status_code=422,
                                detail=error.errors(include_url=False, include_context=False)) from error
        df = pd.DataFrame([r.model_dump() for r in peticion.registros])
    elif tipo in ("text/csv", "application/csv"):
        df = leer_csv(await request.body())
    elif tipo == "multipart/form-data":
        formulario = await request.form()
        archivo = formulario.get("archivo")
        if archivo is None or not hasattr(archivo, "read"):
            raise HTTPException(status_code=400, detail='Envíe el CSV en el campo "archivo"')
        df = leer_csv(await archivo.read())
    else:
        raise HTTPException(status_code=415, detail=(
            "Content-Type no soportado. Use application/json, text/csv o multipart/form-data"))
    return predecir_o_422(df)


@app.post("/predict/csv", response_model=RespuestaPrediccion)
async def predict_csv(
        archivo: Annotated[UploadFile, File(description="CSV con las columnas de Base_de_datos.csv")]) -> dict:
    """Atajo para subir un archivo CSV desde la documentación interactiva (/docs)."""
    if not (archivo.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="El archivo debe tener extensión .csv")
    return predecir_o_422(leer_csv(await archivo.read()))


if __name__ == "__main__":
    import os

    import uvicorn

    # Por defecto solo escucha en la máquina local; en Docker se publica con --host 0.0.0.0.
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")))
