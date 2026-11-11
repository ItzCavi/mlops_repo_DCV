"""
ft_engineering.py

Primera componente del flujo de creación del modelo. A partir de la tabla materializada
(Base_de_datos.csv) genera las características y devuelve los conjuntos de entrenamiento y
evaluación, junto con el pipeline de preprocesamiento que se ajustará con el entrenamiento.

Separación de responsabilidades:

1. generar_features(df): limpieza y atributos derivados fila a fila. No aprende nada de los
   datos, por lo que puede aplicarse antes de la partición y también en producción sobre datos
   nuevos sin riesgo de fuga de información.
2. construir_preprocesador(): pipeline de scikit-learn (imputación, recorte de extremos,
   logaritmo, escalado y codificación). Aprende parámetros (medianas, percentiles, categorías),
   así que solo debe ajustarse con el conjunto de entrenamiento. Ese ajuste ocurre dentro del
   pipeline del modelo en model_training_evaluation.py.

Las decisiones de limpieza y transformación vienen de comprension_eda.ipynb (secciones 3, 8 y 9).

Variable objetivo: el modelo estima la probabilidad de IMPAGO (impago = 1 - Pago_atiempo), porque
la clase de interés para el negocio es la minoritaria (4,7 % de los créditos) y las métricas de
scikit-learn (precisión, recall, AUC-PR) se calculan sobre la clase positiva.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# Rutas relativas a la ubicación del script, para que funcione desde cualquier directorio de trabajo.
# RAIZ: carpeta mlops_pipeline/ (código, configuración y artefactos del modelo).
# RAIZ_REPO: raíz del repositorio, donde está la tabla materializada Base_de_datos.csv.
RAIZ = Path(__file__).resolve().parent
RAIZ_REPO = RAIZ.parent
RUTA_CONFIG = RAIZ / "config.json"


def cargar_configuracion(ruta: Path = RUTA_CONFIG) -> dict:
    """Lee los parámetros del flujo (rutas, partición, umbrales de limpieza, etc.)."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


CONFIG = cargar_configuracion()

# ---------------------------------------------------------------------------------------------
# Grupos de variables tras generar_features. Cada grupo recibe un tratamiento distinto en el
# preprocesador; si se añade una variable nueva, debe incluirse en uno de estos grupos.
# ---------------------------------------------------------------------------------------------

# Montos y ratios: distribución log-normal con colas extremas (EDA 4.4) -> recorte + log1p.
VARIABLES_LOG = [
    "capital_prestado", "salario_cliente", "total_otros_prestamos", "cuota_pactada",
    "saldo_mora", "saldo_total", "saldo_principal", "promedio_ingresos_datacredito",
    "ratio_cuota_salario", "ratio_capital_salario", "ratio_otros_prestamos_salario",
    "ratio_salario_vs_central",
]
# Numéricas sin asimetría extrema (edad, puntaje, conteos) -> recorte + escalado.
VARIABLES_NUMERICAS = [
    "plazo_meses", "edad_cliente", "puntaje_datacredito", "cant_creditosvigentes",
    "huella_consulta", "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal",
]
# Indicadores 0/1 creados en generar_features; no tienen nulos y no necesitan escalado.
VARIABLES_BINARIAS = [
    "tiene_mora", "sin_info_ingresos_central", "sin_puntaje_central",
    "es_fin_de_mes", "es_fin_de_semana",
]
# Categóricas sin orden -> one-hot, agrupando categorías raras (códigos 6, 7 y 68 de tipo_credito).
VARIABLES_NOMINALES = ["tipo_credito", "tipo_laboral"]
# Categórica con orden natural: Decreciente < Estable < Creciente.
VARIABLES_ORDINALES = ["tendencia_ingresos"]

VARIABLES_MODELO = (VARIABLES_LOG + VARIABLES_NUMERICAS + VARIABLES_BINARIAS
                    + VARIABLES_NOMINALES + VARIABLES_ORDINALES)

# Columnas crudas que se esperan en la entrada (las de Base_de_datos.csv sin la variable
# objetivo). Las usan model_deploy.py y model_monitoring.py para validar los datos recibidos.
COLUMNAS_ENTRADA = [
    "tipo_credito", "fecha_prestamo", "capital_prestado", "plazo_meses", "edad_cliente",
    "tipo_laboral", "salario_cliente", "total_otros_prestamos", "cuota_pactada", "puntaje",
    "puntaje_datacredito", "cant_creditosvigentes", "huella_consulta", "saldo_mora",
    "saldo_total", "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
    "tendencia_ingresos",
]


# ---------------------------------------------------------------------------------------------
# 1. Limpieza y atributos derivados (sin parámetros aprendidos)
# ---------------------------------------------------------------------------------------------

def limpiar_datos(df: pd.DataFrame, config: dict = CONFIG) -> pd.DataFrame:
    """Corrige los problemas de calidad detectados en el EDA (sección 3.5).

    Los valores imposibles o centinela pasan a nulo para que el imputador los trate de forma
    uniforme, en lugar de que el modelo aprenda de ellos como si fueran reales.
    """
    lim = config["limpieza"]
    df = df.copy()

    # tendencia_ingresos traía montos numéricos mezclados con las categorías (58 registros).
    categorias = lim["categorias_tendencia"]
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(df["tendencia_ingresos"].isin(categorias))

    # Edades de 121 a 123 años: valor por defecto del sistema de origen.
    df["edad_cliente"] = df["edad_cliente"].where(df["edad_cliente"] <= lim["edad_maxima_valida"])

    # Puntajes de la central de 0, 7 y -7: equivalen a "sin puntaje".
    minimo = lim["puntaje_datacredito_minimo_valido"]
    df["puntaje_datacredito"] = df["puntaje_datacredito"].where(df["puntaje_datacredito"] >= minimo)

    # Fecha en su tipo correcto, para poder derivar atributos de calendario.
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")

    # tipo_credito es un código nominal: como texto, el codificador no le asigna un orden falso.
    df["tipo_credito"] = df["tipo_credito"].astype("Int64").astype(str)
    return df


def crear_atributos_derivados(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos propuestos en el EDA (sección 9). Solo usan valores de la misma fila."""
    df = df.copy()
    # Un salario de 0 no permite calcular ratios; se deja nulo para imputarlo después.
    salario = df["salario_cliente"].where(df["salario_cliente"] > 0)
    ingreso_central = df["promedio_ingresos_datacredito"].where(df["promedio_ingresos_datacredito"] > 0)

    # Capacidad de pago y endeudamiento relativo al ingreso declarado.
    df["ratio_cuota_salario"] = df["cuota_pactada"] / salario
    df["ratio_capital_salario"] = df["capital_prestado"] / salario
    df["ratio_otros_prestamos_salario"] = df["total_otros_prestamos"] / salario
    # Ingreso declarado frente al estimado por la central: la sobredeclaración se asoció a más impago.
    df["ratio_salario_vs_central"] = salario / ingreso_central

    # Indicadores: la ausencia de información fue informativa en el EDA (más impago con nulos).
    df["tiene_mora"] = (df["saldo_mora"] > 0).astype(int)
    df["sin_info_ingresos_central"] = df["promedio_ingresos_datacredito"].isna().astype(int)
    df["sin_puntaje_central"] = df["puntaje_datacredito"].isna().astype(int)

    # Calendario: solo atributos cíclicos cortos. El mes no se usa porque el histórico abarca
    # 17 meses y el modelo aprendería la evolución temporal, no una estacionalidad real.
    fecha = df["fecha_prestamo"]
    df["es_fin_de_mes"] = (fecha.dt.day >= 25).astype(int)
    df["es_fin_de_semana"] = (fecha.dt.dayofweek >= 5).astype(int)
    return df


def generar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + atributos derivados + selección de columnas del modelo.

    Es la función que se reutiliza en producción (model_deploy.py) sobre registros nuevos, por lo que
    no debe depender de la variable objetivo.
    """
    df = crear_atributos_derivados(limpiar_datos(df))
    # Quedan fuera: saldo_mora_codeudor (casi constante), puntaje (sospecha de fuga de
    # información) y fecha_prestamo (ya se extrajo lo útil).
    return df[VARIABLES_MODELO]


# ---------------------------------------------------------------------------------------------
# 2. Preprocesamiento (con parámetros aprendidos en entrenamiento)
# ---------------------------------------------------------------------------------------------

class RecortePercentiles(BaseEstimator, TransformerMixin):
    """Recorta (winsoriza) cada columna entre dos percentiles aprendidos en entrenamiento.

    Limita el efecto de valores extremos como salarios de 22.000 millones sin eliminar registros.
    """

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.limite_inferior_ = np.nanquantile(X, self.inferior, axis=0)
        self.limite_superior_ = np.nanquantile(X, self.superior, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.limite_inferior_, self.limite_superior_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def construir_preprocesador(config: dict = CONFIG) -> ColumnTransformer:
    """Devuelve el ColumnTransformer sin ajustar. Se ajusta dentro del pipeline de cada modelo."""
    pre = config["preprocesamiento"]
    recorte = dict(inferior=pre["percentil_inferior_recorte"], superior=pre["percentil_superior_recorte"])

    pipeline_log = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("recortar", RecortePercentiles(**recorte)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("escalar", StandardScaler()),
    ])
    pipeline_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("recortar", RecortePercentiles(**recorte)),
        ("escalar", StandardScaler()),
    ])
    pipeline_nominal = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Sin dato")),
        # Las categorías con menos del 1 % se agrupan en una sola columna "infrequent", y una
        # categoría nueva en producción se asigna a ese mismo grupo en lugar de fallar.
        ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=pre["frecuencia_minima_categoria"],
                                  sparse_output=False)),
    ])
    pipeline_ordinal = Pipeline([
        # Nulos y valores desconocidos se codifican como -1 ("sin dato"), separados del orden 0-2.
        ("ordinal", OrdinalEncoder(categories=[config["limpieza"]["categorias_tendencia"]],
                                   handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1)),
    ])

    return ColumnTransformer([
        ("log", pipeline_log, VARIABLES_LOG),
        ("numericas", pipeline_numerico, VARIABLES_NUMERICAS),
        ("binarias", "passthrough", VARIABLES_BINARIAS),
        ("nominales", pipeline_nominal, VARIABLES_NOMINALES),
        ("ordinales", pipeline_ordinal, VARIABLES_ORDINALES),
    ], verbose_feature_names_out=False)


# ---------------------------------------------------------------------------------------------
# 3. Punto de entrada: dataset de entrenamiento y evaluación
# ---------------------------------------------------------------------------------------------

def cargar_datos(config: dict = CONFIG) -> pd.DataFrame:
    """Lee la tabla materializada por Cargar_datos.ipynb (en la raíz del repositorio)."""
    ruta = RAIZ_REPO / config["datos"]["archivo"]
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecute primero Cargar_datos.ipynb.")
    return pd.read_csv(ruta)


def construir_dataset(config: dict = CONFIG):
    """Genera las features y devuelve los conjuntos de entrenamiento y evaluación.

    Returns
    -------
    X_train, X_test : DataFrame con las features (sin preprocesar todavía).
    y_train, y_test : Series con la variable objetivo impago (1 = no pagó a tiempo).
    preprocesador   : ColumnTransformer sin ajustar.
    """
    datos = cargar_datos(config)
    objetivo = config["datos"]["objetivo_original"]
    datos = datos.dropna(subset=[objetivo])

    X = generar_features(datos)
    y = (1 - datos[objetivo]).astype(int).rename(config["datos"]["objetivo_modelo"])

    # Partición estratificada: con 4,7 % de impagos, una partición aleatoria simple podría dejar
    # proporciones distintas en entrenamiento y evaluación.
    part = config["particion"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=part["test_size"], stratify=y, random_state=part["random_state"])
    return X_train, X_test, y_train, y_test, construir_preprocesador(config)


def resumen_dataset(X_train, X_test, y_train, y_test, preprocesador) -> None:
    """Imprime un resumen de los conjuntos generados y de las features finales."""
    print("Conjuntos generados")
    print(f"  Entrenamiento: {X_train.shape[0]:>6} filas | impago {y_train.mean():.2%}")
    print(f"  Evaluación:    {X_test.shape[0]:>6} filas | impago {y_test.mean():.2%}")
    print(f"  Features de entrada: {X_train.shape[1]}")
    nulos = X_train.isna().sum()
    print("  Nulos en entrenamiento (se imputan en el pipeline):")
    print(nulos[nulos > 0].to_string())

    # El preprocesador se ajusta aquí solo para mostrar las columnas resultantes; el ajuste
    # definitivo ocurre dentro de cada modelo con validación cruzada.
    matriz = preprocesador.fit(X_train).transform(X_train)
    nombres = preprocesador.get_feature_names_out()
    print(f"  Columnas tras el preprocesamiento: {matriz.shape[1]}")
    print("  " + ", ".join(nombres))
    print(f"  Nulos tras el preprocesamiento: {int(np.isnan(matriz).sum())}")


if __name__ == "__main__":
    resumen_dataset(*construir_dataset())
