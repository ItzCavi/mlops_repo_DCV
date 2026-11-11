"""
model_monitoring.py

Monitoreo del modelo en producción: data drift, drift de las predicciones y análisis temporal.
Tiene dos modos de uso:

1. Trabajo de monitoreo (batch, periódico):
       python mlops_pipeline/model_monitoring.py --job
   Construye la tabla de datos con sus pronósticos (monitoreo_predicciones.csv), divide el
   periodo posterior a la fecha de corte en ventanas (mensuales por defecto), toma una muestra
   aleatoria de cada ventana y calcula las métricas de drift contra el periodo de referencia.
   Guarda el historial (monitoreo_historial_drift.csv), el resumen por periodo
   (monitoreo_resumen_periodos.csv) e imprime alertas y recomendaciones. Es el paso que se
   programaría con la periodicidad definida (por ejemplo, una vez al mes en Jenkins o cron).

2. Aplicación Streamlit (visualización):
       streamlit run mlops_pipeline/model_monitoring.py

Métricas de drift
-----------------
- PSI (Population Stability Index), numéricas y categóricas. Es el indicador estándar en riesgo
  de crédito y define el semáforo:
      PSI < 0,10         estable          (verde)
      0,10 <= PSI < 0,25 cambio moderado  (amarillo)
      PSI >= 0,25        cambio crítico   (rojo)
- Divergencia de Jensen-Shannon (base 2, entre 0 y 1): versión simétrica y acotada de la
  divergencia KL, calculada sobre los mismos intervalos que el PSI.
- Kolmogorov-Smirnov (numéricas) y chi-cuadrado (categóricas): pruebas estadísticas de apoyo.
  Con muestras grandes detectan diferencias muy pequeñas; por eso el semáforo usa el PSI y el
  p-valor se muestra como evidencia complementaria.

Todos los umbrales y la periodicidad se configuran en config.json (sección "monitoreo").
"""

import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")  # en el trabajo batch no hay pantalla; Streamlit recibe las figuras igual
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import stats
from scipy.spatial.distance import jensenshannon

from ft_engineering import (COLUMNAS_ENTRADA, CONFIG, RAIZ, VARIABLES_BINARIAS, VARIABLES_LOG,
                            VARIABLES_MODELO, VARIABLES_NOMINALES, VARIABLES_NUMERICAS,
                            VARIABLES_ORDINALES, cargar_datos, generar_features)

MON = CONFIG["monitoreo"]
VARIABLES_CONTINUAS = VARIABLES_LOG + VARIABLES_NUMERICAS
VARIABLES_DISCRETAS = VARIABLES_BINARIAS + VARIABLES_NOMINALES + VARIABLES_ORDINALES
VARIABLE_PUNTAJE = "probabilidad_impago"  # salida del modelo, se monitorea como una variable más
EPSILON = 1e-4  # evita log(0) en el PSI cuando un intervalo queda vacío
COLORES = {"Verde": "#2e7d32", "Amarillo": "#f9a825", "Rojo": "#c62828", "Sin datos": "#9e9e9e"}
ORDEN_ESTADO = {"Verde": 0, "Amarillo": 1, "Rojo": 2}


# ---------------------------------------------------------------------------------------------
# 1. Métricas de drift (funciones puras, sin Streamlit, para poder probarlas)
# ---------------------------------------------------------------------------------------------

def a_categoria(serie: pd.Series) -> pd.Series:
    """Convierte a texto con los nulos como categoría "Sin dato". Se hace en este orden porque
    astype(str) convierte NaN en el texto "nan" en algunas versiones de pandas."""
    return serie.astype(object).where(serie.notna(), "Sin dato").astype(str)


def proporciones_categoricas(referencia: pd.Series, actual: pd.Series) -> tuple:
    """Proporción de cada categoría en ambos periodos, sobre la unión de categorías."""
    ref, act = a_categoria(referencia), a_categoria(actual)
    categorias = sorted(set(ref) | set(act))
    p_ref = ref.value_counts(normalize=True).reindex(categorias, fill_value=0).values
    p_act = act.value_counts(normalize=True).reindex(categorias, fill_value=0).values
    return p_ref, p_act


def proporciones_numericas(referencia: pd.Series, actual: pd.Series, bins: int = MON["bins_psi"]):
    """Proporción por intervalo, con cortes en los cuantiles de la referencia para que cada
    intervalo tenga una proporción parecida de datos históricos. Devuelve None si la variable es
    casi constante (menos de 2 intervalos útiles): en ese caso se compara como categórica."""
    ref, act = referencia.dropna(), actual.dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 3:
        return None
    cortes[0], cortes[-1] = -np.inf, np.inf
    return np.histogram(ref, cortes)[0] / len(ref), np.histogram(act, cortes)[0] / len(act)


def proporciones(referencia: pd.Series, actual: pd.Series, continua: bool):
    """Distribuciones comparables de ambos periodos; None si alguno no tiene datos."""
    if continua:
        if referencia.dropna().empty or actual.dropna().empty:
            return None
        resultado = proporciones_numericas(referencia, actual)
        if resultado is not None:
            return resultado
    return proporciones_categoricas(referencia, actual)


def psi_desde_proporciones(p_ref: np.ndarray, p_act: np.ndarray) -> float:
    """PSI = suma((actual - referencia) * ln(actual / referencia))."""
    p_ref = np.clip(p_ref, EPSILON, None)
    p_act = np.clip(p_act, EPSILON, None)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def psi_numerico(referencia: pd.Series, actual: pd.Series) -> float:
    p = proporciones(referencia, actual, continua=True)
    return np.nan if p is None else psi_desde_proporciones(*p)


def psi_categorico(referencia: pd.Series, actual: pd.Series) -> float:
    """PSI sobre las frecuencias de cada categoría. Los nulos se tratan como una categoría."""
    return psi_desde_proporciones(*proporciones_categoricas(referencia, actual))


def jensen_shannon(referencia: pd.Series, actual: pd.Series, continua: bool) -> float:
    """Divergencia de Jensen-Shannon en base 2 (0 = distribuciones iguales, 1 = disjuntas).

    scipy devuelve la distancia (raíz cuadrada de la divergencia), por eso se eleva al cuadrado.
    """
    p = proporciones(referencia, actual, continua)
    if p is None:
        return np.nan
    return float(jensenshannon(p[0], p[1], base=2) ** 2)


def prueba_estadistica(referencia: pd.Series, actual: pd.Series, continua: bool) -> tuple:
    """Devuelve (nombre de la prueba, p-valor): KS para numéricas y chi-cuadrado para categóricas."""
    if continua:
        ref, act = referencia.dropna(), actual.dropna()
        if ref.empty or act.empty:
            return "KS", np.nan
        return "KS", float(stats.ks_2samp(ref, act).pvalue)
    tabla = pd.crosstab(
        a_categoria(pd.concat([referencia, actual])),
        np.r_[np.zeros(len(referencia)), np.ones(len(actual))])
    if tabla.shape[0] < 2:
        return "Chi2", 1.0
    return "Chi2", float(stats.chi2_contingency(tabla)[1])


def semaforo(psi: float) -> str:
    if pd.isna(psi):
        return "Sin datos"
    if psi >= MON["psi_critico"]:
        return "Rojo"
    if psi >= MON["psi_alerta"]:
        return "Amarillo"
    return "Verde"


def resumen_variable(serie: pd.Series, continua: bool) -> str:
    if continua:
        return f"{serie.median():,.2f}" if serie.notna().any() else "-"
    moda = serie.mode()
    return str(moda.iloc[0]) if not moda.empty else "-"


def fila_drift(referencia: pd.Series, actual: pd.Series, variable: str, continua: bool, tipo: str) -> dict:
    psi = psi_numerico(referencia, actual) if continua else psi_categorico(referencia, actual)
    prueba, p_valor = prueba_estadistica(referencia, actual, continua)
    return {
        "variable": variable,
        "tipo": tipo,
        "psi": psi,
        "js_divergencia": jensen_shannon(referencia, actual, continua),
        "prueba": prueba,
        "p_valor": p_valor,
        "diferencia_significativa": bool(p_valor < MON["alpha"]) if not pd.isna(p_valor) else False,
        "media_o_moda_ref": resumen_variable(referencia, continua),
        "media_o_moda_act": resumen_variable(actual, continua),
        "nulos_ref_%": round(referencia.isna().mean() * 100, 2),
        "nulos_act_%": round(actual.isna().mean() * 100, 2),
        "estado": semaforo(psi),
    }


def calcular_drift(referencia: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Tabla de drift por variable del modelo, ordenada de mayor a menor PSI.

    Si ambos DataFrames traen la columna probabilidad_impago, también se mide el drift de las
    predicciones (cambio en la distribución del puntaje que entrega el modelo).
    """
    filas = [fila_drift(referencia[v], actual[v], v, v in VARIABLES_CONTINUAS,
                        "Numérica" if v in VARIABLES_CONTINUAS else "Categórica")
             for v in VARIABLES_MODELO]
    if VARIABLE_PUNTAJE in referencia.columns and VARIABLE_PUNTAJE in actual.columns:
        filas.append(fila_drift(referencia[VARIABLE_PUNTAJE], actual[VARIABLE_PUNTAJE],
                                VARIABLE_PUNTAJE, True, "Predicción"))
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def estado_general(drift: pd.DataFrame) -> str:
    if (drift["estado"] == "Rojo").any():
        return "Rojo"
    if (drift["estado"] == "Amarillo").any():
        return "Amarillo"
    return "Verde"


# ---------------------------------------------------------------------------------------------
# 2. Alertas y recomendaciones
# ---------------------------------------------------------------------------------------------

def generar_alertas(drift: pd.DataFrame) -> list:
    """Mensajes de alerta para las variables en rojo o amarillo."""
    alertas = []
    for _, fila in drift[drift["estado"].isin(["Rojo", "Amarillo"])].iterrows():
        nivel = "CRÍTICA" if fila["estado"] == "Rojo" else "MODERADA"
        alertas.append({
            "nivel": fila["estado"],
            "mensaje": (f"Alerta {nivel}: {fila['variable']} con PSI = {fila['psi']:.3f}, "
                        f"JS = {fila['js_divergencia']:.3f} "
                        f"({fila['prueba']} p-valor = {fila['p_valor']:.4f})."),
        })
    return alertas


def listar(variables: list, maximo: int = 5) -> str:
    """Texto con las primeras variables de la lista para que los mensajes no se vuelvan ilegibles."""
    texto = ", ".join(variables[:maximo])
    return texto + (f" y {len(variables) - maximo} más" if len(variables) > maximo else "")


def generar_recomendaciones(drift: pd.DataFrame, tasa_ref: float = None, tasa_act: float = None,
                            tendencias: pd.DataFrame = None) -> list:
    """Recomendaciones automáticas a partir del drift, la tasa de impago y las tendencias.

    Devuelve una lista de {"nivel": "Rojo" | "Amarillo" | "Verde", "mensaje": str}.
    """
    recomendaciones = []
    variables = drift[drift["tipo"] != "Predicción"]
    rojas = variables.loc[variables["estado"] == "Rojo", "variable"].tolist()
    amarillas = variables.loc[variables["estado"] == "Amarillo", "variable"].tolist()

    if len(rojas) >= MON["variables_rojo_para_reentrenar"]:
        recomendaciones.append({"nivel": "Rojo", "mensaje": (
            f"Reentrenar el modelo: {len(rojas)} variables con drift crítico ({listar(rojas)}). "
            "La población actual ya no se parece a la de entrenamiento.")})
    elif rojas:
        recomendaciones.append({"nivel": "Rojo", "mensaje": (
            f"Revisar la fuente y la ingesta de {listar(rojas)} antes de reentrenar: un cambio "
            "aislado suele deberse a un problema de datos o a un cambio de política.")})
    if amarillas:
        recomendaciones.append({"nivel": "Amarillo", "mensaje": (
            f"Vigilar en el próximo periodo: {listar(amarillas)} (cambio moderado).")})

    puntaje = drift[drift["variable"] == VARIABLE_PUNTAJE]
    if not puntaje.empty and puntaje["estado"].iloc[0] in ("Rojo", "Amarillo"):
        recomendaciones.append({"nivel": puntaje["estado"].iloc[0], "mensaje": (
            "Cambió la distribución de las predicciones del modelo: revisar si el umbral de "
            "decisión sigue siendo adecuado y validar el desempeño con datos etiquetados recientes.")})

    cambio_nulos = (variables["nulos_act_%"] - variables["nulos_ref_%"]).abs()
    nulos = variables.loc[cambio_nulos >= MON["cambio_nulos_pp"], "variable"].tolist()
    if nulos:
        recomendaciones.append({"nivel": "Amarillo", "mensaje": (
            f"Calidad de datos: el porcentaje de nulos cambió más de {MON['cambio_nulos_pp']:.0f} "
            f"puntos en {listar(nulos)}. Revisar el proceso que materializa la tabla.")})

    if tasa_ref is not None and tasa_act is not None and not pd.isna(tasa_act):
        diferencia = tasa_act - tasa_ref
        if abs(diferencia) >= MON["cambio_tasa_impago_pp"]:
            recomendaciones.append({"nivel": "Rojo", "mensaje": (
                f"La tasa de impago observada cambió {diferencia:+.2f} puntos ({tasa_ref:.2f} % a "
                f"{tasa_act:.2f} %): drift de concepto posible, evaluar reentrenamiento.")})

    if tendencias is not None and not tendencias.empty:
        crecientes = tendencias.loc[tendencias["tendencia"] == "Creciente", "variable"].tolist()
        abruptas = tendencias.loc[tendencias["cambio_abrupto"] != "-", "variable"].tolist()
        if crecientes:
            recomendaciones.append({"nivel": "Amarillo", "mensaje": (
                f"Drift con tendencia creciente en el tiempo: {listar(crecientes)}. "
                "Programar la revisión del modelo antes de que superen el umbral crítico.")})
        if abruptas:
            recomendaciones.append({"nivel": "Amarillo", "mensaje": (
                f"Cambios abruptos entre periodos consecutivos en: {listar(abruptas)}. "
                "Verificar si coinciden con cambios de producto, política o sistema de origen.")})

    if not recomendaciones:
        recomendaciones.append({"nivel": "Verde", "mensaje": (
            "Sin acción requerida: mantener el monitoreo con la periodicidad definida.")})
    return recomendaciones


# ---------------------------------------------------------------------------------------------
# 3. Tabla de predicciones y análisis temporal (trabajo de monitoreo)
# ---------------------------------------------------------------------------------------------

def cargar_artefacto() -> dict:
    """Carga el modelo guardado por model_training_evaluation.py."""
    ruta = RAIZ / CONFIG["salidas"]["modelo"]
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}. Ejecute primero: python mlops_pipeline/model_training_evaluation.py")
    return joblib.load(ruta)


def construir_tabla_predicciones(crudo: pd.DataFrame, artefacto: dict) -> pd.DataFrame:
    """Tabla de monitoreo: datos crudos + pronóstico del modelo + periodo.

    En producción esta tabla se llenaría con las peticiones que recibe la API; aquí se simula
    puntuando la tabla materializada.
    """
    tabla = crudo.copy()
    probabilidad = artefacto["modelo"].predict_proba(tabla[COLUMNAS_ENTRADA])[:, 1]
    tabla[VARIABLE_PUNTAJE] = probabilidad
    tabla["prediccion_impago"] = (probabilidad >= artefacto["umbral"]).astype(int)
    fechas = pd.to_datetime(tabla["fecha_prestamo"], errors="coerce")
    tabla["periodo"] = fechas.dt.to_period(MON["frecuencia"]).astype(str)
    return tabla


def features_con_puntaje(tabla: pd.DataFrame) -> pd.DataFrame:
    """Variables del modelo (ya limpias, con derivados) más el puntaje que entregó el modelo."""
    features = generar_features(tabla)
    if VARIABLE_PUNTAJE in tabla.columns:
        features[VARIABLE_PUNTAJE] = tabla[VARIABLE_PUNTAJE].values
    return features


def tasa_impago(tabla: pd.DataFrame) -> float:
    """Tasa de impago observada en %, si la tabla trae la variable objetivo."""
    objetivo = CONFIG["datos"]["objetivo_original"]
    if objetivo not in tabla.columns or tabla[objetivo].isna().all():
        return np.nan
    return float((1 - tabla[objetivo].mean()) * 100)


def psi_esperado_por_azar(n_ref: int, n_act: int, bins: int = MON["bins_psi"]) -> float:
    """Valor esperado aproximado del PSI entre dos muestras de la MISMA población.

    Bajo la hipótesis de no cambio, el PSI se comporta como un chi-cuadrado con (bins - 1)
    grados de libertad escalado por (1/n_ref + 1/n_act). Con 10 intervalos y 150 registros
    actuales da cerca de 0,06: un PSI de ese orden no indica drift.
    """
    return (bins - 1) * (1 / n_ref + 1 / n_act)


def analisis_temporal(tabla_ref: pd.DataFrame, tabla_act: pd.DataFrame,
                      tamano_muestra: int = MON["tamano_muestra"], semilla: int = 42) -> tuple:
    """Drift de cada periodo del tramo actual contra la referencia.

    Para cada periodo se toma una muestra aleatoria de hasta `tamano_muestra` registros (el
    muestreo periódico mantiene el costo constante aunque crezca el volumen y hace comparables
    los periodos). Los periodos con menos de `min_registros_periodo` se omiten porque el PSI de
    muestras pequeñas tiene un sesgo positivo grande (ver psi_esperado_por_azar).

    Returns
    -------
    historial : DataFrame largo (periodo x variable) con PSI, JS, p-valor y estado.
    resumen   : DataFrame con una fila por periodo (volumen, tasas y estado general).
    """
    features_ref = features_con_puntaje(tabla_ref)
    tasa_ref = tasa_impago(tabla_ref)
    historiales, resumenes = [], []

    for periodo, grupo in tabla_act.groupby("periodo", sort=True):
        if len(grupo) < MON["min_registros_periodo"]:
            continue
        muestra = grupo.sample(min(tamano_muestra, len(grupo)), random_state=semilla)
        drift = calcular_drift(features_ref, features_con_puntaje(muestra))
        drift.insert(0, "periodo", periodo)
        historiales.append(drift)

        variables = drift[drift["tipo"] != "Predicción"]
        puntaje = drift.loc[drift["variable"] == VARIABLE_PUNTAJE, "psi"]
        resumenes.append({
            "periodo": periodo,
            "registros": len(grupo),
            "muestra": len(muestra),
            "tasa_impago_real_%": tasa_impago(grupo),
            "tasa_impago_referencia_%": tasa_ref,
            "probabilidad_media_%": grupo[VARIABLE_PUNTAJE].mean() * 100,
            "impagos_predichos_%": grupo["prediccion_impago"].mean() * 100,
            "psi_predicciones": float(puntaje.iloc[0]) if not puntaje.empty else np.nan,
            # PSI que se esperaría solo por azar al comparar muestras de este tamaño: sirve para
            # no confundir ruido de muestreo con drift en periodos con pocos registros.
            "psi_esperado_por_azar": psi_esperado_por_azar(len(tabla_ref), len(muestra)),
            "psi_maximo": variables["psi"].max(),
            "variable_psi_maximo": variables.iloc[0]["variable"],
            "variables_rojo": int((variables["estado"] == "Rojo").sum()),
            "variables_amarillo": int((variables["estado"] == "Amarillo").sum()),
            "estado_general": estado_general(drift),
        })

    historial = pd.concat(historiales, ignore_index=True) if historiales else pd.DataFrame()
    return historial, pd.DataFrame(resumenes)


def detectar_tendencias(historial: pd.DataFrame) -> pd.DataFrame:
    """Tendencia y cambios abruptos del PSI de cada variable a lo largo de los periodos.

    - Tendencia: pendiente de una recta ajustada al PSI por periodo. Se considera creciente o
      decreciente si en todo el horizonte el cambio acumulado supera la mitad del umbral de
      alerta; si no, estable.
    - Cambio abrupto: salto del PSI entre dos periodos consecutivos mayor que
      `umbral_cambio_abrupto`. Se informa el primer periodo donde ocurre.
    """
    if historial.empty:
        return pd.DataFrame()
    filas = []
    for variable, datos in historial.groupby("variable", sort=False):
        datos = datos.sort_values("periodo")
        psi = datos["psi"].to_numpy(dtype=float)
        if len(psi) < 2 or np.isnan(psi).all():
            continue
        x = np.arange(len(psi))
        validos = ~np.isnan(psi)
        pendiente = float(np.polyfit(x[validos], psi[validos], 1)[0]) if validos.sum() >= 2 else 0.0
        cambio_total = pendiente * (len(psi) - 1)
        if cambio_total > MON["psi_alerta"] / 2:
            tendencia = "Creciente"
        elif cambio_total < -MON["psi_alerta"] / 2:
            tendencia = "Decreciente"
        else:
            tendencia = "Estable"
        saltos = np.diff(psi)
        abruptos = np.where(saltos > MON["umbral_cambio_abrupto"])[0]
        filas.append({
            "variable": variable,
            "psi_primer_periodo": psi[0],
            "psi_ultimo_periodo": psi[-1],
            "psi_maximo": np.nanmax(psi),
            "pendiente_por_periodo": pendiente,
            "tendencia": tendencia,
            "cambio_abrupto": datos["periodo"].iloc[abruptos[0] + 1] if len(abruptos) else "-",
            "estado_ultimo_periodo": datos["estado"].iloc[-1],
        })
    return pd.DataFrame(filas).sort_values("psi_ultimo_periodo", ascending=False).reset_index(drop=True)


def separar_periodos(datos: pd.DataFrame, fecha_corte: str) -> tuple:
    """Divide los datos en periodo de referencia (antes del corte) y actual (desde el corte)."""
    fechas = pd.to_datetime(datos["fecha_prestamo"], errors="coerce")
    corte = pd.Timestamp(fecha_corte)
    return datos[fechas < corte], datos[fechas >= corte]


def ejecutar_job(guardar: bool = True) -> dict:
    """Trabajo de monitoreo periódico (modo batch). Ver la descripción del módulo."""
    artefacto = cargar_artefacto()
    tabla = construir_tabla_predicciones(cargar_datos(), artefacto)
    tabla_ref, tabla_act = separar_periodos(tabla, MON["fecha_corte_simulacion"])

    historial, resumen = analisis_temporal(tabla_ref, tabla_act)
    tendencias = detectar_tendencias(historial)
    drift_actual = calcular_drift(features_con_puntaje(tabla_ref), features_con_puntaje(tabla_act))
    alertas = generar_alertas(drift_actual)
    recomendaciones = generar_recomendaciones(drift_actual, tasa_impago(tabla_ref),
                                              tasa_impago(tabla_act), tendencias)

    if guardar:
        salidas = MON["salidas"]
        tabla.to_csv(RAIZ / salidas["tabla_predicciones"], index=False, encoding="utf-8-sig")
        historial.to_csv(RAIZ / salidas["historial_drift"], index=False, encoding="utf-8-sig")
        resumen.to_csv(RAIZ / salidas["resumen_periodos"], index=False, encoding="utf-8-sig")

    return {"tabla": tabla, "historial": historial, "resumen": resumen, "tendencias": tendencias,
            "drift_actual": drift_actual, "alertas": alertas, "recomendaciones": recomendaciones}


def imprimir_job(resultado: dict) -> None:
    resumen = resultado["resumen"]
    print(f"Trabajo de monitoreo - {datetime.now():%Y-%m-%d %H:%M}")
    print(f"Referencia: créditos anteriores a {MON['fecha_corte_simulacion']} | "
          f"periodicidad {MON['frecuencia']} | muestra por periodo {MON['tamano_muestra']}")
    if not resumen.empty:
        columnas = ["periodo", "registros", "muestra", "tasa_impago_real_%", "impagos_predichos_%",
                    "psi_predicciones", "psi_esperado_por_azar", "psi_maximo", "variable_psi_maximo",
                    "estado_general"]
        with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
            print("\nResumen por periodo")
            print(resumen[columnas].to_string(index=False))
    print(f"\nEstado general del tramo actual: {estado_general(resultado['drift_actual'])}")
    print("Alertas:")
    for alerta in resultado["alertas"] or [{"mensaje": "Ninguna."}]:
        print(f"  - {alerta['mensaje']}")
    print("Recomendaciones:")
    for rec in resultado["recomendaciones"]:
        print(f"  - [{rec['nivel']}] {rec['mensaje']}")
    salidas = MON["salidas"]
    print(f"\nArchivos generados: {salidas['tabla_predicciones']}, {salidas['historial_drift']}, "
          f"{salidas['resumen_periodos']}")


# ---------------------------------------------------------------------------------------------
# 4. Gráficos
# ---------------------------------------------------------------------------------------------

def grafico_psi(drift: pd.DataFrame):
    """Barras de riesgo: PSI por variable con el color del semáforo y las líneas de umbral."""
    datos = drift.sort_values("psi")
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(datos))))
    ax.barh(datos["variable"], datos["psi"], color=[COLORES.get(e, "gray") for e in datos["estado"]])
    ax.axvline(MON["psi_alerta"], ls="--", color=COLORES["Amarillo"], label=f"Alerta ({MON['psi_alerta']})")
    ax.axvline(MON["psi_critico"], ls="--", color=COLORES["Rojo"], label=f"Crítico ({MON['psi_critico']})")
    ax.set_xlabel("PSI")
    ax.set_title("Índice de estabilidad por variable")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def grafico_comparativo(referencia: pd.Series, actual: pd.Series, variable: str, continua: bool):
    """Distribución histórica frente a actual de una variable."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    if continua:
        ref, act = referencia.dropna(), actual.dropna()
        transformar = variable in VARIABLES_LOG
        if transformar:  # los montos se ven mejor en escala logarítmica
            ref, act = np.log1p(ref.clip(lower=0)), np.log1p(act.clip(lower=0))
        bins = np.histogram_bin_edges(pd.concat([ref, act]), bins=30)
        axes[0].hist(ref, bins=bins, density=True, alpha=0.55, label="Histórico", color="steelblue")
        axes[0].hist(act, bins=bins, density=True, alpha=0.55, label="Actual", color="darkorange")
        axes[0].set_title(f"Densidad{' (log1p)' if transformar else ''}")
        for serie, etiqueta, color in [(ref, "Histórico", "steelblue"), (act, "Actual", "darkorange")]:
            ordenada = np.sort(serie)
            axes[1].plot(ordenada, np.arange(1, len(ordenada) + 1) / len(ordenada), label=etiqueta, color=color)
        axes[1].set_title("Distribución acumulada (base de la prueba KS)")
    else:
        ref = a_categoria(referencia).value_counts(normalize=True)
        act = a_categoria(actual).value_counts(normalize=True)
        tabla = pd.concat([ref.rename("Histórico"), act.rename("Actual")], axis=1).fillna(0).sort_index()
        tabla.plot(kind="bar", ax=axes[0], color=["steelblue", "darkorange"])
        axes[0].set_title("Proporción por categoría")
        axes[0].tick_params(axis="x", rotation=0)
        (tabla["Actual"] - tabla["Histórico"]).plot(kind="bar", ax=axes[1], color="gray")
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Diferencia de proporción (actual - histórico)")
        axes[1].tick_params(axis="x", rotation=0)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    fig.suptitle(variable)
    fig.tight_layout()
    return fig


def grafico_evolucion_psi(historial: pd.DataFrame, variables: list):
    """Evolución del PSI por periodo de las variables elegidas, con los umbrales del semáforo."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for variable in variables:
        datos = historial[historial["variable"] == variable].sort_values("periodo")
        ax.plot(datos["periodo"], datos["psi"], marker="o", label=variable)
    ax.axhline(MON["psi_alerta"], ls="--", color=COLORES["Amarillo"], label="Alerta")
    ax.axhline(MON["psi_critico"], ls="--", color=COLORES["Rojo"], label="Crítico")
    ax.set(ylabel="PSI", title="Evolución del drift por periodo")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def grafico_mapa_estados(historial: pd.DataFrame):
    """Mapa de calor periodo x variable con el estado del semáforo."""
    matriz = (historial.assign(nivel=historial["estado"].map(ORDEN_ESTADO))
              .pivot(index="variable", columns="periodo", values="nivel"))
    matriz = matriz.loc[matriz.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.3 * len(matriz))))
    cmap = ListedColormap([COLORES["Verde"], COLORES["Amarillo"], COLORES["Rojo"]])
    ax.imshow(matriz.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(matriz.shape[1]), matriz.columns, rotation=45, ha="right")
    ax.set_yticks(range(matriz.shape[0]), matriz.index, fontsize=8)
    ax.set_title("Semáforo por variable y periodo")
    fig.tight_layout()
    return fig


def grafico_resumen_periodos(resumen: pd.DataFrame):
    """Tasa de impago real frente a la probabilidad media predicha y PSI de las predicciones."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(resumen["periodo"], resumen["tasa_impago_real_%"], marker="o", label="Impago real")
    # Se compara con el % de créditos que el modelo marca como impago (con su umbral) y no con
    # la probabilidad media: el modelo se entrenó con pesos de clase y su puntaje no está
    # calibrado (ronda el 45 % aunque el impago real sea cercano al 5 %).
    axes[0].plot(resumen["periodo"], resumen["impagos_predichos_%"], marker="s",
                 label="Marcados como impago por el modelo")
    axes[0].axhline(resumen["tasa_impago_referencia_%"].iloc[0], ls="--", color="gray",
                    label="Impago en referencia")
    axes[0].set(ylabel="%", title="Impago observado vs predicho")
    axes[1].bar(resumen["periodo"], resumen["psi_predicciones"],
                color=[COLORES[semaforo(v)] if not pd.isna(v) else "gray" for v in resumen["psi_predicciones"]])
    axes[1].axhline(MON["psi_alerta"], ls="--", color=COLORES["Amarillo"])
    axes[1].axhline(MON["psi_critico"], ls="--", color=COLORES["Rojo"])
    axes[1].plot(resumen["periodo"], resumen["psi_esperado_por_azar"], ls=":", marker=".",
                 color="black", label="Esperado por azar")
    axes[1].legend(fontsize=8)
    axes[1].set(ylabel="PSI", title="Drift de las predicciones")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# 5. Interfaz Streamlit
# ---------------------------------------------------------------------------------------------

def mostrar_mensajes(st, mensajes: list) -> None:
    funciones = {"Rojo": st.error, "Amarillo": st.warning, "Verde": st.success}
    for m in mensajes:
        funciones.get(m["nivel"], st.info)(m["mensaje"])


def main():
    import streamlit as st

    st.set_page_config(page_title="Monitoreo de drift", layout="wide")
    st.title("Monitoreo de data drift - modelo de riesgo de impago")
    st.caption("Compara la distribución histórica (referencia) con la actual en las variables del "
               "modelo y en sus predicciones. El semáforo usa el PSI; Jensen-Shannon, KS y "
               "chi-cuadrado se muestran como apoyo.")

    @st.cache_data
    def datos_puntuados():
        return construir_tabla_predicciones(cargar_datos(), cargar_artefacto())

    @st.cache_data
    def puntuar_csv(crudo):
        return construir_tabla_predicciones(crudo, cargar_artefacto())

    try:
        datos = datos_puntuados()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.sidebar:
        st.header("Datos actuales")
        fuente = st.radio("Fuente", ["Simulación temporal", "Subir CSV"])
        if fuente == "Simulación temporal":
            fecha_corte = st.date_input("Fecha de corte",
                                        value=pd.Timestamp(MON["fecha_corte_simulacion"]).date())
            tabla_ref, tabla_act = separar_periodos(datos, str(fecha_corte))
        else:
            archivo = st.file_uploader("CSV con las columnas de Base_de_datos.csv", type="csv")
            if archivo is None:
                st.info("Suba un archivo para compararlo contra todo el histórico.")
                st.stop()
            crudo = pd.read_csv(archivo)
            faltantes = [c for c in COLUMNAS_ENTRADA if c not in crudo.columns]
            if faltantes:
                st.error(f"El archivo no tiene las columnas esperadas: {faltantes}")
                st.stop()
            tabla_ref, tabla_act = datos, puntuar_csv(crudo)
        st.divider()
        st.markdown(f"**Umbrales PSI**  \nAlerta: {MON['psi_alerta']}  \nCrítico: {MON['psi_critico']}")
        st.markdown(f"**Periodicidad**: {MON['frecuencia']}  \n**Muestra por periodo**: "
                    f"{MON['tamano_muestra']}")

    if len(tabla_ref) < 50 or len(tabla_act) < 50:
        st.error("Cada periodo necesita al menos 50 registros. Ajuste la fecha de corte o el archivo.")
        st.stop()

    # Se monitorean las variables tal como entran al modelo (limpias y con derivados) y el puntaje.
    referencia, actual = features_con_puntaje(tabla_ref), features_con_puntaje(tabla_act)
    drift = calcular_drift(referencia, actual)
    general = estado_general(drift)
    historial, resumen = analisis_temporal(tabla_ref, tabla_act)
    tendencias = detectar_tendencias(historial)
    tasa_ref, tasa_act = tasa_impago(tabla_ref), tasa_impago(tabla_act)

    pestana_estado, pestana_tiempo, pestana_variable = st.tabs(
        ["Estado actual", "Análisis temporal", "Histórico vs actual"])

    with pestana_estado:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Registros históricos", f"{len(referencia):,}")
        c2.metric("Registros actuales", f"{len(actual):,}")
        c3.metric("Variables en rojo", int((drift["estado"] == "Rojo").sum()))
        c4.metric("Variables en amarillo", int((drift["estado"] == "Amarillo").sum()))
        c5.markdown(f"<div style='background:{COLORES[general]};color:white;padding:18px;"
                    f"border-radius:8px;text-align:center;font-weight:bold'>Estado general: "
                    f"{general.upper()}</div>", unsafe_allow_html=True)

        if not pd.isna(tasa_act):
            st.info(f"Tasa de impago: histórico {tasa_ref:.2f} % | actual {tasa_act:.2f} % "
                    f"(diferencia {tasa_act - tasa_ref:+.2f} puntos).")

        col_alertas, col_recomendaciones = st.columns(2)
        with col_alertas:
            st.subheader("Alertas")
            alertas = generar_alertas(drift)
            if not alertas:
                st.success(f"Sin desviaciones significativas: todas las variables tienen PSI menor "
                           f"que {MON['psi_alerta']}.")
            mostrar_mensajes(st, alertas)
        with col_recomendaciones:
            st.subheader("Recomendaciones")
            mostrar_mensajes(st, generar_recomendaciones(drift, tasa_ref, tasa_act, tendencias))

        col_tabla, col_barras = st.columns([3, 2])
        with col_tabla:
            st.subheader("Métricas de drift por variable")

            def colorear(fila):
                color = COLORES.get(fila["estado"], "")
                return [f"background-color:{color};color:white" if c == "estado" else "" for c in fila.index]

            st.dataframe(drift.style.apply(colorear, axis=1).format(
                {"psi": "{:.4f}", "js_divergencia": "{:.4f}", "p_valor": "{:.4f}"}),
                width="stretch", height=560)
            st.download_button("Descargar reporte CSV", drift.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"reporte_drift_{datetime.now():%Y%m%d_%H%M}.csv", mime="text/csv")
        with col_barras:
            st.subheader("Barras de riesgo")
            st.pyplot(grafico_psi(drift))

    with pestana_tiempo:
        if resumen.empty:
            st.warning("El periodo actual no tiene periodos con suficientes registros "
                       f"(mínimo {MON['min_registros_periodo']}) para el análisis temporal.")
        else:
            st.caption(f"Cada periodo ({MON['frecuencia']}) se compara contra la referencia con una "
                       f"muestra aleatoria de hasta {MON['tamano_muestra']} registros. Periodos con "
                       f"menos de {MON['min_registros_periodo']} registros se omiten.")
            st.subheader("Resumen por periodo")
            st.dataframe(resumen, width="stretch", hide_index=True)
            st.pyplot(grafico_resumen_periodos(resumen))

            st.subheader("Evolución del drift")
            por_defecto = tendencias["variable"].head(5).tolist()
            elegidas = st.multiselect("Variables", sorted(historial["variable"].unique()), por_defecto)
            if elegidas:
                st.pyplot(grafico_evolucion_psi(historial, elegidas))
            st.pyplot(grafico_mapa_estados(historial))

            st.subheader("Tendencias y cambios abruptos")
            st.dataframe(tendencias.style.format({c: "{:.4f}" for c in tendencias.columns
                                                  if c.startswith(("psi", "pendiente"))}),
                         width="stretch", hide_index=True)
            abruptos = tendencias[tendencias["cambio_abrupto"] != "-"]
            for _, fila in abruptos.iterrows():
                st.warning(f"Cambio abrupto en {fila['variable']} a partir de {fila['cambio_abrupto']}.")
            st.download_button("Descargar historial de drift",
                               historial.to_csv(index=False).encode("utf-8-sig"),
                               file_name="historial_drift.csv", mime="text/csv")

    with pestana_variable:
        variable = st.selectbox("Variable", drift["variable"],
                                format_func=lambda v: f"{v} ({drift.set_index('variable').loc[v, 'estado']})")
        st.pyplot(grafico_comparativo(referencia[variable], actual[variable], variable,
                                      variable in VARIABLES_CONTINUAS))


if __name__ == "__main__":
    if "--job" in sys.argv:
        imprimir_job(ejecutar_job())
    else:
        main()
