"""
test_pipeline.py

Pruebas automáticas del flujo (pytest). Las ejecuta Jenkins y su cobertura se publica en
SonarCloud. Cubren las reglas de limpieza, el preprocesamiento, las funciones de evaluación,
el cálculo de drift y la API.

Ejecución (desde la raíz del repositorio):  pytest mlops_pipeline -v --cov=mlops_pipeline --cov-report=xml
Las pruebas de la API y del trabajo de monitoreo necesitan mejor_modelo.joblib
(python mlops_pipeline/model_training_evaluation.py).
"""

import io

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ft_engineering import (COLUMNAS_ENTRADA, VARIABLES_MODELO, construir_dataset,
                            construir_preprocesador, generar_features, limpiar_datos)
from model_training_evaluation import build_model, summarize_classification, umbral_optimo_f1
from model_monitoring import (VARIABLE_PUNTAJE, analisis_temporal, calcular_drift,
                              detectar_tendencias, estado_general, generar_recomendaciones,
                              jensen_shannon, psi_categorico, psi_esperado_por_azar, psi_numerico,
                              semaforo)


@pytest.fixture(scope="module")
def dataset():
    return construir_dataset()


@pytest.fixture
def registro_crudo():
    """Un crédito válido con las columnas de la fuente."""
    return pd.DataFrame([{
        "tipo_credito": 4, "fecha_prestamo": "2025-01-31 10:00:00", "capital_prestado": 1_000_000,
        "plazo_meses": 12, "edad_cliente": 35, "tipo_laboral": "Empleado",
        "salario_cliente": 2_000_000, "total_otros_prestamos": 500_000, "cuota_pactada": 100_000,
        "puntaje": 95.2, "puntaje_datacredito": 780, "cant_creditosvigentes": 3,
        "huella_consulta": 2, "saldo_mora": 0, "saldo_total": 10_000, "saldo_principal": 9_000,
        "saldo_mora_codeudor": 0, "creditos_sectorFinanciero": 2, "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 1, "promedio_ingresos_datacredito": 1_500_000,
        "tendencia_ingresos": "Creciente",
    }])


# ---------------------------------------------------------------------------------------------
# Ingeniería de características
# ---------------------------------------------------------------------------------------------

def test_limpieza_convierte_valores_invalidos_en_nulos(registro_crudo):
    sucio = registro_crudo.assign(edad_cliente=122, puntaje_datacredito=0, tendencia_ingresos="8315")
    limpio = limpiar_datos(sucio)
    assert limpio[["edad_cliente", "puntaje_datacredito", "tendencia_ingresos"]].isna().all(axis=None)


def test_generar_features_crea_derivados_y_excluye_puntaje(registro_crudo):
    features = generar_features(registro_crudo)
    assert list(features.columns) == VARIABLES_MODELO
    assert "puntaje" not in features.columns
    assert features.loc[0, "ratio_cuota_salario"] == pytest.approx(0.05)
    assert features.loc[0, "es_fin_de_mes"] == 1
    assert features.loc[0, "tiene_mora"] == 0


def test_salario_cero_no_genera_infinitos(registro_crudo):
    features = generar_features(registro_crudo.assign(salario_cliente=0))
    assert not np.isinf(features.select_dtypes("number").to_numpy(dtype=float)).any()


def test_particion_estratificada(dataset):
    X_train, X_test, y_train, y_test, _ = dataset
    assert len(X_train) + len(X_test) == 10_763
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_preprocesador_sin_nulos_y_tolera_categorias_nuevas(dataset, registro_crudo):
    X_train, *_ = dataset
    preprocesador = construir_preprocesador().fit(X_train)
    assert not np.isnan(preprocesador.transform(X_train)).any()
    nuevo = generar_features(registro_crudo.assign(tipo_credito=999))
    assert preprocesador.transform(nuevo).shape[1] == preprocesador.transform(X_train).shape[1]


# ---------------------------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------------------------

def test_summarize_classification_calcula_metricas_correctas():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = summarize_classification(y_true, y_proba, umbral=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_umbral_optimo_separa_clases_perfectas():
    y_true = np.array([0, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < umbral_optimo_f1(y_true, y_proba) <= 0.8


def test_build_model_entrena_y_predice_probabilidades(dataset):
    X_train, X_test, y_train, _, preprocesador = dataset
    modelo = build_model(LogisticRegression(max_iter=500), preprocesador).fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)[:, 1]
    assert proba.shape == (len(X_test),)
    assert ((proba >= 0) & (proba <= 1)).all()


# ---------------------------------------------------------------------------------------------
# Monitoreo
# ---------------------------------------------------------------------------------------------

def test_psi_es_cero_sin_cambios_y_alto_con_desplazamiento():
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numerico(base, base) == pytest.approx(0, abs=1e-9)
    assert psi_numerico(base, base + 1.5) > 0.25


def test_psi_categorico_detecta_cambio_de_proporciones():
    ref = pd.Series(["A"] * 80 + ["B"] * 20)
    assert psi_categorico(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi_categorico(ref, pd.Series(["A"] * 30 + ["B"] * 70)) > 0.25


def test_semaforo_respeta_umbrales():
    assert semaforo(0.05) == "Verde"
    assert semaforo(0.15) == "Amarillo"
    assert semaforo(0.30) == "Rojo"


def test_sin_drift_en_particion_aleatoria(dataset):
    X_train, X_test, *_ = dataset
    assert estado_general(calcular_drift(X_train, X_test)) == "Verde"


def test_jensen_shannon_acotada_y_detecta_cambio():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 1, 5000))
    assert jensen_shannon(base, base, continua=True) == pytest.approx(0, abs=1e-9)
    desplazada = jensen_shannon(base, base + 3, continua=True)
    assert 0.3 < desplazada <= 1
    ref = pd.Series(["A"] * 50 + ["B"] * 50)
    assert jensen_shannon(ref, pd.Series(["C"] * 100), continua=False) == pytest.approx(1)


def test_psi_esperado_por_azar_disminuye_con_el_tamano():
    assert psi_esperado_por_azar(5000, 100) > psi_esperado_por_azar(5000, 1000)


def test_detectar_tendencias_y_cambios_abruptos():
    historial = pd.DataFrame({
        "periodo": ["2025-01", "2025-02", "2025-03", "2025-04"] * 2,
        "variable": ["sube"] * 4 + ["estable"] * 4,
        "psi": [0.01, 0.05, 0.30, 0.35, 0.02, 0.02, 0.03, 0.02],
        "estado": ["Verde", "Verde", "Rojo", "Rojo", "Verde", "Verde", "Verde", "Verde"],
    })
    tendencias = detectar_tendencias(historial).set_index("variable")
    assert tendencias.loc["sube", "tendencia"] == "Creciente"
    assert tendencias.loc["sube", "cambio_abrupto"] == "2025-03"
    assert tendencias.loc["estable", "tendencia"] == "Estable"
    assert tendencias.loc["estable", "cambio_abrupto"] == "-"


def test_recomendaciones_piden_reentrenar_con_drift_critico():
    drift = pd.DataFrame({
        "variable": ["a", "b", "c"], "tipo": ["Numérica"] * 3, "psi": [0.4, 0.3, 0.01],
        "estado": ["Rojo", "Rojo", "Verde"], "nulos_ref_%": [0, 0, 0], "nulos_act_%": [0, 0, 0],
    })
    mensajes = [r["mensaje"] for r in generar_recomendaciones(drift)]
    assert any("Reentrenar" in m for m in mensajes)
    sin_drift = drift.assign(estado="Verde", psi=0.01)
    assert generar_recomendaciones(sin_drift)[0]["nivel"] == "Verde"


def test_analisis_temporal_genera_historial_por_periodo(dataset):
    X_train, X_test, *_ = dataset
    pytest.importorskip("joblib")
    from ft_engineering import cargar_datos
    from model_monitoring import cargar_artefacto, construir_tabla_predicciones, separar_periodos

    tabla = construir_tabla_predicciones(cargar_datos(), cargar_artefacto())
    referencia, actual = separar_periodos(tabla, "2025-07-01")
    historial, resumen = analisis_temporal(referencia, actual)
    assert not resumen.empty
    assert set(historial["periodo"]) == set(resumen["periodo"])
    assert VARIABLE_PUNTAJE in set(historial["variable"])
    assert {"psi", "js_divergencia", "p_valor", "estado"} <= set(historial.columns)


# ---------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cliente():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from model_deploy import app
    return TestClient(app)


def test_api_health(cliente):
    assert cliente.get("/health").json()["estado"] == "ok"


def test_api_predict_json_por_lotes(cliente, registro_crudo):
    registros = pd.concat([registro_crudo] * 3).to_dict(orient="records")
    respuesta = cliente.post("/predict", json={"registros": registros})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["n_registros"] == 3
    assert 0 <= cuerpo["predicciones"][0]["probabilidad_impago"] <= 1


def test_api_predict_csv(cliente, registro_crudo):
    archivo = io.BytesIO(registro_crudo[COLUMNAS_ENTRADA].to_csv(index=False).encode())
    respuesta = cliente.post("/predict/csv", files={"archivo": ("lote.csv", archivo, "text/csv")})
    assert respuesta.status_code == 200
    assert respuesta.json()["n_registros"] == 1


def test_api_rechaza_entradas_invalidas(cliente):
    assert cliente.post("/predict", json={"registros": []}).status_code == 422
    assert cliente.post("/predict", json={"registros": [{"tipo_credito": 4}]}).status_code == 422
    malo = io.BytesIO(b"a,b\n1,2\n")
    assert cliente.post("/predict/csv", files={"archivo": ("x.csv", malo, "text/csv")}).status_code == 422
