"""
model_training_evaluation.py

Segunda componente del flujo: entrena y evalúa varios modelos supervisados sobre los conjuntos
generados por ft_engineering.py y guarda el mejor como objeto reutilizable.

Proceso
-------
1. Cada modelo se arma con build_model(): preprocesador + estimador en un único pipeline, para
   que la imputación, el escalado y la codificación se ajusten solo con los datos de
   entrenamiento de cada fold (sin fuga de información).
2. Validación cruzada estratificada sobre el conjunto de entrenamiento. De ella salen:
   - rendimiento: AUC-PR (average precision), métrica principal por el desbalance de clases;
   - consistencia: desviación entre folds y brecha entre entrenamiento y validación;
   - escalabilidad: tiempo de entrenamiento y de predicción;
   - umbral de decisión que maximiza F1 sobre las predicciones fuera de fold.
3. Evaluación final en el conjunto de evaluación, que no se usó para ninguna decisión.
4. Selección por un puntaje ponderado de rendimiento, consistencia y escalabilidad (pesos en
   config.json), tabla resumen, gráficos comparativos y guardado del mejor modelo.

Por qué AUC-PR y no accuracy: solo el 4,7 % de los créditos son impagos; un modelo que siempre
predice "paga" tiene 95 % de accuracy y no detecta ningún impago. La AUC-PR de un modelo sin
información es igual a la tasa de impago (cerca de 0,047), que es la línea base de referencia.
"""

import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")  # genera los gráficos como archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from ft_engineering import CONFIG, RAIZ, construir_dataset, generar_features

warnings.filterwarnings("ignore", category=UserWarning)
SEMILLA = CONFIG["particion"]["random_state"]
NOMBRE_BASE = "Línea base"


# ---------------------------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------------------------

def build_model(estimador, preprocesador) -> Pipeline:
    """Une el preprocesador (sin ajustar) y el estimador en un único pipeline.

    Se clona el preprocesador para que cada modelo tenga su propia copia y ningún ajuste se
    comparta entre modelos o entre folds.
    """
    return Pipeline([
        ("preprocesamiento", clone(preprocesador)),
        ("modelo", estimador),
    ])


def summarize_classification(y_true, y_proba, umbral: float = 0.5) -> dict:
    """Métricas de clasificación para la clase positiva (impago).

    - roc_auc y average_precision no dependen del umbral: miden la capacidad de ordenar clientes.
    - precision, recall, f1 y la matriz de confusión dependen del umbral de decisión.
    """
    y_pred = (y_proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def umbral_optimo_f1(y_true, y_proba) -> float:
    """Umbral que maximiza F1. Se calcula con predicciones fuera de fold del entrenamiento,
    nunca con el conjunto de evaluación, para no sobreestimar el resultado final."""
    precision, recall, umbrales = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # El último punto de la curva no tiene umbral asociado.
    return float(umbrales[np.argmax(f1[:-1])]) if len(umbrales) else 0.5


def validacion_cruzada(modelo: Pipeline, X, y, n_folds: int) -> dict:
    """Validación cruzada estratificada que registra rendimiento, consistencia y tiempos.

    Se implementa con un bucle propio (en lugar de cross_validate) para obtener en una sola
    pasada las métricas por fold, las predicciones fuera de fold y los tiempos de predicción.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    proba_oof = np.zeros(len(y))
    ap_val, ap_train, roc_val, tiempos_fit, tiempos_pred = [], [], [], [], []

    for indices_train, indices_val in cv.split(X, y):
        X_tr, X_val = X.iloc[indices_train], X.iloc[indices_val]
        y_tr, y_val = y.iloc[indices_train], y.iloc[indices_val]
        m = clone(modelo)

        inicio = time.perf_counter()
        m.fit(X_tr, y_tr)
        tiempos_fit.append(time.perf_counter() - inicio)

        inicio = time.perf_counter()
        proba_val = m.predict_proba(X_val)[:, 1]
        # Milisegundos por cada 1.000 registros: segundos / filas * 1.000 filas * 1.000 ms.
        tiempos_pred.append((time.perf_counter() - inicio) / len(X_val) * 1000 * 1000)

        proba_oof[indices_val] = proba_val
        ap_val.append(average_precision_score(y_val, proba_val))
        roc_val.append(roc_auc_score(y_val, proba_val))
        ap_train.append(average_precision_score(y_tr, m.predict_proba(X_tr)[:, 1]))

    umbral = umbral_optimo_f1(y, proba_oof)
    oof = summarize_classification(y, proba_oof, umbral)
    return {
        "cv_ap_media": np.mean(ap_val), "cv_ap_desv": np.std(ap_val),
        "cv_ap_folds": ap_val,
        "cv_roc_media": np.mean(roc_val), "cv_roc_desv": np.std(roc_val),
        "cv_f1": oof["f1"], "cv_recall": oof["recall"], "cv_precision": oof["precision"],
        # Brecha positiva grande = el modelo memoriza el entrenamiento (sobreajuste).
        "brecha_train_val": np.mean(ap_train) - np.mean(ap_val),
        "tiempo_fit_s": np.mean(tiempos_fit), "tiempo_pred_ms_1000": np.mean(tiempos_pred),
        "umbral": umbral,
    }


def normalizar(serie: pd.Series, mayor_es_mejor: bool = True) -> pd.Series:
    """Escala min-max a [0, 1], donde 1 es siempre lo mejor."""
    rango = serie.max() - serie.min()
    if rango == 0:
        return pd.Series(1.0, index=serie.index)
    escala = (serie - serie.min()) / rango
    return escala if mayor_es_mejor else 1 - escala


def seleccionar_modelo(resumen: pd.DataFrame, pesos: dict) -> pd.DataFrame:
    """Puntaje ponderado de selección. La línea base se excluye: solo sirve de referencia.

    - rendimiento: AUC-PR media en validación cruzada.
    - consistencia: penaliza la variación entre folds y el sobreajuste (brecha train-val).
    - escalabilidad: penaliza el tiempo de entrenamiento y de predicción (en escala logarítmica,
      porque las diferencias entre modelos son de órdenes de magnitud).
    """
    candidatos = resumen.drop(index=NOMBRE_BASE)
    inconsistencia = candidatos["cv_ap_desv"] + candidatos["brecha_train_val"].clip(lower=0)
    costo = np.log10(candidatos["tiempo_fit_s"]) + np.log10(candidatos["tiempo_pred_ms_1000"])

    resumen.loc[candidatos.index, "puntaje_rendimiento"] = normalizar(candidatos["cv_ap_media"])
    resumen.loc[candidatos.index, "puntaje_consistencia"] = normalizar(inconsistencia, mayor_es_mejor=False)
    resumen.loc[candidatos.index, "puntaje_escalabilidad"] = normalizar(costo, mayor_es_mejor=False)
    resumen["puntaje_total"] = (pesos["rendimiento"] * resumen["puntaje_rendimiento"]
                                + pesos["consistencia"] * resumen["puntaje_consistencia"]
                                + pesos["escalabilidad"] * resumen["puntaje_escalabilidad"])
    resumen["seleccionado"] = resumen.index == resumen["puntaje_total"].idxmax()
    return resumen


# ---------------------------------------------------------------------------------------------
# Modelos candidatos
# ---------------------------------------------------------------------------------------------

def definir_modelos() -> dict:
    """Modelos de complejidad creciente. Todos compensan el desbalance con pesos de clase,
    que no altera los datos (a diferencia del sobremuestreo) y no requiere librerías extra."""
    return {
        # Referencia mínima: predice siempre la proporción de impago. Cualquier modelo útil
        # debe superarla con claridad.
        NOMBRE_BASE: DummyClassifier(strategy="prior"),
        # Modelo lineal interpretable; se beneficia del log y el escalado del preprocesador.
        "Regresión logística": LogisticRegression(class_weight="balanced", max_iter=2000,
                                                  random_state=SEMILLA),
        # Árbol podado (profundidad y hojas mínimas) para evitar que memorice los 400 impagos.
        "Árbol de decisión": DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                    class_weight="balanced", random_state=SEMILLA),
        # Ensamble por bagging: reduce la varianza del árbol individual. Profundidad y hojas
        # limitadas: en una primera prueba con hojas de 10 registros la brecha train-val de
        # AUC-PR fue de 0,84 (memorización) sin mejorar la validación.
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                                class_weight="balanced_subsample", n_jobs=-1,
                                                random_state=SEMILLA),
        # Ensamble por boosting: captura interacciones como plazo x producto (EDA 6.4).
        # Árboles poco profundos y regularización L2: con 15 hojas por árbol la brecha
        # train-val fue de 0,72; con profundidad 2 baja a cerca de 0,1 con la misma AUC-PR.
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150,
                                                            max_depth=2, min_samples_leaf=100,
                                                            l2_regularization=5.0,
                                                            class_weight="balanced",
                                                            random_state=SEMILLA),
    }


# ---------------------------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------------------------

def graficar_comparacion(resumen, curvas, mejor_nombre, mejor_modelo, X_test, y_test, ruta):
    """Panel con la comparación de los modelos y el diagnóstico del seleccionado."""
    modelos = list(resumen.index)
    colores = dict(zip(modelos, plt.cm.tab10.colors, strict=False))
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    # 1. Métricas de validación cruzada con su desviación entre folds.
    ax = axes[0, 0]
    x = np.arange(len(modelos))
    ax.bar(x - 0.2, resumen["cv_ap_media"], 0.4, yerr=resumen["cv_ap_desv"], capsize=3,
           label="AUC-PR", color="steelblue")
    ax.bar(x + 0.2, resumen["cv_roc_media"], 0.4, yerr=resumen["cv_roc_desv"], capsize=3,
           label="ROC-AUC", color="lightgray")
    ax.set_xticks(x, modelos, rotation=25, ha="right")
    ax.set_title("Validación cruzada (media y desviación)")
    ax.legend()

    # 2. Distribución de la AUC-PR por fold: cajas estrechas = modelo consistente.
    ax = axes[0, 1]
    ax.boxplot([resumen.loc[m, "cv_ap_folds"] for m in modelos], tick_labels=modelos)
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Consistencia: AUC-PR por fold")

    # Las curvas omiten la línea base: su referencia ya es la diagonal (ROC) o la tasa de
    # impago (precisión-recall), y su curva degenerada confundiría la lectura.
    curvas_modelos = {n: c for n, c in curvas.items() if n != NOMBRE_BASE}

    # 3. Curvas ROC en evaluación.
    ax = axes[0, 2]
    for nombre, (proba, _) in curvas_modelos.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{nombre} ({roc_auc_score(y_test, proba):.3f})", color=colores[nombre])
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Sin información (0.500)")
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title="Curva ROC (evaluación)")
    ax.legend(fontsize=8)

    # 4. Curvas precisión-recall en evaluación, la vista más informativa con clases desbalanceadas.
    ax = axes[0, 3]
    for nombre, (proba, _) in curvas_modelos.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ax.plot(recall, precision, label=f"{nombre} ({average_precision_score(y_test, proba):.3f})",
                color=colores[nombre])
    ax.axhline(y_test.mean(), ls="--", color="gray", label=f"Tasa de impago ({y_test.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Curva precisión-recall (evaluación)")
    ax.legend(fontsize=8)

    # 5. Escalabilidad: tiempos en escala logarítmica.
    ax = axes[1, 0]
    ax.bar(x - 0.2, resumen["tiempo_fit_s"], 0.4, label="Entrenamiento (s)", color="darkorange")
    ax.bar(x + 0.2, resumen["tiempo_pred_ms_1000"], 0.4, label="Predicción (ms / 1.000 filas)",
           color="seagreen")
    ax.set_yscale("log")
    ax.set_xticks(x, modelos, rotation=25, ha="right")
    ax.set_title("Escalabilidad (escala logarítmica)")
    ax.legend()

    # 6. Puntaje de selección desagregado.
    ax = axes[1, 1]
    pesos = CONFIG["entrenamiento"]["pesos_seleccion"]
    candidatos = resumen.drop(index=NOMBRE_BASE)
    acumulado = np.zeros(len(candidatos))
    for criterio, color in zip(["rendimiento", "consistencia", "escalabilidad"],
                               ["steelblue", "mediumpurple", "seagreen"], strict=True):
        valores = pesos[criterio] * candidatos[f"puntaje_{criterio}"].values
        ax.barh(candidatos.index, valores, left=acumulado, color=color,
                label=f"{criterio} (peso {pesos[criterio]})")
        acumulado += valores
    ax.set_xlim(0, 1)
    ax.set_title("Puntaje de selección (mayor es mejor)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)

    # 7. Matriz de confusión del modelo seleccionado con su umbral.
    ax = axes[1, 2]
    proba, umbral = curvas[mejor_nombre]
    ConfusionMatrixDisplay.from_predictions(y_test, (proba >= umbral).astype(int),
                                            display_labels=["Paga", "Impago"], ax=ax,
                                            colorbar=False, cmap="Blues")
    ax.set(xlabel="Predicción", ylabel="Valor real",
           title=f"{mejor_nombre}: matriz de confusión (umbral {umbral:.2f})")

    # 8. Importancia por permutación: cuánto cae la AUC-PR al desordenar cada variable.
    ax = axes[1, 3]
    imp = permutation_importance(mejor_modelo, X_test, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=SEMILLA, n_jobs=-1)
    importancias = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(12)
    importancias.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{mejor_nombre}: importancia por permutación")
    ax.set_xlabel("Caída de AUC-PR")

    fig.suptitle("Comparación de modelos supervisados - probabilidad de impago", fontsize=15)
    plt.tight_layout()
    fig.savefig(ruta, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return importancias


# ---------------------------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------------------------

def entrenar_y_evaluar():
    X_train, X_test, y_train, y_test, preprocesador = construir_dataset()
    conf_ent = CONFIG["entrenamiento"]
    salidas = CONFIG["salidas"]

    filas, curvas, modelos_ajustados = {}, {}, {}
    for nombre, estimador in definir_modelos().items():
        print(f"Entrenando: {nombre}")
        modelo = build_model(estimador, preprocesador)
        cv = validacion_cruzada(modelo, X_train, y_train, conf_ent["cv_folds"])

        # Ajuste final con todo el entrenamiento y evaluación en datos no vistos.
        modelo.fit(X_train, y_train)
        proba_test = modelo.predict_proba(X_test)[:, 1]
        test = summarize_classification(y_test, proba_test, cv["umbral"])

        filas[nombre] = {**cv, **{f"test_{k}": v for k, v in test.items()}}
        curvas[nombre] = (proba_test, cv["umbral"])
        modelos_ajustados[nombre] = modelo

    # Al transponer, la columna con la lista de folds deja todo como object; se restauran los
    # tipos numéricos del resto de columnas.
    resumen = pd.DataFrame(filas).T
    numericas = resumen.columns.drop("cv_ap_folds")
    resumen[numericas] = resumen[numericas].astype(float)
    resumen = seleccionar_modelo(resumen, conf_ent["pesos_seleccion"])
    mejor_nombre = resumen.index[resumen["seleccionado"]][0]
    mejor_modelo = modelos_ajustados[mejor_nombre]

    # Tabla resumen (sin la lista de folds, que solo se usa en el gráfico).
    columnas = ["cv_ap_media", "cv_ap_desv", "cv_roc_media", "cv_f1", "brecha_train_val",
                "tiempo_fit_s", "tiempo_pred_ms_1000", "umbral",
                "test_average_precision", "test_roc_auc", "test_precision", "test_recall",
                "test_f1", "test_accuracy", "test_balanced_accuracy", "test_tp", "test_fp", "test_fn", "test_tn",
                "puntaje_rendimiento", "puntaje_consistencia", "puntaje_escalabilidad",
                "puntaje_total", "seleccionado"]
    tabla = resumen[columnas].copy()
    tabla.index.name = "modelo"
    conteos = ["test_tp", "test_fp", "test_fn", "test_tn"]
    tabla[conteos] = tabla[conteos].astype(int)
    tabla.to_csv(RAIZ / salidas["tabla_resumen"], encoding="utf-8-sig", float_format="%.4f")

    importancias = graficar_comparacion(resumen, curvas, mejor_nombre, mejor_modelo,
                                        X_test, y_test, RAIZ / salidas["grafico_comparativo"])

    # Objeto final: incluye generar_features para que en producción se le pasen los registros
    # crudos (mismas columnas que Base_de_datos.csv) y aplique exactamente la misma limpieza.
    modelo_produccion = Pipeline([
        ("features", FunctionTransformer(generar_features)),
        ("preprocesamiento", mejor_modelo.named_steps["preprocesamiento"]),
        ("modelo", mejor_modelo.named_steps["modelo"]),
    ])
    artefacto = {
        "modelo": modelo_produccion,
        "nombre_modelo": mejor_nombre,
        "umbral": float(resumen.loc[mejor_nombre, "umbral"]),
        "objetivo": "probabilidad de impago (1 - Pago_atiempo)",
        "metricas_test": {k.replace("test_", ""): float(v) for k, v in tabla.loc[mejor_nombre].items()
                          if k.startswith("test_")},
        "version_sklearn": sklearn.__version__,
        "fecha_entrenamiento": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artefacto, RAIZ / salidas["modelo"])

    imprimir_resumen(tabla, mejor_nombre, importancias, salidas)
    return artefacto, tabla


def imprimir_resumen(tabla, mejor_nombre, importancias, salidas):
    vista = tabla[["cv_ap_media", "cv_ap_desv", "cv_roc_media", "brecha_train_val",
                   "tiempo_fit_s", "umbral", "test_average_precision", "test_roc_auc",
                   "test_precision", "test_recall", "test_f1", "test_accuracy", "puntaje_total"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("\nTabla resumen de evaluación")
        print(vista.to_string())
    print(f"\nModelo seleccionado: {mejor_nombre}")
    fila = tabla.loc[mejor_nombre]
    print("Justificación:")
    print(f"  - Rendimiento: AUC-PR en validación cruzada de {fila['cv_ap_media']:.3f} "
          f"(la línea base es {tabla.loc[NOMBRE_BASE, 'cv_ap_media']:.3f}).")
    print(f"  - Consistencia: desviación entre folds de {fila['cv_ap_desv']:.3f} y brecha "
          f"entrenamiento-validación de {fila['brecha_train_val']:.3f}.")
    print(f"  - Escalabilidad: {fila['tiempo_fit_s']:.2f} s de entrenamiento y "
          f"{fila['tiempo_pred_ms_1000']:.2f} ms por cada 1.000 predicciones.")
    print("Variables más importantes (caída de AUC-PR al permutarlas):")
    print(importancias.sort_values(ascending=False).head(8).round(4).to_string())
    print(f"\nArchivos generados: {salidas['modelo']}, {salidas['tabla_resumen']}, "
          f"{salidas['grafico_comparativo']}")


if __name__ == "__main__":
    entrenar_y_evaluar()
