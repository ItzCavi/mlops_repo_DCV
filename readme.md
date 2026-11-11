# mlops_repo_DCV - Modelo de riesgo de impago de créditos

<!-- Reemplace PROJECT_KEY por el sonar.projectKey de sonar-project.properties para activar los badges. -->
[![CI - pruebas y SonarCloud](https://github.com/ItzCavi/mlops_repo_DCV/actions/workflows/ci.yml/badge.svg)](https://github.com/ItzCavi/mlops_repo_DCV/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=PROJECT_KEY&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=PROJECT_KEY)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=PROJECT_KEY&metric=coverage)](https://sonarcloud.io/summary/new_code?id=PROJECT_KEY)

Proyecto final de Machine Learning. Flujo completo de MLOps para un modelo supervisado que
estima la probabilidad de que un crédito **no se pague a tiempo**: carga de datos, análisis
exploratorio, ingeniería de características, entrenamiento y evaluación, despliegue como API,
monitoreo de data drift e integración continua (GitHub Actions, Jenkins y SonarCloud).

## 1. Caso de negocio

Una entidad que otorga créditos de consumo necesita priorizar la revisión y la cobranza
preventiva de los créditos con mayor riesgo de impago. El modelo asigna a cada crédito un puntaje
de riesgo en el momento del desembolso, usando la información de la solicitud y de la central de
riesgo.

- **Datos:** 10.763 créditos desembolsados entre noviembre de 2024 y abril de 2026, 23 columnas
  (`Base_de_datos.csv`).
- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo). El modelo predice su complemento,
  `impago = 1 - Pago_atiempo`, porque la clase de interés es la minoritaria: **4,7 % de impagos**.
- **Uso esperado:** marcar para revisión los créditos cuyo puntaje supera el umbral de decisión.
  Un falso negativo (impago no detectado) cuesta más que una revisión innecesaria.

## 2. Estructura del repositorio

```
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb             Lee BD_creditos.xlsx y materializa Base_de_datos.csv
│   ├── comprension_eda.ipynb          Análisis exploratorio, reglas de validación y derivados
│   ├── ft_engineering.py              Limpieza, atributos derivados, pipeline y partición
│   ├── model_training_evaluation.py   Entrena, compara y guarda el mejor modelo
│   ├── model_deploy.py                API FastAPI (/predict JSON y CSV, por lotes)
│   ├── model_monitoring.py            Trabajo de monitoreo de drift y app Streamlit
│   ├── test_pipeline.py               Pruebas automáticas (pytest)
│   ├── config.json                    Parámetros del flujo (rutas, umbrales, pesos)
│   ├── mejor_modelo.joblib            Modelo seleccionado (generado)
│   ├── resumen_evaluacion.csv         Tabla resumen de evaluación (generada)
│   └── comparacion_modelos.png        Gráficos comparativos (generado)
├── Base_de_datos.csv                  Tabla materializada (fuente del flujo)
├── BD_creditos.xlsx                   Fuente original (simula el DWH)
├── requirements.txt                   Dependencias del proyecto
├── requirements-deploy.txt            Dependencias mínimas de la imagen Docker
├── .gitignore
├── setup.bat                          Crea el entorno virtual e instala dependencias (Windows)
├── readme.md
├── Dockerfile, .dockerignore          Imagen de la API
├── Jenkinsfile, docker-compose.yml    Pipeline de Jenkins y contenedor local de Jenkins
├── .gitattributes                     Fin de línea de Windows para los .bat
├── sonar-project.properties           Configuración de SonarCloud
└── .github/workflows/ci.yml           Integración continua en GitHub Actions
```

### Ramas

| Rama | Uso |
|---|---|
| `developer` | Desarrollo. Todo cambio entra primero aquí. |
| `certification` | Validación: se integra desde `developer` cuando el pipeline pasa. |
| `master` | Producción: se integra desde `certification`. |

El pipeline de CI (GitHub Actions y Jenkins) valida en las tres ramas que la estructura de
carpetas no haya cambiado antes de entrenar y probar.

## 3. Entorno virtual

**Windows (automático):** ejecute `setup.bat` desde la raíz del repositorio. Crea la carpeta
`venv` si no existe e instala `requirements.txt`.

**Manual (Windows):**

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Manual (Linux / macOS):**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Active el entorno (`venv\Scripts\activate`) cada vez que abra una terminal nueva. La carpeta
`venv` está en `.gitignore` y no se sube al repositorio.

## 4. Orden de ejecución

Todos los comandos se ejecutan desde la **raíz del repositorio** con el entorno activo.

| Paso | Comando | Resultado |
|---|---|---|
| 1 | Abrir `mlops_pipeline/Cargar_datos.ipynb` | Genera `Base_de_datos.csv` |
| 2 | Abrir `mlops_pipeline/comprension_eda.ipynb` | Análisis exploratorio |
| 3 | `python mlops_pipeline/ft_engineering.py` | Resumen de los conjuntos y las features |
| 4 | `python mlops_pipeline/model_training_evaluation.py` | Modelo, tabla resumen y gráficos |
| 5 | `uvicorn model_deploy:app --app-dir mlops_pipeline --reload` | API en http://localhost:8000/docs |
| 6 | `python mlops_pipeline/model_monitoring.py --job` | Trabajo de monitoreo (tablas de drift) |
| 7 | `streamlit run mlops_pipeline/model_monitoring.py` | Tablero de monitoreo |
| 8 | `pytest mlops_pipeline -v` | Pruebas automáticas |
| 9 | `ruff check mlops_pipeline --line-length 120 --select E,F,W,B,UP --exclude "*.ipynb"` | Revisión de estilo |

## 5. Hallazgos del análisis exploratorio

- **Calidad de datos:** nulos concentrados en la información de ingresos de la central (27 %),
  150 edades centinela (121 a 123 años), 147 puntajes de central no válidos (0, 7, -7), 58 valores
  numéricos dentro de `tendencia_ingresos` y salarios en cero o extremos. Las reglas de
  validación R01 a R18 están en la sección 7 del notebook.
- **Desbalance:** 4,7 % de impagos, lo que condiciona las métricas (AUC-PR en lugar de accuracy)
  y obliga a usar partición estratificada y pesos de clase.
- **Posible fuga de información:** `puntaje` separa casi perfectamente las clases; se sospecha que
  se calcula después del pago y se excluye hasta confirmarlo con negocio.
- **Señales útiles (débiles pero consistentes):** plazo, mora previa, puntaje e ingreso estimado
  por la central, huella de consulta, edad, tipo laboral y tendencia de ingresos.
- **Variables monetarias** log-normales con atípicos extremos: requieren recorte y `log1p`.
- **Cambio en el tiempo:** el volumen de créditos y la tasa de impago cambian desde mediados de
  2025, por lo que el monitoreo de drift es necesario.

## 6. Ingeniería de características (`ft_engineering.py`)

- **Limpieza:** edades mayores que 100, puntajes de central menores que 100 y valores numéricos en
  `tendencia_ingresos` pasan a nulo.
- **Variables excluidas:** `puntaje` (sospecha de fuga), `saldo_mora_codeudor` (casi constante) y
  `fecha_prestamo` (se usan solo sus derivados de calendario).
- **Atributos derivados:** ratios cuota/salario, capital/salario, otros préstamos/salario y salario
  frente al ingreso estimado por la central; indicadores de mora, de falta de información en la
  central y de fin de mes o de semana.
- **Pipeline de scikit-learn (`ColumnTransformer`):** imputación por mediana, recorte en los
  percentiles 1 y 99, `log1p` para montos, escalado estándar, one-hot con agrupación de categorías
  raras y codificación ordinal de la tendencia de ingresos.
- **Partición:** 80/20 estratificada por la variable objetivo.

## 7. Entrenamiento y selección (`model_training_evaluation.py`)

Se entrenan una línea base y cuatro modelos con pesos de clase, usando `build_model()` y
`summarize_classification()`, con validación cruzada estratificada de 5 folds. El umbral de
decisión maximiza F1 sobre predicciones fuera de fold.

| Modelo | AUC-PR (CV) | ROC-AUC (CV) | AUC-PR (test) | ROC-AUC (test) | Precisión | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Línea base | 0,048 | 0,500 | 0,047 | 0,500 | 0,047 | 1,000 | 0,091 |
| **Regresión logística** | **0,148** | **0,692** | 0,132 | 0,655 | 0,203 | 0,137 | 0,164 |
| Árbol de decisión | 0,088 | 0,613 | 0,065 | 0,556 | 0,077 | 0,167 | 0,105 |
| Random Forest | 0,151 | 0,691 | 0,131 | 0,635 | 0,126 | 0,157 | 0,140 |
| Gradient Boosting | 0,146 | 0,689 | 0,151 | 0,647 | 0,138 | 0,206 | 0,165 |

La selección pondera **rendimiento** (60 %, AUC-PR en validación cruzada), **consistencia**
(20 %, desviación entre folds y brecha entrenamiento-validación) y **escalabilidad** (20 %,
tiempos de entrenamiento y predicción). Se elige la **regresión logística**: rendimiento en
validación equivalente al de los ensambles (0,148 frente a 0,151 y 0,146), la menor brecha
entrenamiento-validación (0,012 frente a 0,209 de Random Forest) y el menor costo de cómputo.

**Limitación:** el rendimiento es modesto. Sin la variable `puntaje` los datos tienen señales
débiles; el modelo triplica la AUC-PR de la línea base, pero conviene usarlo para priorizar y no
para decidir de forma automática. La tabla completa está en `mlops_pipeline/resumen_evaluacion.csv`
y los gráficos en `mlops_pipeline/comparacion_modelos.png`.

## 8. Despliegue (`model_deploy.py`)

```bash
uvicorn model_deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
```

Documentación interactiva: http://localhost:8000/docs

| Método | Ruta | Entrada |
|---|---|---|
| GET | `/health` | - |
| POST | `/predict` | Lote de registros en **JSON o CSV**, según el `Content-Type` (hasta 10.000 registros) |
| POST | `/predict/csv` | Atajo para subir un archivo CSV desde `/docs` |

`/predict` acepta tres formatos en el mismo endpoint:

```bash
# JSON
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"registros": [{"tipo_credito": 4, "fecha_prestamo": "2025-01-07 14:40:36", ...}]}'

# CSV en el cuerpo de la petición
curl -X POST http://localhost:8000/predict -H "Content-Type: text/csv" --data-binary @Base_de_datos.csv

# Archivo CSV (multipart)
curl -X POST http://localhost:8000/predict -F "archivo=@Base_de_datos.csv"
```

El CSV lleva las columnas de `Base_de_datos.csv` (`Pago_atiempo` es opcional y se ignora). La API
valida el esquema (422 con el detalle si falta un campo o un tipo no coincide), limita el tamaño
del archivo a 20 MB y responde 415 si el formato no es JSON ni CSV. El modelo guardado incluye
la limpieza y los atributos derivados de `ft_engineering.py`, así que se le envían los registros
crudos.

Respuesta:

```json
{
  "modelo": "Regresión logística",
  "umbral": 0.77,
  "n_registros": 1,
  "n_impagos_predichos": 0,
  "predicciones": [
    {"indice": 0, "probabilidad_impago": 0.3946, "prediccion_impago": 0,
     "resultado": "Pago a tiempo esperado"}
  ]
}
```

`probabilidad_impago` es un puntaje de riesgo: el modelo se entrenó con pesos de clase, por lo que
el valor no está calibrado como probabilidad real. La decisión usa el umbral elegido en el
entrenamiento.

### Docker

La imagen contiene el código fuente (`mlops_pipeline/`), las dependencias
(`requirements-deploy.txt`: las de `requirements.txt` que usa la API, sin Jupyter, Streamlit ni
herramientas de pruebas, para que la imagen sea más liviana), el servidor Uvicorn y la
configuración (`Dockerfile`, `.dockerignore`, `config.json`).

```bash
docker build -t riesgo-impago-api .
docker run -p 8000:8000 riesgo-impago-api
```

Durante la construcción se entrena el modelo, para que se serialice con las mismas versiones de
Python y scikit-learn que lo cargan. Luego la API queda en http://localhost:8000/docs y la imagen
tiene un `HEALTHCHECK` sobre `/health`. El contenedor corre con un usuario sin privilegios.

## 9. Monitoreo (`model_monitoring.py`)

**Trabajo periódico** (se programaría mensualmente, por ejemplo en Jenkins):

```bash
python mlops_pipeline/model_monitoring.py --job
```

1. Construye la tabla de datos con los pronósticos del modelo (`monitoreo_predicciones.csv`).
2. Toma como referencia los créditos anteriores a la fecha de corte (`2025-07-01`) y divide el
   periodo posterior en ventanas mensuales.
3. Toma una muestra aleatoria de hasta 500 registros por ventana y calcula, para cada variable del
   modelo y para el puntaje que entrega el modelo:
   - **PSI** (Population Stability Index): define el semáforo (verde < 0,10; amarillo de 0,10 a
     0,25; rojo >= 0,25).
   - **Divergencia de Jensen-Shannon** (base 2, entre 0 y 1).
   - **Kolmogorov-Smirnov** (numéricas) y **chi-cuadrado** (categóricas), con su p-valor.
4. Guarda el historial (`monitoreo_historial_drift.csv`) y el resumen por periodo
   (`monitoreo_resumen_periodos.csv`), detecta tendencias y cambios abruptos y genera alertas y
   recomendaciones (reentrenar, revisar variables, revisar calidad de datos o el umbral).

Los meses con menos de 150 créditos se omiten: el PSI de muestras pequeñas tiene un sesgo positivo
(con 76 registros, cerca de 0,12 solo por azar). La columna `psi_esperado_por_azar` muestra ese
valor de referencia en cada periodo.

**Aplicación Streamlit:**

```bash
streamlit run mlops_pipeline/model_monitoring.py
```

- **Estado actual:** indicadores, semáforo general, alertas, recomendaciones automáticas, tabla de
  métricas por variable (PSI, JS, KS / chi-cuadrado, nulos) y barras de riesgo.
- **Análisis temporal:** resumen por periodo, impago observado frente a predicho, drift de las
  predicciones, evolución del PSI, mapa de semáforos por variable y periodo, tendencias y cambios
  abruptos.
- **Histórico vs actual:** histogramas y distribución acumulada (numéricas) o proporciones
  (categóricas) de cada variable.
- Los datos actuales pueden venir de la simulación temporal (fecha de corte ajustable) o de un CSV
  nuevo. Todos los umbrales están en `config.json`.

**Resultado con la fecha de corte 2025-07-01:** estado general **rojo**. Hay drift crítico en
`promedio_ingresos_datacredito` (PSI 0,33) y en `ratio_otros_prestamos_salario` (PSI 0,26), y
moderado en `total_otros_prestamos` y `plazo_meses`. El drift de ambas variables crece mes a mes
y la tasa de impago observada baja de 5,2 % a 3,2 %. La recomendación automática es revisar la
fuente de la central de riesgo y reentrenar con datos recientes.

**Limitación:** el modelo se entrenó con una partición aleatoria de todo el histórico, por lo que
la simulación mide el cambio de la población, no la degradación del modelo en datos futuros no
vistos. En producción, la referencia sería exactamente el periodo de entrenamiento.

## 10. Calidad e integración continua

### Pruebas y estilo

```bash
pytest mlops_pipeline -v --cov=mlops_pipeline --cov-report=xml --junitxml=resultados_pruebas.xml
ruff check mlops_pipeline --line-length 120 --select E,F,W,B,UP --exclude "*.ipynb"
```

37 pruebas con 94 % de cobertura de líneas:

- **Estructura:** cada archivo obligatorio del proyecto existe en su ruta.
- **Unitarias:** limpieza, atributos derivados, preprocesamiento, métricas de evaluación, PSI,
  Jensen-Shannon, tendencias y recomendaciones.
- **Integración:** entrenamiento completo (con modelos rápidos y en una carpeta temporal), trabajo
  de monitoreo y gráficos, aplicación Streamlit (con `streamlit.testing`) y API (JSON, CSV en el
  cuerpo, archivo CSV, misma predicción en ambos formatos y entradas inválidas).

`ruff` revisa las convenciones de estilo (PEP 8) y errores comunes (variables sin usar,
argumentos mutables, anotaciones obsoletas).

### GitHub Actions

`.github/workflows/ci.yml` se ejecuta en cada push o pull request a `developer`, `certification`
y `master`: valida la estructura, instala dependencias, entrena, prueba con cobertura, ejecuta el
trabajo de monitoreo, guarda los artefactos y, si existe el secreto `SONAR_TOKEN`, envía el
análisis a SonarCloud.

### SonarCloud: qué evalúa

| Aspecto | Cómo se cubre |
|---|---|
| Calidad y mantenibilidad | Código duplicado, complejidad ciclomática y cognitiva, funciones largas y malas prácticas (*code smells*) en `mlops_pipeline/*.py`. |
| Seguridad | Vulnerabilidades y *security hotspots*: exposición de datos sensibles, funciones o librerías inseguras. La API limita el tamaño de los archivos, no devuelve trazas internas y el contenedor no corre como root. |
| Cobertura | Se publica `coverage.xml` (líneas ejecutadas) y `resultados_pruebas.xml` (pruebas ejecutadas). |
| Integridad y estilo | Reglas de estilo de Python de SonarCloud, además de `ruff` en el pipeline. |

### SonarCloud

1. Inicie sesión en https://sonarcloud.io con GitHub e importe este repositorio.
2. En el proyecto, desactive el análisis automático (Administration > Analysis Method >
   Automatic Analysis), porque el análisis lo envía el pipeline.
3. Copie la organización y la clave del proyecto en `sonar-project.properties`
   (`sonar.organization` y `sonar.projectKey`) y en los badges de este archivo (`PROJECT_KEY`).
4. Genere un token (My Account > Security) y guárdelo en GitHub como secreto `SONAR_TOKEN`
   (Settings > Secrets and variables > Actions).
5. Haga push: el resultado aparece en el panel de SonarCloud y en los badges.

### Jenkins

1. Levante Jenkins con `docker compose up -d` y abra http://localhost:8080. La contraseña inicial
   se obtiene con `docker logs jenkins`; elija *Install suggested plugins*.
2. Para que Jenkins pueda usar Docker (SonarCloud e imagen de la API): `docker exec -it jenkins bash`,
   luego `apt-get update` y `apt install -y docker.io`; verifique con `docker ps`.
3. (Opcional, para SonarCloud) Manage Jenkins > Credentials: credencial *Secret text* con ID
   `sonarcloud-token` y el token de SonarCloud.
4. New Item > **Multibranch Pipeline** > Branch Sources: Git con la URL de este repositorio y
   Script Path `Jenkinsfile`. Jenkins descubre `developer`, `certification` y `master` y crea un
   pipeline por rama. (Alternativa: un job *Pipeline* con *Pipeline script from SCM* y una rama.)
5. *Scan Multibranch Pipeline Now* (o *Build Now*).

**Pipeline (`Jenkinsfile`):** clonar repositorio, preparar Python, validar la estructura de
carpetas, entorno virtual, estilo (`ruff`), entrenamiento, pruebas (con reporte JUnit), monitoreo
de drift, SonarCloud e imagen Docker. La imagen solo se construye en `certification` y `master`.
Los artefactos (modelo, tablas y cobertura) quedan archivados en cada build.

**Trigger automático:** `pollSCM('H/5 * * * *')`. Jenkins revisa el repositorio cada 5 minutos y
lanza el build de la rama que recibió commits nuevos, por ejemplo al hacer merge de `developer` a
`certification` o de `certification` a `master`. Se usa sondeo porque un Jenkins local no es
accesible desde GitHub para recibir webhooks; con un Jenkins público se puede reemplazar por el
webhook de GitHub (`githubPush()`).
