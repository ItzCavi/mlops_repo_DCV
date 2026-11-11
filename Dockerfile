# Imagen de la API de predicción de impago (mlops_pipeline/model_deploy.py).
#
# Construir (desde la raíz del repositorio):  docker build -t riesgo-impago-api .
# Ejecutar:                                    docker run -p 8000:8000 riesgo-impago-api
# Probar:                                      http://localhost:8000/docs  (o /health)

FROM python:3.13-slim

# Sin archivos .pyc y con logs sin búfer, para ver la salida de uvicorn en tiempo real.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Las dependencias se instalan antes de copiar el código para aprovechar la caché de Docker:
# si solo cambia el código, no se reinstalan.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Se replica la estructura del repositorio: la tabla en la raíz y el código en mlops_pipeline/.
COPY Base_de_datos.csv ./
COPY mlops_pipeline/config.json mlops_pipeline/ft_engineering.py \
     mlops_pipeline/model_training_evaluation.py mlops_pipeline/model_deploy.py ./mlops_pipeline/

# El modelo se entrena dentro de la imagen: así queda serializado con las mismas versiones de
# Python y scikit-learn que lo van a cargar, y se evitan errores de compatibilidad de joblib.
RUN python mlops_pipeline/model_training_evaluation.py

# Se ejecuta con un usuario sin privilegios.
RUN useradd --create-home api && chown -R api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "model_deploy:app", "--app-dir", "mlops_pipeline", "--host", "0.0.0.0", "--port", "8000"]
