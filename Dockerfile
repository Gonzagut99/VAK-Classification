# Usar una imagen base de Python
FROM python:3.12.4-slim

# Instalar git
RUN apt-get update && apt-get install -y git

# Clonar el repositorio de GitHub
RUN git clone https://github.com/Gonzagut99/ModeloSofiaPrediccionNotas.git /usr/ml_project

WORKDIR /usr/ml_project

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

##
# Ejecutar el script de entrenamiento y validación del modelo
RUN python -m app.ml_models.score_prediction_model

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]