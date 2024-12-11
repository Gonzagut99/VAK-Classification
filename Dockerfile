# Usar una imagen base de Python
FROM python:3.12.4-slim

# Instalar git y limpiar caché de apt
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/

# Clonar el repositorio de GitHub
RUN git clone https://github.com/Gonzagut99/VAK-Classification.git /usr/dl_project

# Actualizar pip
RUN pip install --upgrade pip

WORKDIR /usr/dl_project

# Modificar requirements.txt para usar una versión compatible de TensorFlow
# RUN sed -i 's/tensorflow==2.18.0/tensorflow==2.14.0/g' requirements.txt && \
#     sed -i 's/tensorflow_intel==2.18.0/tensorflow==2.14.0/g' requirements.txt
# RUN sed -i 's/tensorflow_intel==2.18.0/tensorflow_intel==2.14.0/g' requirements.txt

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-2.18.0-py3-none-any.whl

##
# # Ejecutar el script de entrenamiento y validación del modelo
# RUN python -m app.ml_models.score_prediction_model

# Exponer el puerto en el que se ejecutará la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]