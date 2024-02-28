FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY . /app 
COPY . .


ENV GOOGLE_APPLICATION_CREDENTIALS=keys.json
ENV GOOGLE_CLOUD_PROJECT=crp-dev-dig-mlcatalog

# Instala las dependencias
RUN pip install -r requirements.txt
COPY . .

# Expone el puerto (Cloud Run usará la variable $PORT)
EXPOSE 8000

# Inicia la aplicación
CMD ["python", "validacionEndpoint.py"]