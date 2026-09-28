FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1

# Definimos el directorio de trabajo primero
WORKDIR /app

# Copiamos el requirements asumiendo que está dentro de app/
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código y los modelos a la raíz del contenedor
COPY app/ /app/
COPY models/ /models/

EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]