FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Directorio donde estará FastAPI
WORKDIR /app

# Dependencias del sistema necesarias para algunas librerías
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements
COPY requirements.txt /app/requirements.txt

# Instalamos dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Recursos NLTK necesarios para el NLP
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copiamos la aplicación
COPY aplicacion/main.py /app/main.py
COPY aplicacion/index.html /app/index.html

# Copiamos los modelos
COPY models/ /models/

# FastAPI
EXPOSE 8001

# Arrancamos Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]