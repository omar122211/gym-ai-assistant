# One image for both services (API and chat UI); docker-compose picks the command.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY knowledge ./knowledge
COPY .streamlit ./.streamlit

# Run as a normal user, not root
RUN useradd --create-home --uid 10001 gym && mkdir -p /app/data && chown gym:gym /app/data
USER gym

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
