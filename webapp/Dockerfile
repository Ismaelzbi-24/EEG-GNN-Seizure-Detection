# Hugging Face Spaces (Docker SDK) or any Docker host.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user api ./api
COPY --chown=user site ./site
COPY --chown=user models ./models

USER user
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp
EXPOSE 7860
WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
