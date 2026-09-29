# Gram-Vani: forecast API + dashboard
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends libexpat1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY config ./config
COPY src ./src

RUN pip install --no-deps -e .

# Create directories required by the application
RUN mkdir -p /app/data /app/outputs

# Create non-root user and give it ownership of the application
RUN useradd -m app && \
    chown -R app:app /app

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health')"

USER app

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]