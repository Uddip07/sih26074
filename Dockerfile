# Gram-Vani: forecast API + dashboard
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md LICENSE ./
COPY config ./config
COPY src ./src
RUN pip install --no-deps -e .
# Pipeline outputs are mounted (or baked) at run time:
#   docker run -p 8080:8080 -v $PWD/data:/app/data -v $PWD/outputs:/app/outputs agromet
VOLUME ["/app/data", "/app/outputs"]
EXPOSE 8080
HEALTHCHECK --interval=60s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/api/health')"
RUN useradd -m app && chown -R app /app
USER app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
