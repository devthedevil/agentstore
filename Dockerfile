# Slim Python image keeps the final image ~150-200 MB.
FROM python:3.11-slim

WORKDIR /app

# Install build deps separately so pip cache invalidates only on requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Render/Railway/Fly all set $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Healthcheck hits FastAPI's /api/health so orchestrators know when we're ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{__import__(\"os\").environ.get(\"PORT\",\"8000\")}/api/health').status==200 else 1)"

CMD uvicorn agentstore.server:app --host 0.0.0.0 --port ${PORT:-8000}
