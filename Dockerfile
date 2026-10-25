# ── Stage 1: Build the React frontend ────────────────────────────
FROM node:22-alpine AS frontend

WORKDIR /build
COPY src/web/package.json src/web/package-lock.json ./
RUN npm ci --ignore-scripts

COPY src/web/ ./
RUN npm run build


# ── Stage 2: Python runtime ─────────────────────────────────────
FROM python:3.13-slim AS runtime

# System dependencies for numpy, faiss-cpu, and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

# Copy the pre-built frontend into the expected location
COPY --from=frontend /build/dist/ src/web/dist/

# Default graph storage location (mount a volume here)
RUN mkdir -p /app/data
ENV ENGRAM_GRAPH=/app/data/engram_graph.db

# Default to connecting to the Ollama sidecar container
ENV OLLAMA_HOST=http://ollama:11434

EXPOSE 8000

# Launch the API server (serves both REST API and web UI)
CMD ["engram-server"]
