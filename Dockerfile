# syntax=docker/dockerfile:1.7
# Cohort Monitoring Platform - production image (uv-managed, locked dependencies, non-root runtime).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# libgomp1: OpenMP runtime required by LightGBM; curl: container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

WORKDIR /app

# Set to 1 to include the optional proprietary yFiles graph engine (it loads its runtime from
# yworks.com at render time and reports anonymised usage data; see documentation/developer/Graph_Rendering.md).
ARG WITH_YFILES=0

# Dependencies first so the layer is cached across source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_YFILES" = "1" ]; then \
        uv sync --frozen --no-install-project --extra survival --extra graph-yfiles; \
    else \
        uv sync --frozen --no-install-project --extra survival; \
    fi

COPY . .

# Run as an unprivileged user; /app/data is the mount point for DATA_ROOT (per-user workspaces).
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    DATA_ROOT=/app/data \
    CMP_GRAPH_RENDERER=plotly \
    CMP_IAM_PROVIDER=local \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
