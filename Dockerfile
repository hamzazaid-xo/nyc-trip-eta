# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# base — Python + uv, nothing project-specific.
# The venv deliberately lives at /opt/venv, OUTSIDE the /app bind mount, so a
# host .venv can never shadow it (and a macOS/arm venv can never leak into Linux).
# ---------------------------------------------------------------------------
FROM python:3.14.7-slim-trixie AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

# libgomp1: LightGBM ships a py3-universal wheel that dynamically links the system
# OpenMP runtime instead of bundling it, and slim images do not include it.
# Without this, `import lightgbm` dies with: libgomp.so.1: cannot open shared object file.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_INSTALLER_METADATA=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src:/app \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app

# ---------------------------------------------------------------------------
# deps — runtime dependencies ONLY (the [project].dependencies table).
# Separate stage so the production image never sees shap/geopandas/jupyter.
# Copying only pyproject.toml + uv.lock means editing source never busts this layer.
# ---------------------------------------------------------------------------
FROM base AS deps
COPY pyproject.toml uv.lock ./
# sharing=locked: the dev and runtime stages build in parallel and share this cache.
# Without it they race for uv's own per-wheel lock and one of them times out.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-install-project --no-default-groups

# ---------------------------------------------------------------------------
# dev — everything: runtime + train + dev groups. Source is bind-mounted at
# run time, so no source is baked in here.
# ---------------------------------------------------------------------------
FROM base AS dev
COPY pyproject.toml uv.lock ./
# sharing=locked: the dev and runtime stages build in parallel and share this cache.
# Without it they race for uv's own per-wheel lock and one of them times out.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-install-project --no-default-groups --group dev --group train
ENV MPLCONFIGDIR=/tmp/mpl
CMD ["bash"]

# ---------------------------------------------------------------------------
# runtime — the Phase 6 serving image. Runtime deps + source, non-root, no mount.
# ---------------------------------------------------------------------------
FROM base AS runtime
COPY --from=deps /opt/venv /opt/venv
COPY src/ ./src/
COPY api/ ./api/
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/models \
 && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
