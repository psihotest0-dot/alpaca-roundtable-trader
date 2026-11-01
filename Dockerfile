# Python 3.11 is the version the project is developed and tested on (pyproject
# `python = "^3.11"`, CI, the local venv). Do not move the tag to a newer minor
# on its own: the locked numpy 1.26 has no wheels past Python 3.12, onnxruntime
# none past 3.14, and pandas none for 3.15, so the dependencies would not install.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Same Poetry version CI uses, so the image resolves poetry.lock the way CI does.
RUN pip install "poetry==1.8.2"

# Dependencies first: this layer is rebuilt only when pyproject.toml or poetry.lock change.
COPY pyproject.toml poetry.lock ./

# Runtime dependencies only. The dev group (pytest, ruff, mypy, safety) stays out of the image.
RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-root --no-interaction --no-ansi

# Application source. .dockerignore keeps .env, var/, venv/ and caches out of the build context.
COPY . .

# Install the project itself (src/trading_system) without touching the dependencies again.
RUN poetry install --only-root --no-interaction --no-ansi

# var/ holds the SQLite database, logs and the local vector store. It is git-ignored and
# excluded from the build context, so create it; `alembic upgrade head` needs it to exist.
RUN mkdir -p /app/var

CMD ["python", "-m", "trading_system"]
