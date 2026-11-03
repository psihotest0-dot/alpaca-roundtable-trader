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

# Same Poetry version CI uses, so the image resolves poetry.lock the way CI does. The base image
# brings its own setuptools (79.x), which has a known vulnerability fixed in 83.0.0 and is not
# in poetry.lock, so the dependency check never sees it: the floor is set here.
RUN pip install "poetry==1.8.2" "setuptools>=83.0.0"

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

# Run as an ordinary user, not as root: a bug or a bad dependency then cannot write over the
# image, only into /app/var (the database, the log, the heartbeat). A new volume mounted there
# takes this ownership; a volume created while the image still ran as root needs a one-time
# `chown -R 10001:10001` (see docs/operations.md).
RUN groupadd --system --gid 10001 trader \
    && useradd --system --uid 10001 --gid trader --create-home --shell /usr/sbin/nologin trader \
    && chown -R trader:trader /app/var
USER trader

CMD ["python", "-m", "trading_system"]
