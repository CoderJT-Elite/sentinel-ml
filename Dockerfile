FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 SENTINEL_HOME=/app GIT_PYTHON_REFRESH=quiet
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e . --no-deps

# Run as an unprivileged user. UID 1000 matches the usual first user on a Linux host, so the bind-mounted ./data stays writable.
RUN groupadd -g 1000 sentinel && useradd -u 1000 -g sentinel -m -s /bin/bash sentinel \
    && mkdir -p /app/runs /app/data /mlartifacts \
    && chown -R sentinel:sentinel /app /mlartifacts

EXPOSE 8000
ENV PORT=8000

USER sentinel
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
