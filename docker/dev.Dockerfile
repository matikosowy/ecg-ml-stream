FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip --no-cache-dir install --upgrade pip setuptools wheel

COPY pyproject.toml .
RUN mkdir -p "src/ecg_ml_stream" \
    && pip install --no-cache-dir ".[dev]" \
    && rm -rf "src"


# Final image
FROM python:3.11-slim-bookworm AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java

RUN addgroup --gid 1001 ecg_group \
    && adduser --uid 1001 --gid 1001 ecg_user

WORKDIR /app

COPY --from=builder --chown=ecg_user:ecg_group /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=ecg_user:ecg_group pyproject.toml .
COPY --chown=ecg_user:ecg_group src ./src

RUN pip install --no-cache-dir -e ".[dev]"
USER ecg_user:ecg_group
