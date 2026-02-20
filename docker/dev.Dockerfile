FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip --no-cache-dir install --upgrade pip setuptools wheel

COPY pyproject.toml .
RUN mkdir -p "src/ecg" \
    && pip install --no-cache-dir "." \
    && rm -rf "src"

COPY src ./src


# Final image
FROM python:3.11-slim AS final

RUN addgroup --gid 1001 ecg \
    && adduser --uid 1001 --gid 1001 ecg

WORKDIR /app

COPY --from=builder --chown=ecg:ecg /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=ecg:ecg pyproject.toml .
COPY --chown=ecg:ecg src ./src

RUN pip install --no-cache-dir -e "."
USER ecg:ecg
