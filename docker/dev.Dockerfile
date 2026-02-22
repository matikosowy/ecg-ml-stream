FROM python:3.11-slim AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip --no-cache-dir install --upgrade pip setuptools wheel

COPY pyproject.toml .
RUN mkdir -p "src/ecg" \
    && pip install --no-cache-dir ".[dev]" \
    && rm -rf "src"

COPY src ./src


# Final image
FROM python:3.11-slim AS final

RUN addgroup --gid 1001 ecg_group \
    && adduser --uid 1001 --gid 1001 ecg_user

WORKDIR /app

COPY --from=builder --chown=ecg_user:ecg_group /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=ecg_user:ecg_group pyproject.toml .
COPY --chown=ecg_user:ecg_group src ./src

RUN pip install --no-cache-dir -e ".[dev]"
USER ecg_user:ecg_group
