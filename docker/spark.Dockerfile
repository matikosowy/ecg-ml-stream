FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip --no-cache-dir install --upgrade pip setuptools wheel

COPY pyproject.toml .
RUN mkdir -p "src/ecg_ml_stream" \
    && pip install --no-cache-dir ".[spark]" \
    && rm -rf "src"


FROM python:3.11-slim-bookworm AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
        default-jdk-headless \
        wget \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV SPARK_HOME=/opt/venv/lib/python3.11/site-packages/pyspark
ENV JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=jdk.unsupported/sun.misc=ALL-UNNAMED"

RUN addgroup --gid 1001 ecg_group \
    && adduser --uid 1001 --gid 1001 ecg_user

WORKDIR /app

COPY --from=builder --chown=ecg_user:ecg_group /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

ENV PYSPARK_PYTHON=/opt/venv/bin/python

COPY --chown=ecg_user:ecg_group pyproject.toml .
COPY --chown=ecg_user:ecg_group src ./src

RUN pip install --no-cache-dir -e ".[spark]"

ARG MAVEN=https://repo1.maven.org/maven2
ARG SCALA=2.12
ARG SPARK_VER=3.5.3
ARG KAFKA_CLIENTS_VER=3.4.1
ARG COMMONS_POOL2_VER=2.11.1

RUN wget -q -P "${SPARK_HOME}/jars" \
        "${MAVEN}/org/apache/spark/spark-sql-kafka-0-10_${SCALA}/${SPARK_VER}/spark-sql-kafka-0-10_${SCALA}-${SPARK_VER}.jar" \
        "${MAVEN}/org/apache/spark/spark-token-provider-kafka-0-10_${SCALA}/${SPARK_VER}/spark-token-provider-kafka-0-10_${SCALA}-${SPARK_VER}.jar" \
        "${MAVEN}/org/apache/kafka/kafka-clients/${KAFKA_CLIENTS_VER}/kafka-clients-${KAFKA_CLIENTS_VER}.jar" \
        "${MAVEN}/org/apache/commons/commons-pool2/${COMMONS_POOL2_VER}/commons-pool2-${COMMONS_POOL2_VER}.jar" \
    && chown ecg_user:ecg_group \
        "${SPARK_HOME}/jars/spark-sql-kafka-0-10_${SCALA}-${SPARK_VER}.jar" \
        "${SPARK_HOME}/jars/spark-token-provider-kafka-0-10_${SCALA}-${SPARK_VER}.jar" \
        "${SPARK_HOME}/jars/kafka-clients-${KAFKA_CLIENTS_VER}.jar" \
        "${SPARK_HOME}/jars/commons-pool2-${COMMONS_POOL2_VER}.jar"

RUN mkdir -p "${SPARK_HOME}/conf" /app/checkpoints /app/logs \
    && echo "spark.master spark://spark-master:7077" > "${SPARK_HOME}/conf/spark-defaults.conf" \
    && chown ecg_user:ecg_group /app /app/checkpoints /app/logs

USER ecg_user:ecg_group
