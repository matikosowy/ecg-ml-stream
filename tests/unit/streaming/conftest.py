"""Fixtures for streaming unit tests in ECG-ML-STREAM.

Copyright 2026 Mateusz Golebiewski
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_query() -> MagicMock:
    """Return a mock Spark streaming query."""
    return MagicMock()


@pytest.fixture
def mock_df(mock_query: MagicMock) -> MagicMock:
    """Return a self-chaining mock DataFrame with a write stream ending in mock_query.

    All DataFrame transformation methods (withColumn, withColumnRenamed, select)
    return the same mock, making the chain predictable regardless of depth.
    The writeStream chain routes start() to mock_query.
    """
    write_chain = MagicMock()
    write_chain.format.return_value = write_chain
    write_chain.foreachBatch.return_value = write_chain
    write_chain.option.return_value = write_chain
    write_chain.outputMode.return_value = write_chain
    write_chain.trigger.return_value = write_chain
    write_chain.start.return_value = mock_query

    df = MagicMock()
    df.withColumn.return_value = df
    df.withColumnRenamed.return_value = df
    df.select.return_value = df
    df.writeStream = write_chain

    return df


@pytest.fixture
def mock_spark(mock_df: MagicMock) -> MagicMock:
    """Return a mock SparkSession whose readStream chain ends in mock_df.

    All readStream option calls are recorded on mock_spark.readStream.option
    regardless of chain depth, because the read chain returns itself on every call.
    """
    read_chain = MagicMock()
    read_chain.format.return_value = read_chain
    read_chain.option.return_value = read_chain
    read_chain.load.return_value = mock_df

    spark = MagicMock()
    spark.readStream = read_chain

    return spark


@pytest.fixture
def mock_spark_builder(mock_spark: MagicMock) -> MagicMock:
    """Return a mock SparkSession builder that chains config calls and yields mock_spark."""
    builder = MagicMock()
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value.__enter__.return_value = mock_spark
    builder.getOrCreate.return_value.__exit__.return_value = False

    return builder
