#!/bin/bash -e

mkdir -p assets

coverage run --source="src/ecg_ml_stream" -m pytest "./tests"
coverage report -m
coverage xml -o coverage.xml
genbadge coverage -i coverage.xml -o "${PWD}/assets/unit-coverage.svg"
