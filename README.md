# ecg-ml-stream
Application for Engineering Thesis at WUT

| | |
| --- | --- |
| Unit Tests | ![Coverage](./assets/unit-coverage.svg)

### Build dev image
```
docker build -t ecg-ml-stream-dev -f docker/dev.Dockerfile .
```

### Run dev container
```
docker run -it --rm \
    --name ecg-ml-stream-dev \
    -v ${PWD}:/app \
    ecg-ml-stream-dev /bin/bash
```

---

### Install project
```
pip install .
```

---

### Build and start the application stack
```
docker compose -f docker/docker-compose.yml up -d --build
```

This starts: Kafka, Spark master + 2 workers, ECG consumer, Kafka UI, and Streamlit Dashboard.

| Service | URL |
| --- | --- |
| Kafka UI | http://localhost:8090 |
| Spark master | http://localhost:8080 |
| Spark driver | http://localhost:4040 |
| Streamlit Dashboard | http://localhost:8501 |

### Stop the stack
```
docker compose -f docker/docker-compose.yml down
```

---

### Run producer
```
python -m ecg_ml_stream.producer.ecg_producer \
    --bootstrap-servers localhost:29092 \
    --topic ecg-pending \
    --data-path data/ptb-xl-1.0.3 \
    --num-threads 4 \
    --interval 10.0 \
    --sampling-rate 100
```

---

### Run training
```
python -m ecg_ml_stream.ml.train \
    --data-path data/ptb-xl-1.0.3 \
    --output-dir models \
    --epochs 20 \
    --batch-size 64
```

### Resume training
```
python -m ecg_ml_stream.ml.train \
    --data-path data/ptb-xl-1.0.3 \
    --output-dir models \
    --epochs 50 \
    --batch-size 64 \
    --resume models/run-xyz/checkpoint.pt
```
