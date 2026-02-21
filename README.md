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
