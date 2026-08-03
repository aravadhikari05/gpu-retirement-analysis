FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers pynvml pandas

COPY benchmarks/ /app/benchmarks/
COPY measurement/ /app/measurement/

WORKDIR /app

ENTRYPOINT ["python3", "-m", "measurement.runner"]
