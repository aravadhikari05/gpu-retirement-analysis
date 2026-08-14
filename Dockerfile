FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
RUN pip install --no-cache-dir nvidia-ml-py
WORKDIR /app
COPY matmul_benchmark.py power_monitor.py /app/
ENTRYPOINT ["python", "matmul_benchmark.py"]
