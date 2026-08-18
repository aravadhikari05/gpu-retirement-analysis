# Reverted on 2026-08-18 to the pre-2026-08-13 base and install method.
#
# The cu121 index URL is load-bearing, not incidental. PyTorch stopped
# publishing cu121 wheels after 2.5.1, and those wheels still carry sm_61.
# Newer CUDA 12.8 builds dropped Pascal. The GTX 1080 Ti works because of this
# URL, and every verified result in paper/methods-notes.md was measured on an
# image built this way.
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Pinned rather than resolved. Unpinned, these happen to land on 2.5.1 only
# because the cu121 index stops there, which makes a load-bearing version an
# accident of the index contents. 0.20.1 is the last cu121 torchvision built
# for cp310, which is what ubuntu22.04's python3 is.
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# transformers is pinned here rather than installed loose, so the pin that
# produced the verified work_hash actually reaches the image. The previous
# Dockerfile ran `pip install transformers` directly and the pin in
# requirements.txt never applied.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY benchmarks/ /app/benchmarks/
COPY measurement/ /app/measurement/

WORKDIR /app

ENTRYPOINT ["python3", "-m", "measurement.runner"]
