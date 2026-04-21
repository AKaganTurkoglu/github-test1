FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.1.0+cpu torchvision==0.16.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    timm==1.0.3 \
    "fastapi==0.115.0" \
    "uvicorn[standard]==0.30.0" \
    python-multipart==0.0.9 \
    boto3==1.35.0 \
    Pillow==10.4.0 \
    "numpy<2"

# Only copy the code — NO model weights file
# Model gets downloaded from S3 when container starts
COPY app.py .

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]