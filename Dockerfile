FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import lightgbm; print('lightgbm OK:', lightgbm.__version__)" && \
    python -c "import sklearn; print('sklearn OK:', sklearn.__version__)"

COPY . .

CMD ["python", "quant_engine.py"]
