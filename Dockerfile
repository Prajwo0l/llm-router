FROM python:3.11-slim

WORKDIR /app

# Core deps + the semantic cache extra (faiss + numpy only, no torch --
# see requirements-cache.txt) -- caching is a headline feature of this
# project, so it's part of the default image. The classifier (torch/
# transformers), local-embedding backend (also torch, via
# sentence-transformers), tracing, and benchmark extras are NOT installed
# here: they're either a Colab-only training concern or an opt-in choice
# that pulls in a real local ML install, and skipping them by default
# keeps the image small and the build fast.
COPY requirements.txt requirements-cache.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-cache.txt

COPY app/ ./app/

# Mounted at runtime via docker-compose (see its volumes:), so a trained
# checkpoint copied in after the fact is picked up without a rebuild.
RUN mkdir -p /app/training/checkpoints /app/data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
