FROM python:3.12-slim

# libgomp1 is pulled transitively by torch wheels when running on CPU;
# preinstalling it avoids a runtime import error on minimal base images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

# Install torch from the CUDA 12.1 wheel index first so the GPU stack is
# resolved cleanly before the rest of the deps.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir .

USER 1000
ENTRYPOINT ["elfcnndet"]
