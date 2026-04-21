FROM harbor.lolday.svc:80/lolday/pytorch-cu12-base:2.7.0-cu126

# The base image already ships torch 2.7.0 + CUDA 12.6 + cuDNN 9 + all
# transitive nvidia-cu12 wheels. We only need to install the detector's
# own dependencies on top — pyproject.toml's torch constraint is
# satisfied by the base image, so --no-deps on the editable install is
# safe (pip resolves existing packages from the base env first anyway).
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

# Install detector's deps excluding torch (already present in base).
# uv is installed in the base image via conda; use it for fast installs.
RUN pip install --no-cache-dir \
        "islab-malware-detector[mlflow]>=0.5.0" \
        "numpy>=1.24.0" \
        "pandas>=2.0.0" \
        "scikit-learn>=1.3.0" \
        "pyelftools>=0.30" \
        "structlog>=24.0.0" \
        "typer>=0.9.0" \
    && pip install --no-cache-dir --no-deps .

USER 1000
ENTRYPOINT ["elfcnndet"]
