FROM harbor.lolday.svc:80/lolday/pytorch-cu12-base:2.7.0-cu126

# Base ships Python 3.12 + torch 2.7.0 cu126 + numpy/pandas/sklearn/
# pyelftools/structlog/typer + islab-malware-detector. Only the
# detector's own package needs to land on top.
USER root
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps . \
    && rm -rf /root/.cache/pip

USER 1000
ENTRYPOINT ["elfcnndet"]
