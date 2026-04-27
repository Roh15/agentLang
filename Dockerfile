FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Install uv
RUN pip install uv

WORKDIR /workspace

# Copy package definition first for layer caching
COPY pyproject.toml .

# Install HF deps on top of base image's PyTorch.
# --system: install into system Python (not a venv)
# --no-build-isolation: lets the build see system torch so we don't pull a second one
RUN uv pip install --system --no-build-isolation \
    "transformers>=4.37.0" \
    "accelerate>=0.26.0" \
    "pytest>=7.4.0" \
    "matplotlib>=3.7.0" \
    "numpy>=1.24.0"

# Copy source
COPY . .

# Model weights are bind-mounted at runtime — never baked into the image
VOLUME ["/models"]

CMD ["python3", "benchmark.py"]
