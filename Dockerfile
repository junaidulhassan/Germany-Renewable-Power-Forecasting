# GPU image for training. Not built or run during development; treat as a starting point.
#   docker build -t renewcast .
#   docker run --gpus all -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
#       renewcast run-all
FROM tensorflow/tensorflow:2.17.0-gpu

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# TensorFlow (and its pinned NumPy) already ship in the base image.
RUN pip install --no-cache-dir .

COPY configs ./configs
ENV MPLBACKEND=Agg
ENTRYPOINT ["renewcast"]
CMD ["--help"]
