# A100 / dedicated-GPU Hugging Face Space (Docker SDK).
# Models load once at boot and stay warm; pause the Space to stop billing.

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYOPENGL_PLATFORM=egl \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    SAM3D_DIR=/app/sam-3d-body \
    HF_HOME=/app/.cache/huggingface

# System libs for OpenCV / pyrender / video decoding, plus a C++ toolchain.
# build-essential is required because the runtime base ships no compiler and
# detectron2 (a1ce2f9) builds its ops from source against the installed torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 libosmesa6 libegl1 \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the model repo (added to sys.path by pipeline/gpu.py via SAM3D_DIR)
RUN git clone https://github.com/facebookresearch/sam-3d-body.git /app/sam-3d-body

# Python deps. detectron2 is pinned to a1ce2f9 and built --no-build-isolation
# --no-deps so it compiles against the torch already in the base image — the
# single most fragile step; the pinned base keeps it reproducible.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir \
       "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps \
    && pip install --no-cache-dir "git+https://github.com/microsoft/MoGe.git"

# Runtime cache redirection. HF Spaces runs as a non-root user with HOME=/, so
# anything writing to ~/.cache (torch.hub clones DINOv3 here at model load,
# matplotlib, fontconfig, ...) hits a read-only /.cache. These are only needed at
# runtime, so they sit BELOW the pip layer to keep that expensive layer cached.
ENV HOME=/app \
    XDG_CACHE_HOME=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    MPLCONFIGDIR=/app/.cache/matplotlib

# App (UI + the pipeline package; GPU code is isolated in pipeline/gpu.py)
COPY app.py /app/app.py
COPY pipeline /app/pipeline

# Writable cache + home (HF Spaces runs as a non-root user). torch.hub needs to
# clone into TORCH_HOME at runtime, so the whole /app tree must be writable.
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /app/.cache/matplotlib \
    && chmod -R 777 /app

EXPOSE 7860
CMD ["python", "app.py"]
