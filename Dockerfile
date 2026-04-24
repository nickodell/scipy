FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for building SciPy
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash claude
WORKDIR /projects
RUN chown claude:claude /projects

USER claude

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/home/claude/.local/bin:$PATH"

# Clone SciPy and initialize submodules
RUN git clone https://github.com/scipy/scipy.git /projects/scipy
WORKDIR /projects/scipy
RUN git submodule update --init

# Install Python requirements (build + dev + test; skip doc)
RUN pip install --break-system-packages \
    -r requirements/build.txt \
    -r requirements/dev.txt \
    -r requirements/test.txt \
    pytest-json-report

# Build SciPy
RUN spin build

# Copy in workflow scripts
COPY --chown=claude:claude establish_baseline.sh check_slow_coverage.sh parse_test_times.py agent_prompt.md ./

WORKDIR /projects/scipy
CMD ["bash"]
