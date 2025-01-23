FROM python:3.11-slim

WORKDIR /trees

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy relevant files
COPY requirements_dev.txt /trees/requirements.txt
COPY pyproject.toml /trees/pyproject.toml
COPY src/tree /trees/src/tree
COPY configs /trees/configs

# Install dependencie(s)
RUN pip install -r /trees/requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Open terminal to develop
ENTRYPOINT ["/bin/bash"]
