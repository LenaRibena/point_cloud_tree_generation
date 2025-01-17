FROM python:3.11-slim

WORKDIR /trees

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN --mount=type=cache,target=~/pip/.cache/pip pip install -r /trees/requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# Open terminal to develop
ENTRYPOINT ["/bin/bash"]