# Use the official nvidia CUDA image with Python 3.8 and CUDA 11.3
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Add maintainer information
LABEL maintainer="Christoph Werner <christoph.werner@hs-wismar.de>"
LABEL version="1.0"
LABEL description="Image to run scripts from GitHub Repo 'S3BERT: Semantically Structured Sentence Embeddings' easily with all dependencies"
LABEL build_date="2024-10-16"
LABEL org.opencontainers.image.source="https://github.com/chr-werner/S3BERT"
LABEL org.opencontainers.image.documentation="https://github.com/chr-werner/S3BERT/README.md"

# Accept a build argument for the city and time zone
ARG CITY="UTC"
ARG TIMEZONE="UTC"

# Example of how to use these variables (set environment variables or configure the system)
ENV CITY=$CITY
ENV TIMEZONE=$TIMEZONE

# You can also use this argument to configure system settings, like setting the time zone.
RUN ln -sf /usr/share/zoneinfo/$TIMEZONE /etc/localtime && \
    echo $TIMEZONE > /etc/timezone

# Set a working directory
WORKDIR /app
# Install system dependencies, including those required for pyenv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev \   
    git \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.8
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-venv \
    python3.8-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl -L -o get-pip.py https://bootstrap.pypa.io/get-pip.py && \
   python3.8 get-pip.py && \
   rm get-pip.py

# Install pypi packages
RUN pip install --no-cache-dir \
    torch==1.11.0+cu113 \
    transformers==4.16.1 \
    sentence-transformers==2.1.0 \
    numpy==1.21.2 \ 
    scipy==1.7.3 \ 
    huggingface-hub==0.10.0 \ 
    --extra-index-url https://download.pytorch.org/whl/cu113

# Copy all files from the current directory to the container
COPY . .

# Download and extract the dataset
RUN curl -L -o amr_data_set.tar.gz https://cl.uni-heidelberg.de/~opitz/data/amr_data_set.tar.gz && \
    tar -xvzf amr_data_set.tar.gz && \
    rm amr_data_set.tar.gz