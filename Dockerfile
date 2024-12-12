FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    tk-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz \
    && tar xzf Python-3.12.7.tgz \
    && cd Python-3.12.7 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.12.7 Python-3.12.7.tgz

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.12 1

RUN pip3.12 install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .

RUN pip3.12 install --no-cache-dir -r requirements.txt

# Login to Hugging Face

COPY . .

EXPOSE 5000

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "5000"]