#FROM python:3.11-slim as builder
#
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#        gcc \
#        libc-dev \
#        libffi-dev \
#        libssl-dev && \
#    rm -rf /var/lib/apt/lists/*
#
#
#WORKDIR /app
#
#COPY requirements.txt .
#RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r requirements.txt
#
#COPY . .
#
#EXPOSE 8000
#
#CMD [ "python", "api/inference_api.py" ]
