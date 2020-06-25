# Build an image that can do training and inference in SageMaker
# This is a Python image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# Depending on the chosen framework, update the base image here.
# The Docker transformer requires that all base images (those following FROM in your Dockerfile) are sourced from ECR.
# This is a strict security requirement, and we cannot grant exceptions to this.
# https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.2.0-gpu-py37-cu102-ubuntu18.04

MAINTAINER AWS


# update here to use apt instead of yum when using ubuntu base image.
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
         unzip \
         ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY requirements.txt /opt/program/requirements.txt
# COPY config /opt/program
# COPY /home/ubuntu/dev/AwsGeminiScienceAutoregressive/requirements.txt /opt/program

RUN mkdir -p /opt/ml/io/
RUN mkdir -p /opt/ml/model/
RUN mkdir -p /opt/ml/output/

# Here we get all python packages.
RUN pip install -r /opt/program/requirements.txt

WORKDIR /opt/program


