FROM python:3.10.12

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y cmake

RUN pip install -r requirements.txt