FROM python:3.12

RUN apt-get update && apt-get install -y vim && apt-get install -y sqlite3

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt