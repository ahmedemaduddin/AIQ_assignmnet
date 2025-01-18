
FROM python:3.14.0a1-slim

WORKDIR /app

COPY ./app.py /app
COPY ./utils.py /app
COPY ./Challenge2.csv /app

COPY ./requirements.txt /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 6000

CMD ["python", "app.py"]