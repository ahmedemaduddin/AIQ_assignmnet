FROM python:3.9-slim

WORKDIR /app

# Copy application and model files to the container
COPY ./app.py /app
COPY ./process_image.py /app
COPY ./requirements.txt /app
COPY ./coin.onnx /app/coin.onnx  

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
