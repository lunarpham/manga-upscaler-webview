FROM python:3.9-slim

# Install curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create models directory and download the model
RUN mkdir -p models && \
    curl -L https://github.com/lunarpham/MangaUpscalingModels/releases/download/0.2.1/2xLiloScale_80K.onnx -o models/2xLiloScale_80K.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create uploads and outputs directories
RUN mkdir -p uploads outputs

EXPOSE 5000

CMD ["python", "app.py"]