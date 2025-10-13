FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY predict.py .
COPY demo.py .

# Expose Gradio default port
EXPOSE 7860

# Run Gradio web app (downloads model from HuggingFace automatically)
CMD ["python", "app.py"]