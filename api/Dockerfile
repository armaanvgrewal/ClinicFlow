FROM python:3.11-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (FIX: add trailing slash)
COPY *.pkl ./

# Copy app
COPY app.py .

# Expose port
EXPOSE 8080

# Run
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 120 app:app
