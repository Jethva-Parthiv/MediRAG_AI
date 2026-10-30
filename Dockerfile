FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "--keep-alive", "5", "app:app"]