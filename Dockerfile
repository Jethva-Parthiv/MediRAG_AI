FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

EXPOSE 8080

CMD ["python3", "app.py"]