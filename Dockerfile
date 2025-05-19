FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Start app using Gunicorn (production WSGI server)
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "app:app"]
