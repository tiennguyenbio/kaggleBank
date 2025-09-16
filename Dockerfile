# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy files
COPY app.py /app
COPY requirements.txt /app
COPY model /app/model
COPY ms /app/ms

# Install dependencies
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Run with Gunicorn
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
CMD ["app:app"]