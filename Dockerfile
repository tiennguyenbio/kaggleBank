FROM python:3.10-slim

WORKDIR /app

# Limit threads for heavy packages
ENV OPENBLAS_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt

COPY app.py .
COPY ms ./ms
COPY model ./model
COPY preprocessing.py .

EXPOSE 8000

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8000", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]
CMD ["app:app"]