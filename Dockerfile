FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Limit threads for heavy packages
ENV OPENBLAS_NUM_THREADS=1

COPY requirements.txt .
RUN pip install --no-cache-dir --progress-bar=off -r requirements.txt

COPY app.py .
COPY preprocessing.py .
COPY services.py .
COPY model ./model
COPY templates ./templates
COPY test_json_1.json .

EXPOSE 7860

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:7860", "--access-logfile", "-", "--error-logfile", "-", "--timeout", "120"]

CMD ["app:app"]