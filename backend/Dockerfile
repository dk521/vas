FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SPEECHBRAIN_MODEL_SOURCE=speechbrain/spkrec-ecapa-voxceleb
ENV SPEECHBRAIN_MODEL_DIR=/app/ml_models/ecapa

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake the speaker-recognition model into the image. ECS tasks then start with
# the model already available and do not depend on a Hugging Face download.
RUN python -c "import os; from speechbrain.inference.speaker import SpeakerRecognition; from speechbrain.utils.fetching import LocalStrategy; SpeakerRecognition.from_hparams(source=os.environ['SPEECHBRAIN_MODEL_SOURCE'], savedir=os.environ['SPEECHBRAIN_MODEL_DIR'], local_strategy=LocalStrategy.COPY)" \
    && test -f "$SPEECHBRAIN_MODEL_DIR/hyperparams.yaml"

COPY . .

RUN DJANGO_SECRET_KEY=build-time-only-not-a-production-secret \
    DEBUG=False \
    DB_ENGINE=sqlite \
    USE_S3=False \
    python manage.py collectstatic --noinput

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health/', timeout=3)" || exit 1

# ECAPA is memory-intensive. Each Gunicorn worker has its own Python process
# and model instance, so use one worker in the small Fargate task.
CMD ["gunicorn", "vas.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-"]
