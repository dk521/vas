# Voice Access System

## Run locally with Docker

This is the quickest way to start both the Django API and the React client.

```bash
docker compose up --build
```

Open `http://localhost:5173`. The API is available at `http://localhost:8000`.
The Docker setup uses SQLite and local `backend/media/` storage; it does not use
the configured remote database or S3 credentials. Stop it with `Ctrl+C`.

## Run locally without Docker

1. Copy `backend/.env.example` to `backend/.env` and keep `DB_ENGINE=sqlite`
   and `USE_S3=False`.
2. Install FFmpeg (for example, `brew install ffmpeg` on macOS).
3. Create/activate a Python virtual environment, then install backend packages:

   ```bash
   cd backend
   python -m pip install -r requirements.txt
   python manage.py migrate
   python manage.py seed_pangrams
   python manage.py runserver
   ```

4. In a second terminal, start the frontend:

   ```bash
   cd frontend
   npm ci
   npm run dev
   ```

Open `http://localhost:5173`. The frontend proxies `/api` requests to Django.

For a deployment using PostgreSQL or S3, set `DB_ENGINE=postgres` and/or
`USE_S3=True` with the matching credentials in `backend/.env`.

## Container health check

The backend exposes `GET /health/` for Docker and AWS load-balancer health
checks. A production image starts Gunicorn; the Compose configuration keeps
using Django's development server for local hot reload.

## Production environment

For AWS, set `DEBUG=False`, a long random `DJANGO_SECRET_KEY`, the public API
hostname in `ALLOWED_HOSTS`, and the frontend HTTPS origin in
`CSRF_TRUSTED_ORIGINS` and `CORS_ALLOWED_ORIGINS`. Django then trusts the load balancer's
`X-Forwarded-Proto` header, enforces HTTPS for application routes, enables
secure cookies and HSTS, and leaves `/health/` available for health checks.

The backend uses WhiteNoise to serve its collected static assets (such as the
Django admin). Build the React app separately with `npm run build` and host
`frontend/dist/` on S3 + CloudFront. If CloudFront routes `/api/*` to the API,
the frontend can use its default relative API URL; otherwise set
`VITE_API_URL=https://api.example.com` during the frontend build.

The ECAPA speaker-recognition model is downloaded while the backend image is
built and stored at `/app/ml_models/ecapa`. It is therefore available to every
ECS task at startup without a first-request model download.
