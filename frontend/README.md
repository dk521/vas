# Voice Access frontend

## Run locally

1. Start Django from the repository root: `python manage.py runserver`
2. In this directory, install dependencies: `npm install`
3. Start the client: `npm run dev`

The development server proxies `/api` requests to `http://127.0.0.1:8000`; no Django CORS changes are needed for this local setup.

After sign-in, the UI calls `GET /api/voice/enrollment-status/` to show a returning volunteer's current progress. It displays the completion screen immediately for anyone with all 10 samples.

To point a deployed client at a different API, create `.env` with `VITE_API_URL=https://your-api.example.com` and configure CORS on the Django server.

The backend accepts WAV, M4A, MP3, and WebM for enrollment. FFmpeg must be installed on the Django host because the existing API converts each upload to WAV for processing.
