# DGXDawg

**Walk a simulated robot dog to coordinates.** Full-stack app: **Flask** (Python) API + **React** (Vite) frontend. You input (x, y) and the dog walks there; walking is the main functionality.

## Quick start

**One terminal (from project root):**

```bash
npm install
npm run dev
```

This starts both the Flask API (port 5000) and the Vite dev server (port 5173). Open **http://localhost:5173**.

---

**First-time setup:** create the backend venv and install deps once:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

Then from the project root, `npm run dev` runs backend + frontend together.

**Optional — split terminals:** run `npm run dev:backend` in one terminal and `npm run dev:frontend` in another.

## What’s included

- **Backend** (`backend/`): Flask API that keeps the dog’s (x, y) position. `POST /api/dog/walk` with `{ x, y }` moves the dog to that coordinate (instant for now; replace with real sim/stepping when you integrate).
- **Frontend** (`frontend/`): React UI with a 2D grid you can click to set target coordinates, X/Y inputs, a Walk button, and a log of walk commands and arrivals. Optional **ElevenLabs** voice: tick “Speak dog responses” or click the speaker on a log line to hear the text spoken.

## Scripts

| Command           | Where   | Description                    |
|-------------------|---------|--------------------------------|
| `npm run dev`     | root    | Run API + frontend (one terminal) |
| `npm run dev:backend`  | root | Run Flask API only (port 5000) |
| `npm run dev:frontend` | root | Run Vite dev server only (port 5173) |
| `npm run build`   | frontend| Production build               |
| `npm run preview` | frontend| Preview production build       |

## API

- `GET /api/health` — Health check
- `GET /api/dog/state` — Current dog state: `{ x, y, status }` (status: `idle` or `walking`)
- `POST /api/dog/walk` — Command the dog to walk to coordinates. Body: `{ "x": 3, "y": -2 }`. Response: `{ x, y, status }` (position after the move)
- `POST /api/tts` — **ElevenLabs text-to-speech.** Body: `{ "text": "Hello" }`. Returns `audio/mpeg`. Optional body: `{ "voice_id": "..." }` (defaults to env or built-in voice).
- `POST /api/dog/walk/video` — **Video (robot sim).** Body: `{ "x", "y", "from_x"?, "from_y"? }`. Stub for now. **Final videos will be from the robotic simulation**, not the 2D grid. Integrate your sim in this endpoint (run sim for the path, capture video, return URL). The app’s “Record placeholder video” option records the grid/dot for testing only.
- `POST /api/video/convert` — **Convert video format.** Form: `file` = video file, query `format=mp4` or `webm`. Returns the converted file. Requires **ffmpeg** on the server for MP4 output; if missing, the app still offers WebM download.

## Optional: ElevenLabs voice

To enable spoken “dog” responses (e.g. “Arrived at (3, -2)”):

1. Get an API key from [ElevenLabs](https://elevenlabs.io/) (Profile → API Key).
2. In the `backend/` folder, create a file named `.env` or `background.env` and add:
   ```
   ELEVENLABS_API_KEY=your_api_key_here
   ELEVENLABS_VOICE_ID=optional_voice_id
   ```
   (Copy from `backend/.env.example` and fill in. Voice ID is optional; default voice is used if unset.)
3. Restart the Flask backend so it picks up the env var. The frontend will use the “Speak dog responses” checkbox and the speaker button on log lines.
