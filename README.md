# KineticK9

A full-stack app for controlling a simulated quadruped robot dog. Give it coordinates, a direction, or a voice command and watch it walk — rendered as an MP4 video from a physics simulation powered by MuJoCo MJX and Brax.

The robot is a **Google Barkour vB** quadruped driven by a PPO-trained locomotion policy (JAX/Brax). A React frontend talks to a Flask API that runs the simulation on each command and streams back a video of the walk.

## Features

- **Walk to coordinates** — type (x, y) or click the 2D grid; the dog walks there
- **Directional walking** — walk forward or backward using vector geometry (heading-based waypoint steering)
- **Ball chase** — send the dog toward a red ball in the scene
- **Voice commands** — say "walk to 3, 5", "move forward", or "chase the ball" (ElevenLabs STT)
- **Spoken responses** — the dog announces its arrival via text-to-speech (ElevenLabs TTS)
- **Persistent position** — the robot resumes from where it last stopped across runs
- **Simulation video** — every command produces an MP4 rendered from the MuJoCo scene

## Tech Stack

| Layer | Technology |
|---|---|
| Simulation | MuJoCo MJX · Brax · JAX |
| Policy | PPO (128×4 hidden layers), trained with Brax |
| Robot model | Google Barkour vB (from MuJoCo Menagerie) |
| Backend | Flask · Python 3.10+ |
| Frontend | React 18 · Vite 5 |
| Voice | ElevenLabs (TTS + STT) — optional |
| Video | mediapy (MP4 rendering) |

## Project Structure

```
dawg/
├── run_quadruped.py          # Simulation engine — BarkourEnv, policy inference, video rendering
├── mjx_brax_quadruped_policy # Saved PPO policy parameters
├── robot_state.json          # Persisted robot position (auto-generated)
├── robotdawg.ipynb           # Training/exploration notebook
├── mujoco_menagerie/         # MuJoCo Menagerie robot models (includes google_barkour_vb)
└── DGXDawg-main/             # Full-stack web app
    ├── package.json          # Root scripts (concurrently runs backend + frontend)
    ├── backend/
    │   ├── app.py            # Flask API — walk, direction, ball chase, TTS/STT, video
    │   └── requirements.txt  # Python deps (flask, flask-cors, requests, python-dotenv)
    └── frontend/
        ├── src/
        │   ├── App.jsx       # Main UI — grid, controls, voice, video player, log
        │   └── api.js        # API client functions
        └── package.json      # React + Vite deps
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **JAX** with GPU support recommended (CPU works but is slower)
- **MuJoCo 3.0+**, **mujoco-mjx**, **Brax**
- **ffmpeg** (optional, for video format conversion)

## Setup

### 1. Python environment (simulation + backend)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mujoco mujoco-mjx brax jax mediapy ml_collections etils

cd DGXDawg-main/backend
pip install -r requirements.txt
cd ../..
```

### 2. Frontend

```bash
cd DGXDawg-main/frontend
npm install
cd ..
npm install          # installs concurrently at the root level
```

### 3. (Optional) ElevenLabs voice

Create `DGXDawg-main/backend/.env`:

```
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=optional_voice_id
```

## Running

### Web app (recommended)

From the `DGXDawg-main/` directory:

```bash
npm run dev
```

This starts the Flask API on port **5000** and the Vite dev server on port **5173**. Open **http://localhost:5173**.

### CLI only

Run the simulation directly without the web UI:

```bash
# Walk with specific velocities
python run_quadruped.py --x_vel 1.0 --y_vel 0.0 --ang_vel 0.5

# Walk forward
python run_quadruped.py --walk_direction forward --walk_distance 3.0

# Chase the ball
python run_quadruped.py --move_to_ball --output ball_chase.mp4

# Custom ball position
python run_quadruped.py --move_to_ball --ball_pos 2.0 1.0 0.1 --output chase.mp4
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/dog/state` | Current dog position and status |
| `POST` | `/api/dog/walk` | Walk to `{ x, y }` (updates state) |
| `POST` | `/api/dog/walk/video` | Simulate walk to `{ x, y }`, returns MP4 |
| `POST` | `/api/dog/walk/direction` | Walk `{ direction: "forward"\|"backward" }`, returns MP4 |
| `POST` | `/api/dog/walk/ball` | Chase ball at `{ ball_x, ball_y }`, returns MP4 |
| `POST` | `/api/tts` | Text-to-speech via ElevenLabs |
| `POST` | `/api/stt` | Speech-to-text via ElevenLabs |
| `POST` | `/api/video/convert` | Convert video format (requires ffmpeg) |

## How It Works

1. **Environment** — `BarkourEnv` (in `run_quadruped.py`) wraps the Barkour vB MuJoCo model with a reward structure that tracks velocity commands, penalises falling, and encourages a natural gait.

2. **Policy** — A PPO agent with four 128-unit hidden layers was trained in Brax/MJX. The saved parameters are loaded at inference time; a JIT-compiled inference function maps observations to joint-angle actions at 50 Hz.

3. **Steering** — For coordinate/ball targets a proportional controller computes `[x_vel, y_vel, ang_vel]` commands each step by projecting the world-frame displacement into the robot's local frame. Directional walking places a virtual waypoint along the robot's heading.

4. **Rendering** — After the rollout, MuJoCo renders frames from the `track` camera and mediapy writes them as an MP4.

5. **Persistence** — The robot's final `(x, y, quaternion)` is saved to `robot_state.json` so the next simulation resumes from there.

