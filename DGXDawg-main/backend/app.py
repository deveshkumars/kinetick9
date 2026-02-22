"""
Flask API for KineticK9 — simulated robot dog driven by target coordinates.
Walking to (x, y) is the main functionality. Optional ElevenLabs TTS for voice.
"""
import os
import sys
import shutil
import subprocess
import tempfile
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS

# Add the dawg project root (two levels up) so we can import run_quadruped.py
_backend_dir_for_path = os.path.dirname(os.path.abspath(__file__))
_dawg_root = os.path.dirname(os.path.dirname(_backend_dir_for_path))
sys.path.insert(0, _dawg_root)
from run_quadruped import run_quadruped

# Load env: backend folder first, then project root (so .env in either place works)
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_backend_dir)
load_dotenv(os.path.join(_backend_dir, ".env"))
load_dotenv(os.path.join(_backend_dir, "background.env"))
load_dotenv(os.path.join(_project_root, ".env"))

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# ElevenLabs: set ELEVENLABS_API_KEY in env; optional ELEVENLABS_VOICE_ID (default voice if unset)
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

# Simulated robot dog state: position and walk target
dog_state = {
    "x": 0.0,
    "y": 0.0,
    "status": "idle",  # idle | walking
    "target_x": None,
    "target_y": None,
}


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "KineticK9 API"})


@app.route("/api/dog/state")
def get_state():
    return jsonify(dog_state)


@app.route("/api/dog/walk", methods=["POST"])
def walk():
    """Command the dog to walk to coordinates (x, y)."""
    data = request.get_json() or {}
    try:
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "x and y must be numbers"}), 400

    # For now: instant move to target (replace with real sim/stepping later)
    dog_state["target_x"] = x
    dog_state["target_y"] = y
    dog_state["status"] = "walking"
    dog_state["x"] = x
    dog_state["y"] = y
    dog_state["status"] = "idle"
    dog_state["target_x"] = None
    dog_state["target_y"] = None

    return jsonify({
        "x": dog_state["x"],
        "y": dog_state["y"],
        "status": dog_state["status"],
    })


@app.route("/api/tts", methods=["POST"])
def tts():
    """Convert text to speech via ElevenLabs. Requires ELEVENLABS_API_KEY in env."""
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ElevenLabs not configured: set ELEVENLABS_API_KEY"}), 503
    data = request.get_json() or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing or empty 'text'"}), 400
    voice_id = data.get("voice_id") or ELEVENLABS_VOICE_ID
    url = ELEVENLABS_URL.format(voice_id=voice_id)
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    payload = {
        "text": text[:5000],  # API limit
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return Response(r.content, mimetype="audio/mpeg")
    except requests.exceptions.RequestException as e:
        status = getattr(e.response, "status_code", 500) if hasattr(e, "response") else 500
        body = {}
        if hasattr(e, "response") and e.response is not None and e.response.text:
            try:
                body = e.response.json()
            except Exception:
                body = {"detail": e.response.text[:200]}
        return jsonify({"error": "ElevenLabs TTS failed", "detail": body.get("detail", str(e))}), min(status, 502)


@app.route("/api/stt", methods=["POST"])
def stt():
    """Transcribe audio via ElevenLabs Speech-to-Text. Form: file=audio file. Returns { text }."""
    if not ELEVENLABS_API_KEY:
        return jsonify({"error": "ElevenLabs not configured: set ELEVENLABS_API_KEY"}), 503
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No audio file uploaded"}), 400
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    # Read file so we can send it (Flask file stream may not be rewindable)
    file_content = file.read()
    files = {"file": (file.filename or "audio.webm", file_content, file.content_type or "audio/webm")}
    data = {"model_id": "scribe_v2"}
    try:
        r = requests.post(ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=60)
        r.raise_for_status()
        out = r.json()
        text = out.get("text") or ""
        if isinstance(out.get("transcripts"), list) and out["transcripts"]:
            text = out["transcripts"][0].get("text") or text
        return jsonify({"text": text.strip()})
    except requests.exceptions.RequestException as e:
        status = getattr(e.response, "status_code", 500) if hasattr(e, "response") else 500
        body = {}
        if hasattr(e, "response") and e.response is not None and e.response.text:
            try:
                body = e.response.json()
            except Exception:
                body = {"detail": e.response.text[:200]}
        return jsonify({"error": "ElevenLabs STT failed", "detail": body.get("detail", str(e))}), min(status, 502)


@app.route("/api/dog/walk/direction", methods=["POST"])
def walk_direction_video():
    """
    Run the quadruped simulation walking forward or backward and return the rendered MP4 video.
    Body: { "direction": "forward"|"backward", "distance": float (optional, default 2.0),
            "n_steps": int (optional), "render_every": int (optional) }.
    Uses vector geometry: extracts the robot's heading from its torso quaternion,
    places a virtual waypoint at `distance` metres along that direction, and
    steers toward it each step.
    """
    data = request.get_json() or {}
    direction = data.get("direction", "forward")
    if direction not in ("forward", "backward"):
        return jsonify({"error": "direction must be 'forward' or 'backward'"}), 400

    distance = float(data.get("distance", 2.0))
    n_steps = int(data.get("n_steps", 500))
    render_every = int(data.get("render_every", 2))

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        _, final_x, final_y = run_quadruped(
            walk_direction=direction,
            walk_distance=distance,
            output=tmp_path,
            n_steps=n_steps,
            render_every=render_every,
        )

        # Update server-side dog state
        dog_state["x"] = final_x
        dog_state["y"] = final_y
        dog_state["status"] = "idle"

        response = send_file(
            tmp_path,
            mimetype="video/mp4",
            as_attachment=False,
            download_name=f"quadruped_walk_{direction}.mp4",
        )
        # Pass final position via custom headers so the frontend can update its state
        response.headers["X-Final-X"] = str(final_x)
        response.headers["X-Final-Y"] = str(final_y)
        response.headers["Access-Control-Expose-Headers"] = "X-Final-X, X-Final-Y"
        return response
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {e}"}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.route("/api/dog/walk/ball", methods=["POST"])
def walk_to_ball_video():
    """
    Run the quadruped simulation to chase the red ball and return the rendered MP4 video.
    Body (all optional): { "ball_x": float (default 1.0), "ball_y": float (default 0.0),
                           "n_steps": int, "render_every": int }.
    Returns the video with X-Final-X / X-Final-Y headers for the robot's final position.
    """
    data = request.get_json() or {}
    ball_x = float(data.get("ball_x", 1.0))
    ball_y = float(data.get("ball_y", 0.0))
    n_steps = int(data.get("n_steps", 500))
    render_every = int(data.get("render_every", 2))

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        _, final_x, final_y = run_quadruped(
            move_to_ball=True,
            ball_pos=(ball_x, ball_y, 0.1),
            output=tmp_path,
            n_steps=n_steps,
            render_every=render_every,
        )

        # Update server-side dog state
        dog_state["x"] = final_x
        dog_state["y"] = final_y
        dog_state["status"] = "idle"

        response = send_file(
            tmp_path,
            mimetype="video/mp4",
            as_attachment=False,
            download_name="quadruped_walk_to_ball.mp4",
        )
        response.headers["X-Final-X"] = str(final_x)
        response.headers["X-Final-Y"] = str(final_y)
        response.headers["Access-Control-Expose-Headers"] = "X-Final-X, X-Final-Y"
        return response
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {e}"}), 500
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.route("/api/dog/walk/video", methods=["POST"])
def walk_video():
    """
    Run the quadruped simulation to walk toward (x, y) and return the rendered MP4 video.
    Body: { "x": float, "y": float, "n_steps": int (optional), "render_every": int (optional) }.
    The simulation uses move_to_ball mode with ball_pos=(x, y, 0.1).
    """
    data = request.get_json() or {}
    try:
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "x and y must be numbers"}), 400

    n_steps = int(data.get("n_steps", 500))
    render_every = int(data.get("render_every", 2))

    # Create a temp file for the output video
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        _, final_x, final_y = run_quadruped(
            move_to_ball=True,
            ball_pos=(x, y, 0.1),
            output=tmp_path,
            n_steps=n_steps,
            render_every=render_every,
        )
        return send_file(
            tmp_path,
            mimetype="video/mp4",
            as_attachment=False,
            download_name="quadruped_walk.mp4",
        )
    except Exception as e:
        return jsonify({"error": f"Simulation failed: {e}"}), 500
    finally:
        # Clean up temp file after response is sent
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


@app.route("/api/video/convert", methods=["POST"])
def video_convert():
    """
    Convert uploaded video to another format. Form: file=video, format=mp4 (or webm).
    Returns the converted file. Requires ffmpeg installed for mp4 output.
    """
    fmt = (request.args.get("format") or request.form.get("format") or "mp4").strip().lower()
    if fmt not in ("mp4", "webm"):
        return jsonify({"error": "format must be mp4 or webm"}), 400
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No video file uploaded"}), 400
    if not _ffmpeg_available():
        return jsonify({"error": "Server does not have ffmpeg installed; cannot convert to mp4"}), 503
    tmp_in = tmp_out = None
    try:
        fd_in, tmp_in = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[-1] or ".webm")
        os.close(fd_in)
        file.save(tmp_in)
        fd_out, tmp_out = tempfile.mkstemp(suffix=f".{fmt}")
        os.close(fd_out)
        if fmt == "mp4":
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", tmp_out],
                check=True, capture_output=True, timeout=120,
            )
        else:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in, "-c:v", "libvpx-vp9", "-c:a", "libopus", tmp_out],
                check=True, capture_output=True, timeout=120,
            )
        with open(tmp_out, "rb") as f:
            data = f.read()
        mime = "video/mp4" if fmt == "mp4" else "video/webm"
        return Response(data, mimetype=mime, headers={"Content-Disposition": f"attachment; filename=kinetick9-walk.{fmt}"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Conversion failed", "detail": (e.stderr or b"").decode()[:500]}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    if ELEVENLABS_API_KEY:
        print("ElevenLabs: configured (voice on)")
    else:
        print("ElevenLabs: not configured — add ELEVENLABS_API_KEY to backend/.env and restart")
    app.run(host="0.0.0.0", port=port, debug=True)
