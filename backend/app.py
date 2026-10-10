"""
NeuDecide Voice-Controlled Pong Demo - Backend API.

Serves the three NeuDecide ONNX graphs and exposes a simple HTTP API:
  POST /api/infer  - audio (WAV) in, tool call + visualization data out
  GET  /api/health - health check

The frontend (Pong game + cell visualization) talks to this API.
"""
import io
import json
import time
import wave
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import sentencepiece as spm

MDIR = "/home/ubuntu/neudecide"
FRONTEND_DIR = "/home/ubuntu/quilt-demo/frontend"

# --- Pong tools (what the model can do) ---
PONG_TOOLS = [
    {
        "name": "moveUp",
        "description": "Moves the Pong paddle up.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "moveDown",
        "description": "Moves the Pong paddle down.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "startGame",
        "description": "Starts the Pong game.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "stopGame",
        "description": "Stops/pauses the Pong game.",
        "parameters": {"type": "object", "properties": {}},
    },
]

app = FastAPI(title="NeuDecide Pong Demo")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model loading (at startup) ---
print("Loading NeuDecide models...")
OPTS = ort.SessionOptions()
OPTS.intra_op_num_threads = 4

ae = ort.InferenceSession(f"{MDIR}/audio_encoder_q4.onnx",
                          providers=["CPUExecutionProvider"], sess_options=OPTS)
te = ort.InferenceSession(f"{MDIR}/tool_encoder_q4.onnx",
                          providers=["CPUExecutionProvider"], sess_options=OPTS)
dc = ort.InferenceSession(f"{MDIR}/decoder_step_q4.onnx",
                          providers=["CPUExecutionProvider"], sess_options=OPTS)

sp = spm.SentencePieceProcessor(model_file=f"{MDIR}/tokenizer.model")

# Pre-encode tools (they don't change)
TOOLS_STR = json.dumps(PONG_TOOLS)
TOOL_IDS = sp.encode(TOOLS_STR)
print(f"Tools encoded: {len(TOOL_IDS)} tokens")
print("Models loaded.")

BOS, EOS, MAX_TOKENS = 2, 1, 32


def wav_to_mono_16k(wav_bytes: bytes) -> np.ndarray:
    """Convert WAV bytes to mono float32 at 16kHz."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    # Decode based on sample width
    if sampwidth == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    # Resample to 16kHz (simple linear interpolation)
    if framerate != 16000:
        duration = len(audio) / framerate
        n_target = int(duration * 16000)
        old_idx = np.linspace(0, len(audio) - 1, len(audio))
        new_idx = np.linspace(0, len(audio) - 1, n_target)
        audio = np.interp(new_idx, old_idx, audio).astype(np.float32)

    # Truncate to 30s max
    max_samples = 30 * 16000
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    return audio


def downsample(arr: np.ndarray, n: int) -> list:
    """Downsample 1D array to n points for visualization."""
    if len(arr) <= n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return arr[idx].tolist()


def heatmap_2d(arr: np.ndarray, h: int = 16, w: int = 32) -> list:
    """Downsample 2D array to hxw heatmap for visualization."""
    # arr shape: (time, features) or (features, time) - handle both
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Take mean across batch if present
    while arr.ndim > 2:
        arr = arr.mean(axis=0)

    th, tw = arr.shape
    # Downsample
    h_idx = np.linspace(0, th - 1, min(h, th)).astype(int)
    w_idx = np.linspace(0, tw - 1, min(w, tw)).astype(int)
    small = arr[np.ix_(h_idx, w_idx)]
    return small.tolist()


@app.get("/api/health")
def health():
    return {"status": "ok", "models": "loaded"}


@app.post("/api/infer")
async def infer(audio: UploadFile = File(...)):
    """
    Run NeuDecide on uploaded WAV audio.

    Returns:
      - tool_call: parsed tool call (or null)
      - tokens: generated token IDs and text
      - viz: visualization data (waveform, embeds heatmap, etc.)
      - timings: per-stage latency in ms
    """
    timings = {}
    t_start = time.perf_counter()

    # 1. Decode audio
    wav_bytes = await audio.read()
    waveform = wav_to_mono_16k(wav_bytes)
    n_samples = len(waveform)
    timings["audio_decode_ms"] = (time.perf_counter() - t_start) * 1000

    # 2. Audio encoder
    t0 = time.perf_counter()
    ae_out = ae.run(None, {
        "waveform": waveform.reshape(1, -1).astype(np.float32),
        "waveform_len": np.array([n_samples], dtype=np.int64),
    })
    audio_embeds, audio_len = ae_out[0], ae_out[1]
    timings["audio_encoder_ms"] = (time.perf_counter() - t0) * 1000

    # 3. Tool encoder
    t0 = time.perf_counter()
    te_out = te.run(None, {
        "audio_embeds": audio_embeds.astype(np.float32),
        "audio_len": audio_len.astype(np.int64),
        "tools_ids": np.array([TOOL_IDS], dtype=np.int64),
    })
    cross_key, cross_value, enc_mask = te_out
    timings["tool_encoder_ms"] = (time.perf_counter() - t0) * 1000

    # 4. Decoder (greedy)
    t0 = time.perf_counter()
    past_k = np.zeros((8, 1, 4, 0, 64), dtype=np.float32)
    past_v = np.zeros((8, 1, 4, 0, 64), dtype=np.float32)
    inp = np.array([[BOS]], dtype=np.int64)
    gen_tokens = []
    step_times = []

    for _ in range(MAX_TOKENS):
        ts = time.perf_counter()
        logits, past_k, past_v = dc.run(None, {
            "input_ids": inp,
            "past_key": past_k,
            "past_value": past_v,
            "cross_key": cross_key,
            "cross_value": cross_value,
            "encoder_mask": enc_mask,
        })
        step_times.append((time.perf_counter() - ts) * 1000)
        tok = int(np.argmax(logits[0]))
        gen_tokens.append(tok)
        inp = np.array([[tok]], dtype=np.int64)
        if tok == EOS:
            break

    timings["decoder_ms"] = (time.perf_counter() - t0) * 1000
    timings["decoder_per_token_ms"] = float(np.mean(step_times)) if step_times else 0
    timings["total_ms"] = (time.perf_counter() - t_start) * 1000

    # 5. Decode and parse
    gen_text = sp.decode(gen_tokens)

    # Try to parse as tool call JSON
    tool_call = None
    try:
        # The model outputs something like [{"name": "...", "arguments": {...}}]
        parsed = json.loads(gen_text)
        if isinstance(parsed, list) and len(parsed) > 0:
            tool_call = parsed[0]
        elif isinstance(parsed, dict):
            tool_call = parsed
    except:
        pass

    # 6. Build visualization data
    viz = {
        "waveform": downsample(waveform, 200),
        "waveform_len": n_samples,
        "audio_embeds_shape": list(audio_embeds.shape),
        "audio_embeds_heatmap": heatmap_2d(audio_embeds[0], h=16, w=32),
        "cross_key_shape": list(cross_key.shape),
        "tokens": gen_tokens,
        "tokens_text": gen_text,
        "n_tokens": len(gen_tokens),
    }

    return {
        "tool_call": tool_call,
        "tokens_text": gen_text,
        "viz": viz,
        "timings": timings,
    }


# Serve frontend static files
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")
