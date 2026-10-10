# NeuDecide Pong — Magic Demystified

Voice-controlled Pong with visible AI cells. Speak "up", "down", "start", "stop" —
watch the Pong paddle move AND watch the NeuDecide neural network light up in real-time.

## The Demo

**Left:** Pong game (canvas). Voice controls the paddle.

**Right:** Spreadsheet-like cells showing the AI's inner workings:
- 📥 **Audio Input** — Your voice as a waveform
- 🔊 **Audio Encoder** — 28.7M params, waveform → latent frames (heatmap)
- 🔧 **Tool Encoder** — 11.8M params, audio + tools → cross-attention
- 🧠 **Decoder** — 15M params, generates tokens one by one
- ⚡ **Tool Call** — The final parsed action

Click any cell to expand it and see details. As you speak, cells light up in sequence.

## Architecture

```
Browser (mic) → WAV → POST /api/infer → Oracle (ONNX) → tool call → Pong moves
                                                        ↓
                                              visualization data → cells light up
```

**Backend:** FastAPI on Oracle, serving three ONNX graphs.
**Frontend:** HTML/JS, Web Audio API, Canvas. No build step.

## API

### POST /api/infer
Accepts: WAV audio file (multipart/form-data, 16kHz mono recommended, auto-converted).

Returns:
```json
{
  "tool_call": {"name": "moveUp", "arguments": {}} | null,
  "tokens_text": "decoded token string",
  "viz": {
    "waveform": [...200 points...],
    "waveform_len": 32000,
    "audio_embeds_shape": [1, 375, 512],
    "audio_embeds_heatmap": [[...16x32...]],
    "cross_key_shape": [1, 8, 4, 512, 64],
    "tokens": [2, 456, 123, ...],
    "n_tokens": 12
  },
  "timings": {
    "audio_decode_ms": 0.5,
    "audio_encoder_ms": 25.3,
    "tool_encoder_ms": 44.8,
    "decoder_ms": 42.1,
    "decoder_per_token_ms": 3.2,
    "total_ms": 113.2
  }
}
```

### GET /api/health
Returns: `{"status": "ok", "models": "loaded"}`

## Running

**Backend (Oracle):**
```bash
cd /home/ubuntu/quilt-demo/backend
python3 -m uvicorn app:app --host 0.0.0.0 --port 8001
```

**Frontend:** Served automatically at `http://<oracle-ip>:8001/`

## Performance (Oracle, 4-core ARM)

| Stage | Latency |
|-------|---------|
| Audio encoder (3s) | ~25ms |
| Tool encoder | ~45ms |
| Decoder | ~3ms/token |
| **End-to-end** | **~113ms** |

## The Point

This isn't a video explaining AI. It's AI working with the hood open.
You speak. You see the sound become numbers. You see the numbers become a decision.
You see the decision move the paddle.

Magic, demystified.
