# Quilt NeuDecide Demo — Deployment

## Live URLs

| Component | URL |
|-----------|-----|
| **Frontend** (play the game) | https://quilt-neudecide-demo.pages.dev |
| **Backend API** (health) | https://quilt-demo.superinstance.ai/api/health |
| **Backend API** (inference) | https://quilt-demo.superinstance.ai/api/infer |
| **GitHub repo** | https://github.com/SuperInstance/quilt-neudecide-demo |

## Architecture

```
User's Browser
    │
    │  (Cloudflare Pages - static)
    ▼
quilt-neudecide-demo.pages.dev
    │
    │  POST /api/infer (WAV audio)
    │  GET  /api/health
    ▼
quilt-demo.superinstance.ai  (Cloudflare Tunnel)
    │
    │  (cloudflared on Oracle)
    ▼
Oracle: localhost:8001  (FastAPI + ONNX)
    │
    ├── audio_encoder_q4.onnx  (22MB, ~25ms)
    ├── tool_encoder_q4.onnx   (9.7MB, ~45ms)
    └── decoder_step_q4.onnx   (9.5MB, ~3ms/token)
```

**End-to-end:** ~113ms (audio → tool call)

## How to Play

1. Open https://quilt-neudecide-demo.pages.dev
2. Click "Hold to Talk"
3. Say "up", "down", "start", or "stop"
4. Watch the Pong paddle move AND the cells light up showing how the AI turned your voice into action

## Infrastructure

### Backend (Oracle)
- **Host:** Oracle Cloud (147.224.38.131), ARM 4-core
- **Models:** `/home/ubuntu/neudecide/` (43MB total)
- **Server:** `/home/ubuntu/quilt-demo/backend/app.py` (FastAPI, port 8001)
- **Process:** `uvicorn app:app --host 0.0.0.0 --port 8001`
- **Logs:** `/home/ubuntu/quilt-demo/backend/server.log`

### Cloudflare Tunnel
- **Name:** `quilt-neudecide-api`
- **ID:** `75578d3c-0715-42a1-87eb-c49ce24510ae`
- **Hostname:** `quilt-demo.superinstance.ai`
- **Binary:** `/home/ubuntu/cloudflared`
- **Config:** `/home/ubuntu/quilt-config.yml`
- **Credentials:** `/home/ubuntu/quilt-tunnel-creds.json` (600 perms)
- **Token:** `/home/ubuntu/.quilt-tunnel-token` (600 perms)
- **Logs:** `/home/ubuntu/quilt-tunnel.log`
- **Start:** `nohup /home/ubuntu/cloudflared tunnel --config /home/ubuntu/quilt-config.yml run > /home/ubuntu/quilt-tunnel.log 2>&1 &`

### Cloudflare Pages
- **Project:** `quilt-neudecide-demo`
- **Production:** https://quilt-neudecide-demo.pages.dev
- **Account ID:** `049ff5e84ecf636b53b162cbb580aae6`

### DNS
- `quilt-demo.superinstance.ai` → CNAME → `75578d3c-0715-42a1-87eb-c49ce24510ae.cfargotunnel.com` (proxied)

## Restarting Services

**Backend:**
```bash
# On Oracle via ~/workspace/ssh/osh.sh
pkill -f "uvicorn.*8001"
cd /home/ubuntu/quilt-demo/backend
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8001 > server.log 2>&1 &
```

**Tunnel:**
```bash
# On Oracle
pkill -f "quilt-neudecide-api"
nohup /home/ubuntu/cloudflared tunnel --config /home/ubuntu/quilt-config.yml run > /home/ubuntu/quilt-tunnel.log 2>&1 &
```

## Updating the Frontend

1. Edit `frontend/index.html` locally
2. Deploy: `python3 /tmp/cf_pages_deploy_quilt.py` (after updating SOURCE_DIR)
   Or via the Cloudflare dashboard

## Updating the Backend

1. Edit `backend/app.py` locally
2. Copy to Oracle: `scp` via the ssh wrapper
3. Restart the backend (see above)

## Security Notes

- Tunnel token and credentials are stored on Oracle with 600 permissions
- Never commit `.quilt-tunnel-token` or `quilt-tunnel-creds.json` to Git
- Backend has CORS enabled (`allow_origins=["*"]`) for the Pages frontend
- The tunnel exposes only port 8001; no firewall changes needed
