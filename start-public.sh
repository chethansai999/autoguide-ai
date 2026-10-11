#!/bin/bash
# Relaunch the full AutoGuide AI stack PLUS a public Cloudflare quick tunnel.
# Run from anywhere: bash ~/workspace/autoguide-ai/start-public.sh
#
# SANDBOX NOTE (verified 2026-10-11): inside this VM, direct outbound TCP is
# disabled by the Muse platform permission "other_tcp" (Muse settings ->
# Permissions -> Direct network protocols). cloudflared needs direct TCP to
# Cloudflare's edge, so the tunnel step FAILS here until that permission is
# switched from Deny to Ask/Allow. On any normal network (or once the
# permission is granted) this script works as-is: it starts the stack,
# opens the tunnel, and writes the public URL to logs/hosting.txt.
# The tunnel is temporary either way: the URL changes on every restart and
# dies with the VM.
set -u
PROJ="$HOME/workspace/autoguide-ai"
BIN_DIR="$HOME/workspace/bin"
cd "$PROJ" || exit 1
mkdir -p logs "$BIN_DIR"

# 1) Local stack: ollama + API (:8000) + UI (:8501)
bash "$PROJ/start-all.sh"

# 2) cloudflared binary (the UI runs the backend in-process because
#    AUTOGUIDE_API_URL is empty, so exposing port 8501 exposes the full app)
CF="$BIN_DIR/cloudflared"
if [ ! -x "$CF" ]; then
  echo "downloading cloudflared..."
  curl -fsSL -o "$CF" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CF"
fi

# 3) (Re)start the quick tunnel
for p in $(pgrep -x cloudflared); do kill "$p" 2>/dev/null || true; done
sleep 1
nohup "$CF" tunnel --url http://127.0.0.1:8501 > logs/cloudflared.log 2>&1 &
TUNNEL_PID=$!
echo "cloudflared starting (pid $TUNNEL_PID)..."

URL=""
for i in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" logs/cloudflared.log | grep -v "api.trycloudflare" | head -1)
  [ -n "$URL" ] && break
  sleep 2
done

{
  echo "public_url=${URL:-UNAVAILABLE - see logs/cloudflared.log; in this VM, direct TCP (other_tcp) is denied by platform permission}"
  echo "tunnel_pid=$TUNNEL_PID"
  echo "started=$(date -Is)"
  echo "local_ui=http://127.0.0.1:8501"
  echo "local_api=http://127.0.0.1:8000"
  echo "note=Temporary Cloudflare quick tunnel. Dies if the VM is replaced; URL changes on restart. Rerun start-public.sh to relaunch everything and get a fresh URL."
  echo "processes:"
  pgrep -af "ollama serve|uvicorn api.main|streamlit run app.py|cloudflared tunnel" || true
} > logs/hosting.txt
cat logs/hosting.txt
