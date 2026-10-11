#!/bin/bash
# Restart the full AutoGuide AI stack (local Llama via Ollama + API + UI).
# Run from anywhere: bash ~/workspace/autoguide-ai/start-all.sh
set -u
PROJ="$HOME/workspace/autoguide-ai"
# This sandbox's inherited no_proxy contains bracketed IPv6 entries that
# crash httpx's proxy parsing ("Invalid port: ':1]'"), silently breaking
# the NHTSA Recalls API. Keep only what the local services need.
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="localhost,127.0.0.1"
export LD_LIBRARY_PATH="$HOME/workspace/bin:${LD_LIBRARY_PATH:-}"
export OLLAMA_MODELS="$HOME/.ollama/models"

cd "$PROJ" || exit 1
mkdir -p logs

if ! curl -s --max-time 5 localhost:11434/api/tags >/dev/null 2>&1; then
  nohup "$HOME/workspace/bin/ollama" serve > logs/ollama.log 2>&1 &
  echo "ollama serve starting..."
  sleep 8
fi
curl -s --max-time 10 localhost:11434/api/tags | head -c 120; echo

if ! curl -s --max-time 5 localhost:8000/health >/dev/null 2>&1; then
  nohup "$PROJ/.venv/bin/python" -m uvicorn api.main:app --host 127.0.0.1 --port 8000 > logs/api.log 2>&1 &
  echo "api starting..."
  sleep 20
fi
curl -s --max-time 10 localhost:8000/health; echo

if ! curl -s --max-time 5 -o /dev/null localhost:8501 2>/dev/null; then
  nohup "$PROJ/.venv/bin/python" -m streamlit run app.py \
    --server.address 127.0.0.1 --server.port 8501 --server.headless true > logs/ui.log 2>&1 &
  echo "ui starting..."
  sleep 25
fi
curl -s -o /dev/null -w 'ui http=%{http_code}\n' --max-time 15 localhost:8501
echo "done."
