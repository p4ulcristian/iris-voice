#!/usr/bin/env bash
# Install Iris's voice (Breeze TTS 2) on a GPU machine. Safe to run again.
#
#   VOICE_HOST=<address to listen on> voice/install.sh
#
# VOICE_HOST should be a private address only the Iris server can reach (a
# WireGuard tunnel, say): the API has no authentication. Needs Linux, an
# NVIDIA GPU with 16 GB free for the fast path (about 8 GB without it), uv,
# git and sudo. Takes ~10 GB of disk: the code, its venv and the 7 GB weights.
set -euo pipefail
: "${VOICE_HOST:?set VOICE_HOST to the private address to listen on}"
PORT=${VOICE_PORT:-7860}
CODE=${VOICE_CODE:-$HOME/Work/breeze-tts}
WEIGHTS=${VOICE_WEIGHTS:-$HOME/Work/breeze-tts-2}
UNIT_DIR="$(dirname "$(readlink -f "$0")")"

[ -d "$CODE/.git" ] || git clone -q https://github.com/breezeblue-ai/breeze-tts.git "$CODE"
cd "$CODE"
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
# torch built for CUDA 12.8, which covers Blackwell (RTX 50xx) as well as older cards.
uv pip install -q --python .venv/bin/python torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python .venv/bin/python -r requirements.txt "huggingface_hub[cli]"
# Iris's fix to Breeze's API: requests queue instead of being refused, and a stream cut off
# early (a reload, an interruption) no longer leaves the server refusing everything after it.
git apply --reverse --check "$UNIT_DIR/breeze-api.patch" 2>/dev/null || git apply "$UNIT_DIR/breeze-api.patch"
.venv/bin/hf download BreezeBlue/breeze-tts-2 --local-dir "$WEIGHTS"

sed -e "s|YOUR_USER|$USER|g" -e "s|CODE_DIR|$CODE|g" -e "s|WEIGHTS_DIR|$WEIGHTS|g" \
    -e "s|VOICE_HOST|$VOICE_HOST|g" -e "s|VOICE_PORT|$PORT|g" \
    "$UNIT_DIR/iris-voice.service" | sudo tee /etc/systemd/system/iris-voice.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now iris-voice.service
echo "Starting; the fast path warms up for about 100 s. Logs: journalctl -u iris-voice -f"
echo "Then set IRIS_VOICE_URL=http://$VOICE_HOST:$PORT in server/.env and deploy."
