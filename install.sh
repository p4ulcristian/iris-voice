#!/usr/bin/env bash
# Install Iris's voice (OmniVoice) on the machine the Iris server runs on. Safe to run again.
#
#   voice/install.sh
#
# Needs Linux, an NVIDIA GPU with ~2.5 GB free, uv and git. Takes ~8 GB of disk:
# OmniVoice's code and venv (VOICE_CODE) and the model in the Hugging Face cache.
# Runs as a user service listening on 127.0.0.1 (the API has no authentication).
set -euo pipefail
PORT=${VOICE_PORT:-7860}
CODE=${VOICE_CODE:-$HOME/.local/share/omnivoice}
HERE="$(dirname "$(readlink -f "$0")")"

[ -d "$CODE/.git" ] || git clone -q https://github.com/k2-fsa/OmniVoice.git "$CODE"
(cd "$CODE" && uv sync -q)

mkdir -p ~/.config/systemd/user
sed -e "s|CODE_DIR|$CODE|g" -e "s|SERVER_PY|$HERE/server.py|g" -e "s|VOICE_PORT|$PORT|g" \
    "$HERE/iris-voice.service" > ~/.config/systemd/user/iris-voice.service
systemctl --user daemon-reload
systemctl --user enable --now iris-voice.service
systemctl --user restart iris-voice.service
echo "Starting: ~30 s, plus the model download (~3 GB) the first time. Logs: journalctl --user -u iris-voice -f"
echo "Iris's default IRIS_VOICE_URL (http://127.0.0.1:$PORT) points here already."
