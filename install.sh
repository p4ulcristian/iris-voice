#!/usr/bin/env bash
# Install iris-voice (OmniVoice) as a user service. Safe to run again.
#
#   ./install.sh
#   VOICE_HOSTS="127.0.0.1 <vpn address>" ./install.sh   # also listen on the VPN
#
# Needs Linux, an NVIDIA GPU with ~5 GB free, uv and git. Takes ~8 GB of disk:
# OmniVoice's code and venv (VOICE_CODE) and the model in the Hugging Face cache.
# Every request needs the bearer token in VOICE_TOKEN_FILE; the first install makes one.
set -euo pipefail
PORT=${VOICE_PORT:-7860}
HOST_ARGS=$(printf -- '--host %s ' ${VOICE_HOSTS:-127.0.0.1})
CODE=${VOICE_CODE:-$HOME/.local/share/omnivoice}
TOKEN_FILE=${VOICE_TOKEN_FILE:-$HOME/.config/iris-voice/token}
HERE="$(dirname "$(readlink -f "$0")")"

if [ ! -s "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE")
  echo "Made a token in $TOKEN_FILE"
fi

[ -d "$CODE/.git" ] || git clone -q https://github.com/k2-fsa/OmniVoice.git "$CODE"
(cd "$CODE" && uv sync -q)

mkdir -p ~/.config/systemd/user
sed -e "s|CODE_DIR|$CODE|g" -e "s|SERVER_PY|$HERE/server.py|g" -e "s|VOICE_PORT|$PORT|g" -e "s|HOST_ARGS|${HOST_ARGS% }|g" -e "s|TOKEN_FILE|$TOKEN_FILE|g" \
    "$HERE/iris-voice.service" > ~/.config/systemd/user/iris-voice.service
systemctl --user daemon-reload
systemctl --user enable --now iris-voice.service
systemctl --user restart iris-voice.service
echo "Starting: ~30 s, plus the model download (~3 GB) the first time. Logs: journalctl --user -u iris-voice -f"
echo "Callers send: Authorization: Bearer \$(cat $TOKEN_FILE)  to http://127.0.0.1:$PORT"
