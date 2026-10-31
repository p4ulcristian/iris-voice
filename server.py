"""Iris's voice: OmniVoice (k2-fsa, 0.6B) behind the API Iris already speaks.

    POST /v1/audio/speech   multipart form → raw 16-bit mono PCM at 24 kHz, streamed
        text [+ voice, seed, language]                                     a named voice (below)
        text + ref_audio + ref_text (its exact words) [+ seed, language]   clone a voice
        text + instruction [+ seed, language]                              design one
    GET  /v1/voices         the named voices and their languages, and the default
    GET  /health            "ok" once the model is loaded

Named voices are folders in --voices-dir: <name>/<language>.flac (or .wav) with
<language>.txt holding its exact words, one pair per language (en.flac, hu.flac...).
Text alone is spoken in --default-voice, from the clip in the request's language,
else its English clip, else its first; without a default voice OmniVoice picks one.

Every request needs `Authorization: Bearer <token>`, the token in --token-file
(install.sh makes one); without it the answer is 401. So it is safe on any
address, behind a proxy or not.

Run it with OmniVoice's own venv:  <omnivoice>/.venv/bin/python server.py
It listens on 127.0.0.1 by default. Each --host adds an address; one that is
not up yet (a VPN's) is bound anyway and starts answering when it comes up.

OmniVoice makes a sentence at a time, not a stream, so long text is spoken
sentence by sentence and the first one plays while the rest are made. A
reference clip is encoded once and kept (by its bytes and words), so a voice
costs nothing after its first sentence. One request at a time; others wait.
"""

from __future__ import annotations

import os
from pathlib import Path

# A cached model loads without asking the Hub first: that check can hang for
# minutes on a bad network and keep her silent. Set before huggingface_hub loads.
_HUB = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub")
if any((_HUB.expanduser() / "models--k2-fsa--OmniVoice" / "snapshots").glob("*/config.json")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import asyncio
import hashlib
import logging
import re
import secrets
import socket
import tempfile
import threading
from collections import OrderedDict

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

log = logging.getLogger("iris-voice")
RATE = 24000
MAX_CHARS = 5000
PROMPTS_KEPT = 16         # encoded reference clips kept in memory

token = b""                 # from --token-file, before the first request
voices_dir = Path()         # from --voices-dir
default_voice = ""          # from --default-voice; "" = OmniVoice picks a voice
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
bearer = HTTPBearer(auto_error=False)


def authorized(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
    if cred is None or not secrets.compare_digest(cred.credentials.encode(), token):
        raise HTTPException(401, "a bearer token is needed", headers={"WWW-Authenticate": "Bearer"})


# No /docs or /openapi.json: the API is in this docstring and the README.
app = FastAPI(dependencies=[Depends(authorized)], docs_url=None, redoc_url=None, openapi_url=None)
model: OmniVoice | None = None
lock = threading.Lock()   # the GPU makes one sentence at a time
turn = asyncio.Lock()     # and serves one request at a time; a hang-up gives up the turn
prompts: OrderedDict[str, object] = OrderedDict()

# OmniVoice designs a voice from tags, not prose. Iris's voices are described in
# words (server/speech.ts), so the words are mapped to the nearest tags.
TAGS = [
    (r"\b(woman|female|girl|she)\b", "female"),
    (r"\b(man|male|boy|he)\b", "male"),
    (r"\b(child|kid)\b", "child"),
    (r"\bteen", "teenager"),
    (r"\b(older|elderly|old)\b", "elderly"),
    (r"\b(thirties|forties|fifties|middle-aged)\b", "middle-aged"),
    (r"\b(young|twenties)\b", "young adult"),
    (r"\b(very deep|very low)\b", "very low pitch"),
    (r"\b(deep|low|husky)\b", "low pitch"),
    (r"\b(bright|high)\b", "high pitch"),
    (r"\bwhisper", "whisper"),
    (r"\bbritish\b", "british accent"),
    (r"\bamerican\b", "american accent"),
    (r"\baustralian\b", "australian accent"),
]
GROUPS = [{"female", "male"}, {"child", "teenager", "young adult", "middle-aged", "elderly"},
          {"very low pitch", "low pitch", "high pitch"}]


def tags_for(description: str) -> str:
    """ "A calm woman in her thirties with a low voice" → "female, middle-aged, low pitch".
    The first match of each kind wins; "woman" beats "man" by coming first."""
    words, picked = description.lower(), []
    for pattern, tag in TAGS:
        if re.search(pattern, words) and not any(tag in g and g & set(picked) for g in GROUPS) \
                and tag not in picked:
            picked.append(tag)
    return ", ".join(picked) or "female, young adult"


def sentences(text: str) -> list[str]:
    """Split at sentence ends, joining fragments too short to say well on their own. The
    first sentence stays short, so it plays while the rest are made."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(out[-1]) < 20:        # "Okay." alone sounds clipped
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def pcm(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def clone_prompt(data: bytes, suffix: str, ref_text: str):
    key = hashlib.sha256(data + b"\0" + ref_text.encode()).hexdigest()
    if key in prompts:
        prompts.move_to_end(key)
        return prompts[key]
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        prompt = model.create_voice_clone_prompt(ref_audio=f.name, ref_text=ref_text)
    prompts[key] = prompt
    while len(prompts) > PROMPTS_KEPT:
        prompts.popitem(last=False)
    return prompt


def language_code(language: str) -> str | None:
    """ "hu", "Hungarian" or "hungarian" → "hu"; None when OmniVoice does not know it."""
    key = language.strip()
    return key if key in LANG_IDS else LANG_NAME_TO_ID.get(key.lower())


def voice_languages(name: str) -> dict[str, Path]:
    """A named voice's clips by language code: <code>.flac or <code>.wav with <code>.txt beside it."""
    folder = voices_dir / name
    clips = {}
    for words in sorted(folder.glob("*.txt")):
        audio = next((a for a in (words.with_suffix(".flac"), words.with_suffix(".wav")) if a.is_file()), None)
        if audio:
            clips[words.stem] = audio
    return clips


def named_voice(name: str, language: str) -> tuple[bytes, str, str]:
    """The clip (bytes, suffix) and its words that speak `language` in voice `name`."""
    if not NAME.match(name):
        raise HTTPException(400, "voice: lowercase letters, digits, - and _")
    clips = voice_languages(name)
    if not clips:
        raise HTTPException(400, f"no voice named {name!r}; GET /v1/voices lists them")
    code = language_code(language) if language else None
    audio = clips.get(code) or clips.get("en") or next(iter(clips.values()))
    return audio.read_bytes(), audio.suffix, audio.with_suffix(".txt").read_text().strip()


@app.get("/v1/voices")
def voices():
    names = sorted(d.name for d in voices_dir.iterdir() if d.is_dir() and NAME.match(d.name)) \
        if voices_dir.is_dir() else []
    listed = {n: sorted(voice_languages(n)) for n in names}
    return {"default": default_voice or None, "voices": {n: langs for n, langs in listed.items() if langs}}


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(503, "loading")
    return PlainTextResponse("ok")


@app.post("/v1/audio/speech")
async def speech(
    text: str = Form(...),
    ref_audio: UploadFile | None = None,
    ref_text: str = Form(""),
    instruction: str = Form(""),
    cfg_scale: str = Form(""),     # Breeze's; accepted and ignored
    seed: int = Form(7),
    language: str = Form(""),
    voice: str = Form(""),
):
    if model is None:
        raise HTTPException(503, "the voice is still loading")
    words = text.strip()[:MAX_CHARS]
    if not words:
        raise HTTPException(400, "no text")
    has_ref = ref_audio is not None and bool(ref_audio.filename)
    if has_ref != bool(ref_text.strip()):
        raise HTTPException(400, "ref_audio and ref_text must be given together")
    if sum([has_ref, bool(instruction.strip()), bool(voice.strip())]) > 1:
        raise HTTPException(400, "one of voice, ref_audio + ref_text, or instruction")
    ref = (await ref_audio.read(), Path(ref_audio.filename).suffix or ".wav") if has_ref else None
    name = voice.strip() or ("" if instruction.strip() or has_ref else default_voice)
    if name:
        data, suffix, ref_text = named_voice(name, language)
        ref = (data, suffix)

    def say(part: str, kw: dict) -> bytes:
        with lock:
            torch.manual_seed(seed)
            return pcm(model.generate(text=part, **kw)[0])

    def prompt_for() -> object:
        with lock:
            return clone_prompt(ref[0], ref[1], ref_text.strip())

    # Async, so a client that hangs up cancels the stream and frees the turn. The old
    # sync generator sat paused at its yield after a hang-up, holding the lock forever,
    # and every later request waited behind it.
    async def stream():
        async with turn:
            kw: dict = {"language": language or None}
            if ref:
                kw["voice_clone_prompt"] = await asyncio.to_thread(prompt_for)
            elif instruction.strip():
                kw["instruct"] = tags_for(instruction)
            for part in sentences(words):
                try:
                    yield await asyncio.to_thread(say, part, kw)
                except Exception:
                    log.exception("generate failed on %r", part[:80])
                    return

    return StreamingResponse(stream(), media_type="application/octet-stream")


def main() -> None:
    global model, token, voices_dir, default_voice
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", action="append", help="address to listen on; repeat for more (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    ap.add_argument("--token-file", required=True, type=Path,
                    help="file holding the bearer token every request must carry")
    ap.add_argument("--voices-dir", type=Path, default=Path("~/.config/iris-voice/voices"),
                    help="named voices: <name>/<language>.flac + <language>.txt")
    ap.add_argument("--default-voice", default="", help="the named voice for text alone")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = args.token_file.expanduser().read_text().strip().encode()
    if not token:
        ap.error(f"{args.token_file} is empty")
    voices_dir = args.voices_dir.expanduser()
    default_voice = args.default_voice
    if default_voice and not voice_languages(default_voice):
        ap.error(f"--default-voice {default_voice}: no clips in {voices_dir / default_voice}")
    log.info("loading %s", args.model)
    model = OmniVoice.from_pretrained(args.model, device_map="cuda:0", dtype=torch.float16)
    model.generate(text="Ready.")            # warm up: the first sentence is slow otherwise
    hosts = args.host or ["127.0.0.1"]
    log.info("ready on %s, port %d", ", ".join(hosts), args.port)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[listen(h, args.port) for h in hosts])


def listen(host: str, port: int) -> socket.socket:
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # A VPN address may come up after this service; bind it now and serve it once it does.
    sock.setsockopt(socket.IPPROTO_IP, getattr(socket, "IP_FREEBIND", 15), 1)   # 15 on Linux
    sock.bind((host, port))
    return sock


if __name__ == "__main__":
    main()
