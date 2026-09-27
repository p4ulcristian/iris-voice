"""Iris's voice: OmniVoice (k2-fsa, 0.6B) behind the API Iris already speaks.

    POST /v1/audio/speech   multipart form → raw 16-bit mono PCM at 24 kHz, streamed
        text + ref_audio + ref_text (its exact words) [+ seed, language]   clone a voice
        text + instruction [+ seed, language]                              design one
    GET  /health            "ok" once the model is loaded

Run it with OmniVoice's own venv:  <omnivoice>/.venv/bin/python voice/server.py
It listens on 127.0.0.1 by default: there is no authentication.

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
import hashlib
import logging
import re
import tempfile
import threading
from collections import OrderedDict

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from omnivoice import OmniVoice

log = logging.getLogger("iris-voice")
RATE = 24000
MAX_CHARS = 5000
PROMPTS_KEPT = 16         # encoded reference clips kept in memory

app = FastAPI()
model: OmniVoice | None = None
lock = threading.Lock()   # the GPU makes one sentence at a time
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
):
    if model is None:
        raise HTTPException(503, "the voice is still loading")
    words = text.strip()[:MAX_CHARS]
    if not words:
        raise HTTPException(400, "no text")
    has_ref = ref_audio is not None and bool(ref_audio.filename)
    if has_ref != bool(ref_text.strip()):
        raise HTTPException(400, "ref_audio and ref_text must be given together")
    ref = (await ref_audio.read(), Path(ref_audio.filename).suffix or ".wav") if has_ref else None

    def stream():
        with lock:
            kw: dict = {"language": language or None}
            if ref:
                kw["voice_clone_prompt"] = clone_prompt(ref[0], ref[1], ref_text.strip())
            elif instruction.strip():
                kw["instruct"] = tags_for(instruction)
            for part in sentences(words):
                torch.manual_seed(seed)
                try:
                    audio = model.generate(text=part, **kw)[0]
                except Exception:
                    log.exception("generate failed on %r", part[:80])
                    return
                yield pcm(audio)

    return StreamingResponse(stream(), media_type="application/octet-stream")


def main() -> None:
    global model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", default="k2-fsa/OmniVoice")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("loading %s", args.model)
    model = OmniVoice.from_pretrained(args.model, device_map="cuda:0", dtype=torch.float16)
    model.generate(text="Ready.")            # warm up: the first sentence is slow otherwise
    log.info("ready on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
