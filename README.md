# voice

Iris's voice: [OmniVoice](https://github.com/k2-fsa/OmniVoice), a 0.6B
text-to-speech model, run on the same machine as the Iris server by
`server.py` here. It speaks 600+ languages, Hungarian among them.

Her ears are still to be built here; the plan is to reuse
[iris-dictation](../dictation)'s model.

## How it fits

```
Iris server (server/speech.ts)  ── text + reference clip ──▶  iris-voice (server.py, OmniVoice, GPU)
                                ◀── 24 kHz PCM, streamed ───
```

- **Voices** are described in words in `server/speech.ts` (`VOICES`); Settings
  lists them. Each is designed once from its description, speaking a fixed
  line; that reference clip is kept in `.memory/voices/`. Every sentence after
  that clones the clip, so she sounds like the same person each time. Editing
  a description makes a new clip. OmniVoice designs from tags, not prose, so
  `server.py` maps a description to the nearest ones ("a calm woman in her
  thirties with a low voice" → `female, middle-aged, low pitch`).
- **Other languages**: a clip in one language gives that language's accent in
  another. For accent-free Hungarian she needs a Hungarian reference clip.
- **Speed** on an RTX 5060 Ti: about 7× real time. OmniVoice makes a sentence
  at a time, so the first sentence plays while the rest are made.
- **Memory**: about 2.3 GB of GPU memory while it runs.
- **License**: the code is Apache 2.0, the model weights CC-BY-NC 4.0
  (non-commercial).

## Install

```bash
voice/install.sh
```

It clones OmniVoice (to `~/.local/share/omnivoice`, or `VOICE_CODE`), syncs its
venv, downloads the model (~2.5 GB) and starts the user service `iris-voice`
on `127.0.0.1:7860`, which is the Iris server's default `IRIS_VOICE_URL`. `VOICE_HOSTS="127.0.0.1 <address>"` makes it listen on more addresses
(a VPN's, for a proxy in front); only a firewall keeps others out, there is no
authentication here. If
`server/.env` sets `IRIS_VOICE_URL` to somewhere else, remove it and restart
the Iris server.

## API

`POST /v1/audio/speech`, multipart form, returns raw 16-bit mono PCM at 24 kHz
while it is made:

| fields | does |
|---|---|
| `text`, `ref_audio`, `ref_text` (its exact words), `seed` | clone a voice |
| `text`, `instruction`, `seed` | design a voice from a description |
| `language` (optional, e.g. `hu`) | the text's language; a little better than guessing |

`cfg_scale` is accepted and ignored (Breeze, the voice before this one, used
it). One request at a time; others wait. `GET /health` answers `ok` once the
model is loaded.
