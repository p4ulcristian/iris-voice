# voice

Iris's voice: [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts), a
3B text-to-speech model, served on a GPU machine and called by the Iris server.
English only (Breeze also speaks Chinese).

Her ears are still to be built here; the plan is to reuse
[iris-dictation](../dictation)'s model.

## How it fits

```
Iris server (server/speech.ts)  ── text + reference clip ──▶  iris-voice (Breeze, GPU)
                                ◀── 24 kHz PCM, streamed ───
```

- **Voices** are described in words in `server/speech.ts` (`VOICES`); Settings
  lists them. Breeze designs each one once from its description, speaking a
  fixed line; that reference clip is kept in `.memory/voices/`. Every sentence
  after that clones the clip, so she sounds like the same person each time.
  Editing a description makes a new clip.
- **Speed** on an RTX 5090, from the Iris server over a tunnel: first audio in
  0.2 to 0.5 s, about 3x real time. The reference upload (~130 KB, FLAC) is most
  of the difference from the 0.2 s Breeze manages with no reference.
- **License**: the code is Apache 2.0, the weights are for research and
  non-commercial use only.

## Install (on the GPU machine)

```bash
VOICE_HOST=<private address> voice/install.sh
```

It clones Breeze, makes its venv (torch for CUDA 12.8), downloads the weights
(7 GB) and starts `iris-voice.service`. With `--fast-all` it holds about 16 GB of
GPU memory and warms up for about 100 s at start; drop the flag for about 8 GB
and slower first audio.

The API has no authentication, so `VOICE_HOST` must be an address only the Iris
server can reach. Then, on the Iris machine, set
`IRIS_VOICE_URL=http://<address>:7860` in `server/.env` and deploy.

## API

`POST /v1/audio/speech`, multipart form, returns raw 16-bit mono PCM at 24 kHz
while it is made:

| fields | does |
|---|---|
| `text`, `instruction`, `cfg_scale` (4), `seed` | design a voice from a description |
| `text`, `ref_audio`, `ref_text` (its exact words), `seed` | clone a voice (no `cfg_scale`) |

One request at a time; others wait up to 30 s (`breeze-api.patch`, applied by `install.sh`: upstream refuses a second request with 409, and a stream cut off early left it refusing every request until a restart).
