# iris-voice

Text to speech on your own GPU: [OmniVoice](https://github.com/k2-fsa/OmniVoice),
a 0.6B model that speaks 600+ languages (Hungarian among them), behind an
OpenAI-style `POST /v1/audio/speech` and a bearer token. It is the voice of
[Iris](https://github.com/topics/iris); its ears are
[iris-listen](https://github.com/p4ulcristian/iris-listen).

## How it fits

```
any program ── Bearer token, text (+ a voice) ──▶ iris-voice (server.py, OmniVoice, GPU)
            ◀── 24 kHz PCM, streamed ──────────
```

- **The token**: every request needs `Authorization: Bearer <token>`, or the
  answer is 401. The token is in `~/.config/iris-voice/token`, made by the
  first install. So the service is safe on any address, with or without a
  proxy in front.
- **Named voices**: folders in `~/.config/iris-voice/voices/` (`VOICE_VOICES_DIR`),
  one per voice: `<language>.flac` (or `.wav`) and `<language>.txt` with its
  exact words, one pair per language (`iris/en.flac`, `iris/hu.flac`...). A
  request names one with `voice`; text alone is spoken in the default voice
  (`VOICE_DEFAULT` at install). The clip used is the one in the request's
  `language`, else English, else the first. The folders are local: nothing
  of them is in this repo.
- **Other voices**: a voice is cloned from a reference clip (`ref_audio` + its words
  in `ref_text`), or designed from a description (`instruction`). A
  description alone drifts a little from one sentence to the next, so a caller
  that wants the same voice every time designs it once and clones that clip
  after. OmniVoice designs from tags, not prose, so `server.py` maps a
  description to the nearest ones ("a calm woman in her thirties with a low
  voice" → `female, middle-aged, low pitch`).
- **Other languages**: a clip in one language gives that language's accent in
  another. For accent-free Hungarian, clone a Hungarian clip.
- **Speed** on an RTX 5060 Ti: about 7× real time. OmniVoice makes a sentence
  at a time, so the first sentence plays while the rest are made.
- **Memory**: about 5 GB of GPU memory while it runs.
- **License**: the code is Apache 2.0, the model weights CC-BY-NC 4.0
  (non-commercial).

## Install

```bash
git clone https://github.com/p4ulcristian/iris-voice && iris-voice/install.sh
```

It makes the token, clones OmniVoice (to `~/.local/share/omnivoice`, or
`VOICE_CODE`), syncs its venv, downloads the model (~2.5 GB) and starts the
user service `iris-voice` on `127.0.0.1:7860` (`VOICE_PORT`).
`VOICE_HOSTS="127.0.0.1 <address>"` makes it listen on more addresses, a VPN's
for a proxy in front; an address that is not up yet is bound anyway and
answers once it is. `VOICE_TOKEN_FILE` puts the token somewhere else;
`VOICE_DEFAULT=<name>` makes a named voice the one for text alone.

## API

`POST /v1/audio/speech`, multipart form, returns raw 16-bit mono PCM at 24 kHz
while it is made:

| fields | does |
|---|---|
| `text`, `voice` (optional), `seed` | a named voice; without `voice`, the default one |
| `text`, `ref_audio`, `ref_text` (its exact words), `seed` | clone a voice |
| `text`, `instruction`, `seed` | design a voice from a description |
| `language` (optional, e.g. `hu`) | the text's language; a little better than guessing |

`voice`, `ref_audio` and `instruction` exclude each other. `GET /v1/voices`
lists the named voices and their languages, and the default:
`{"default": "iris", "voices": {"iris": ["en", "hu"]}}`.

`cfg_scale` is accepted and ignored (Breeze, the voice before this one, used
it). One request at a time; others wait. `GET /health` answers `ok` once the
model is loaded.

```bash
curl -s -H "Authorization: Bearer $(cat ~/.config/iris-voice/token)" \
  -F text="Hello there." \
  http://127.0.0.1:7860/v1/audio/speech | aplay -f S16_LE -r 24000 -c 1
```
