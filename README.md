# Hviske STT API

A lightweight server that exposes the [hviske-v5.3](https://huggingface.co/syvai/hviske-v5.3) Danish speech-to-text model as an OpenAI-compatible REST API. Run your own Whisper-compatible transcription endpoint locally or in production — no external API keys required.

hviske-v5.3 is a 2B-parameter Conformer encoder-decoder ASR model fine-tuned on the [CoRal v3](https://huggingface.co/datasets/CoRal-project/coral-v3) dataset. It outperforms Whisper Large v3 and GPT-4o Transcribe on Danish speech, achieving 3.63% CER on read-aloud and 11.35% CER on conversational speech (beam search, strict normalization).

[Model card](https://huggingface.co/syvai/hviske-v5.3) · [Hviske product page](https://syv.ai/produkter/hviske)

## Quick start

```bash
# Configure
cp .env.example .env
# Edit .env — set HF_TOKEN if the model is gated

# Run
docker compose up --build
```

The server starts on `http://localhost:8000`. The model is downloaded and loaded at startup before accepting requests.

## Transcribe audio

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=whisper-1"
```

## Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("sample.wav", "rb"),
)
print(transcript.text)
```

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/audio/transcriptions` | Transcribe audio to text |
| `POST /v1/audio/translations` | Not supported (returns 400) |
| `GET /v1/models` | List available models |
| `GET /v1/models/{id}` | Retrieve model info |

## Response formats

| Format | Description |
|---|---|
| `json` | `{text, usage}` (default) |
| `text` | Raw transcript string |
| `verbose_json` | `{task, language, duration, text, segments, words}` |
| `srt` | SubRip subtitle format |
| `vtt` | WebVTT subtitle format |

## Configuration

All settings are environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token for gated models |
| `HVISKE_MODEL_ID` | `syvai/hviske-v5.3` | Model to load |
| `HVISKE_TARGET_SR` | `16000` | Target sample rate |
| `HVISKE_DEFAULT_LANGUAGE` | `da` | Default language |
| `HVISKE_MAX_NEW_TOKENS` | `256` | Max generation tokens |
| `HVISKE_NUM_BEAMS` | `1` | Beam search count (5 = lowest WER, ~75% slower) |
| `HVISKE_LENGTH_PENALTY` | `1.0` | Beam search length penalty |
| `HVISKE_PUNCTUATION` | `true` | Enable punctuation tokens |
| `HVISKE_HOST` | `0.0.0.0` | Bind address |
| `HVISKE_PORT` | `8000` | Listen port |
| `HVISKE_LOG_LEVEL` | `info` | Uvicorn log level |

## Supported formats

wav, flac, mp3, ogg, m4a, mp4, webm (m4a/mp4 decoded via ffmpeg)

## License

This project is released under [CC BY-NC 4.0](LICENSE). See the [model card](https://huggingface.co/syvai/hviske-v5.3) for full details.
