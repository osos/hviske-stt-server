import io
import json
import os
import time
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from fastapi import FastAPI, File, Form, UploadFile, Query, Path
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

# --- Configuration via environment variables ---

MODEL_ID = os.getenv("HVISKE_MODEL_ID", "syvai/hviske-v5.3")
TARGET_SR = int(os.getenv("HVISKE_TARGET_SR", "16000"))
DEFAULT_LANGUAGE = os.getenv("HVISKE_DEFAULT_LANGUAGE", "da")
MAX_NEW_TOKENS = int(os.getenv("HVISKE_MAX_NEW_TOKENS", "256"))
NUM_BEAMS = int(os.getenv("HVISKE_NUM_BEAMS", "1"))
LENGTH_PENALTY = float(os.getenv("HVISKE_LENGTH_PENALTY", "1.0"))
PUNCTUATION = os.getenv("HVISKE_PUNCTUATION", "true").lower() in ("true", "1", "yes")
HOST = os.getenv("HVISKE_HOST", "0.0.0.0")
PORT = int(os.getenv("HVISKE_PORT", "8000"))
LOG_LEVEL = os.getenv("HVISKE_LOG_LEVEL", "info")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_CACHE_DIR = os.getenv("HVISKE_MODEL_CACHE_DIR", "/root/.cache/huggingface")

app = FastAPI(
    title="Hviske STT",
    description="""
OpenAI-compatible speech-to-text API powered by [hviske-v5.3](https://huggingface.co/syvai/hviske-v5.3).

Danish ASR model (2B params) fine-tuned on CoRal v3. Optimized for Danish read-aloud and conversational speech.

## Quick start

Upload an audio file to `/v1/audio/transcriptions` with `model=whisper-1` (or any model ID — all map to the loaded Hviske model).

## Response formats

- **`json`** — `{text, usage}` (default)
- **`text`** — raw transcript string
- **`verbose_json`** — `{task, language, duration, text, segments, words}`
- **`srt`** — SubRip subtitle format
- **`vtt`** — WebVTT subtitle format

## Examples

### JSON response (default)

```bash
curl http://localhost:8000/v1/audio/transcriptions \\
  -F "file=@sample.wav" \\
  -F "model=whisper-1"
```

### Verbose JSON with segment timestamps

```bash
curl http://localhost:8000/v1/audio/transcriptions \\
  -F "file=@sample.wav" \\
  -F "model=whisper-1" \\
  -F "response_format=verbose_json" \\
  -F "timestamp_granularities=segment"
```

### Plain text output

```bash
curl http://localhost:8000/v1/audio/transcriptions \\
  -F "file=@sample.wav" \\
  -F "model=whisper-1" \\
  -F "response_format=text"
```

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("sample.wav", "rb"),
)
print(transcript.text)
```
    """,
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Audio",
            "description": "Speech-to-text transcription. Compatible with OpenAI Whisper API.",
        },
        {
            "name": "Models",
            "description": "List and retrieve available models.",
        },
    ],
)

processor = None
model = None


def load_model():
    global processor, model
    if processor is None or model is None:
        kwargs = {"trust_remote_code": True, "cache_dir": MODEL_CACHE_DIR}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        print(f"Loading processor from {MODEL_ID} ...")
        processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
        print(f"Loading model from {MODEL_ID} ...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            **kwargs,
        ).to(device="cuda").eval()
        print(f"Model loaded and ready on CUDA.")


@app.on_event("startup")
async def startup_event():
    print("Starting Hviske STT — loading model (this may take a minute)...")
    load_model()
    print("Warm-up complete. Server is ready.")


def read_audio(audio_bytes: bytes):
    import subprocess

    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf)
        audio = np.asarray(audio, dtype=np.float32)
        return audio, sr
    except Exception:
        pass

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = subprocess.run(
            ["ffmpeg", "-i", tmp.name, "-f", "s16le", "-ac", "1", "-ar", str(TARGET_SR), "-"],
            capture_output=True,
        )
        raw = result.stdout
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        sr = TARGET_SR

    return audio, sr


def transcribe(
    audio_bytes: bytes,
    language: str = DEFAULT_LANGUAGE,
    prompt: str | None = None,
    temperature: float = 0.0,
):
    import tempfile

    load_model()

    audio, sr = read_audio(audio_bytes)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    duration = len(audio) / sr

    inputs = processor(
        audio=audio,
        sampling_rate=sr,
        return_tensors="pt",
        language=language,
        punctuation=PUNCTUATION,
    )

    audio_chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = inputs.to(model.device, dtype=model.dtype)

    if "decoder_attention_mask" not in inputs:
        if "decoder_input_ids" in inputs:
            inputs["decoder_attention_mask"] = torch.ones_like(inputs["decoder_input_ids"])
        else:
            batch_size = inputs["input_features"].shape[0]
            inputs["decoder_attention_mask"] = torch.ones(
                (batch_size, 1),
                dtype=torch.long,
                device=model.device,
            )

    start_time = time.time()

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_beams": NUM_BEAMS,
        "length_penalty": LENGTH_PENALTY,
        "do_sample": False,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    elapsed = time.time() - start_time

    transcription = processor.decode(
        outputs[0],
        skip_special_tokens=True,
        audio_chunk_index=audio_chunk_index,
        language=language,
    )

    return {
        "text": transcription,
        "language": language,
        "duration": round(duration, 2),
        "processing_time": round(elapsed, 2),
    }


# --- Response models matching OpenAI spec ---


class TranscriptionResponse(BaseModel):
    """Transcription result in compact JSON format."""

    text: str = Field(
        ...,
        description="The transcribed text.",
        examples=["Hej og velkommen til Danmark."],
    )
    logprobs: list | None = Field(
        None,
        description="Log probabilities of tokens. Only populated when `include=logprobs`.",
    )
    usage: dict | None = Field(
        None,
        description="Usage statistics.",
        examples=[{"seconds": 12.34, "type": "duration"}],
    )


class TranscriptionSegment(BaseModel):
    """A segment of the transcription with timing info."""

    id: int = Field(..., description="Unique segment identifier.")
    avg_logprob: float = Field(0.0, description="Average log probability of the segment.")
    compression_ratio: float = Field(1.0, description="Compression ratio of the segment.")
    end: float = Field(..., description="End time in seconds.", examples=[5.2])
    start: float = Field(..., description="Start time in seconds.", examples=[0.0])
    temperature: float = Field(0.0, description="Sampling temperature used.")
    tokens: list[int] = Field([], description="Token IDs for this segment.")
    text: str = Field(..., description="Transcribed text for this segment.")


class TranscriptionWord(BaseModel):
    """A word-level timestamp annotation."""

    word: str = Field(..., description="The word text.")
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")


class VerboseTranscriptionResponse(BaseModel):
    """Detailed transcription result with metadata and optional segments/words."""

    task: str = Field("transcribe", description="Task type. Always `transcribe`.")
    language: str = Field(
        ...,
        description="Detected or specified language code.",
        examples=["da"],
    )
    duration: float = Field(
        ...,
        description="Audio duration in seconds.",
        examples=[12.34],
    )
    text: str = Field(..., description="Full transcribed text.")
    segments: list[TranscriptionSegment] = Field(
        [],
        description="Populated when `timestamp_granularities` includes `segment`.",
    )
    words: list[TranscriptionWord] = Field(
        [],
        description="Populated when `timestamp_granularities` includes `word`.",
    )


class TranscriptionDiarizedSegment(BaseModel):
    id: str
    end: float
    speaker: str
    start: float
    text: str
    type: str = "transcript.text.segment"


class TranscriptionDiarizedResponse(BaseModel):
    duration: float
    segments: list[TranscriptionDiarizedSegment]
    task: str = "transcribe"
    text: str
    usage: dict | None = None


class UsageDuration(BaseModel):
    seconds: float
    type: str = "duration"


def format_srt(result: dict) -> str:
    text = result["text"]
    duration = result["duration"]
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    milliseconds = int((duration % 1) * 1000)
    end_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return f"1\n00:00:00,000 --> {end_time}\n{text}\n"


def format_vtt(result: dict) -> str:
    text = result["text"]
    duration = result["duration"]
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    millis = int((duration % 1) * 1000)
    end_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"WEBVTT\n\n00:00:00.000 --> {end_time}\n{text}\n"


def build_usage(duration: float) -> dict:
    return {"seconds": round(duration, 2), "type": "duration"}


def build_segment(result: dict) -> list[TranscriptionSegment]:
    duration = result["duration"]
    return [
        TranscriptionSegment(
            id=0,
            start=0.0,
            end=duration,
            text=result["text"],
        )
    ]


# --- Endpoints ---


@app.post(
    "/v1/audio/transcriptions",
    tags=["Audio"],
    summary="Create transcription",
    description="Transcribe audio into text. Accepts flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.",
    response_model=TranscriptionResponse,
    responses={
        200: {
            "description": "Successful transcription.",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Hej og velkommen til Danmark.",
                        "usage": {"seconds": 12.34, "type": "duration"},
                    }
                },
                "text/plain": {
                    "example": "Hej og velkommen til Danmark."
                },
            },
        },
        400: {
            "description": "Invalid request.",
            "content": {
                "application/json": {
                    "example": {"error": "Diarization is not supported by Hviske."}
                }
            },
        },
    },
)
async def create_transcription(
    file: UploadFile = File(
        ...,
        description="Audio file to transcribe. Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm.",
    ),
    model: str = Form(
        ...,
        description="Model ID to use. Any value is accepted; the server uses the configured Hviske model.",
        examples=["whisper-1", "whisper-large-v3"],
    ),
    language: str | None = Form(
        None,
        description="Language of the input audio in ISO-639-1 format (e.g. `da`, `en`). Improves accuracy when set.",
        examples=["da"],
    ),
    prompt: str | None = Form(
        None,
        description="Text prompt to guide model style or continue a previous segment. Should match the audio language.",
        examples=["Dette er et interview med"],
    ),
    response_format: str = Form(
        "json",
        description="Output format. `json` returns {text, usage}. `text` returns raw string. `verbose_json` includes segments and metadata. `srt` and `vtt` return subtitle formats.",
        examples=["json"],
    ),
    temperature: float = Form(
        0.0,
        description="Sampling temperature between 0 and 1. Higher values produce more random output. 0 uses log-probability auto-selection.",
        examples=[0.0, 0.5],
    ),
    timestamp_granularities: str | None = Form(
        None,
        description="Comma-separated list of timestamp granularities: `word`, `segment`. Only effective with `response_format=verbose_json`.",
        examples=["segment", "word,segment"],
    ),
    include: str | None = Form(
        None,
        description="Comma-separated list of additional fields to include. Supported: `logprobs`.",
        examples=["logprobs"],
    ),
    stream: bool = Form(
        False,
        description="Stream response as server-sent events. Currently not supported; parameter accepted for compatibility.",
    ),
    chunking_strategy: str | None = Form(
        None,
        description="Chunking strategy: `auto` or `server_vad`. Accepted for compatibility; Hviske handles chunking internally.",
        examples=["auto"],
    ),
    known_speaker_names: str | None = Form(
        None,
        description="Comma-separated speaker names for diarization. Not supported by Hviske.",
    ),
    known_speaker_references: str | None = Form(
        None,
        description="Audio samples for known speaker references. Not supported by Hviske.",
    ),
):
    audio_bytes = await file.read()

    lang = language or DEFAULT_LANGUAGE
    result = transcribe(audio_bytes, language=lang, prompt=prompt, temperature=temperature)

    granularities = []
    if timestamp_granularities:
        granularities = [g.strip() for g in timestamp_granularities.split(",")]

    includes = []
    if include:
        includes = [i.strip() for i in include.split(",")]

    usage = build_usage(result["duration"])

    if response_format == "json":
        resp = TranscriptionResponse(text=result["text"])
        if "logprobs" in includes:
            resp.logprobs = []
        resp.usage = usage
        return resp

    elif response_format == "text":
        return PlainTextResponse(content=result["text"])

    elif response_format == "srt":
        return PlainTextResponse(content=format_srt(result), media_type="text/plain")

    elif response_format == "vtt":
        return PlainTextResponse(content=format_vtt(result), media_type="text/vtt")

    elif response_format == "verbose_json":
        segments = build_segment(result) if "segment" in granularities else []
        words = []
        return VerboseTranscriptionResponse(
            task="transcribe",
            language=result["language"],
            duration=result["duration"],
            text=result["text"],
            segments=segments,
            words=words,
        )

    elif response_format == "diarized_json":
        return PlainTextResponse(
            content=json.dumps({"error": "Diarization is not supported by Hviske. Use response_format=json or verbose_json instead."}),
            status_code=400,
            media_type="application/json",
        )

    else:
        return TranscriptionResponse(text=result["text"], usage=usage)


@app.post(
    "/v1/audio/translations",
    tags=["Audio"],
    summary="Create translation",
    description="Translate audio to English. Not supported by Hviske — use the transcriptions endpoint instead.",
)
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate."),
    model: str = Form(..., description="Model ID. Any value accepted."),
    language: str | None = Form(None, description="Language of input audio."),
    prompt: str | None = Form(None, description="Translation prompt."),
    response_format: str = Form("json", description="Output format."),
    temperature: float = Form(0.0, description="Sampling temperature."),
):
    return PlainTextResponse(
        content=json.dumps({"error": "Translation is not supported by Hviske. Use the transcriptions endpoint instead."}),
        status_code=400,
        media_type="application/json",
    )


@app.get(
    "/v1/models",
    tags=["Models"],
    summary="List models",
    description="List available models. All model IDs map to the loaded Hviske model.",
    responses={
        200: {
            "description": "List of available models.",
            "content": {
                "application/json": {
                    "example": {
                        "object": "list",
                        "data": [
                            {
                                "id": "syvai/hviske-v5.3",
                                "object": "model",
                                "created": 1700000000,
                                "owned_by": "syvai",
                            }
                        ],
                    }
                }
            },
        }
    },
)
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 1700000000,
                "owned_by": "syvai",
            },
            {
                "id": "whisper-large-v3",
                "object": "model",
                "created": 1700000000,
                "owned_by": "syvai",
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "syvai",
            },
        ],
    }


@app.get(
    "/v1/models/{model_id:path}",
    tags=["Models"],
    summary="Retrieve model",
    description="Get details for a specific model. Any model ID is accepted and returns metadata for the loaded Hviske model.",
    responses={
        200: {
            "description": "Model details.",
            "content": {
                "application/json": {
                    "example": {
                        "id": "syvai/hviske-v5.3",
                        "object": "model",
                        "created": 1700000000,
                        "owned_by": "syvai",
                    }
                }
            },
        }
    },
)
async def retrieve_model(
    model_id: str = Path(..., description="Model ID to retrieve.", examples=["syvai/hviske-v5.3", "whisper-1"]),
):
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "syvai",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL)
