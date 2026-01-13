# External API Integration Documentation

## Overview

This document details the integration of external APIs (OpenAI and ElevenLabs) in the MuseTalk server, following official documentation and best practices.

---

## OpenAI Integration

### Documentation Reference
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create)

### Implementation

#### Client Initialization
```python
from openai import AsyncOpenAI, OpenAIError

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=30.0,      # 30 second timeout
    max_retries=0      # Manual retry handling
)
```

#### Chat Completions
```python
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=False
)

text = response.choices[0].message.content
```

#### Speech-to-Text (Whisper)
```python
with open(audio_path, "rb") as audio_file:
    transcript = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
```

### Error Handling

#### Retry Logic
- **Attempts**: 3 retries
- **Backoff**: Exponential (1s, 2s, 4s)
- **Exceptions**: `OpenAIError` for API-specific errors

```python
for attempt in range(max_retries):
    try:
        response = await client.chat.completions.create(...)
        break
    except OpenAIError as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
        else:
            # Fallback response
            return "I'm sorry, I'm having trouble..."
```

### Features Used

✅ **Chat Completions** - AI conversation
✅ **Whisper API** - Speech-to-Text transcription
✅ **Async Client** - Non-blocking operations
✅ **Error Handling** - Graceful degradation
✅ **Retry Logic** - Resilience to temporary failures

---

## ElevenLabs Integration

### Documentation Reference
- [ElevenLabs API Introduction](https://elevenlabs.io/docs/api-reference/introduction)
- [Streaming Documentation](https://elevenlabs.io/docs/api-reference/streaming)
- [Text-to-Speech API](https://elevenlabs.io/docs/api-reference/text-to-speech)

### Implementation

#### Client Initialization
```python
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

client = AsyncElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    timeout=30.0
)
```

#### Streaming Text-to-Speech

Following the [official streaming guide](https://elevenlabs.io/docs/api-reference/streaming):

```python
audio_generator = await client.text_to_speech.convert_as_stream(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
    text="Hello, world!",
    model_id="eleven_multilingual_v2",
    voice_settings=VoiceSettings(
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True
    ),
    output_format="mp3_44100_128"  # High quality MP3
)

# Stream chunks as they arrive
async for chunk in audio_generator:
    if chunk:
        # Process or send chunk immediately
        process_audio_chunk(chunk)
```

### Output Formats

Available formats (from [API docs](https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input)):

| Format | Sample Rate | Bitrate | Use Case |
|--------|-------------|---------|----------|
| `mp3_44100_128` | 44.1 kHz | 128 kbps | **High quality (recommended)** |
| `mp3_44100_192` | 44.1 kHz | 192 kbps | Premium quality |
| `mp3_22050_32` | 22.05 kHz | 32 kbps | Low bandwidth |
| `pcm_44100` | 44.1 kHz | - | Uncompressed |

**Our Choice**: `mp3_44100_128` - Best balance of quality and file size

### Voice Settings

```python
VoiceSettings(
    stability=0.5,           # 0-1: Lower = more expressive
    similarity_boost=0.75,   # 0-1: Higher = more similar to original
    style=0.0,               # 0-1: Style exaggeration
    use_speaker_boost=True   # Enhance speaker characteristics
)
```

### Error Handling

#### Retry Logic
- **Attempts**: 3 retries
- **Backoff**: Exponential (1s, 2s, 4s)
- **Streaming**: Retry entire request on failure

```python
for attempt in range(max_retries):
    try:
        audio_generator = await client.text_to_speech.convert_as_stream(...)
        
        async for chunk in audio_generator:
            yield chunk
        
        break  # Success
        
    except Exception as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay)
            retry_delay *= 2
        else:
            raise
```

### Features Used

✅ **Streaming TTS** - Low-latency audio generation
✅ **Voice Settings** - Customizable voice characteristics
✅ **Output Formats** - High-quality MP3 output
✅ **Async Client** - Non-blocking operations
✅ **Chunked Transfer** - Progressive audio delivery

---

## Integration Architecture

### Data Flow

```
User Speech
    ↓
[Audio Buffer]
    ↓
[OpenAI Whisper] ← Speech-to-Text
    ↓
[Transcript]
    ↓
[OpenAI GPT-4] ← Chat Completion
    ↓
[AI Response Text]
    ↓
[ElevenLabs TTS] ← Text-to-Speech Streaming
    ↓
[Audio Chunks] → Client (streaming playback)
    ↓
[MuseTalk] ← Lip-sync Generation
    ↓
[Video Frames] → Client (streaming display)
```

### Latency Optimization

1. **Streaming TTS**: Audio chunks sent immediately as generated
2. **Parallel Processing**: Lip-sync generation starts while audio streams
3. **Client Buffering**: Client can start playback before full audio ready
4. **Async Operations**: All API calls are non-blocking

### Performance Metrics

| Stage | Average Time | API |
|-------|--------------|-----|
| Speech-to-Text | ~500ms | OpenAI Whisper |
| Chat Completion | ~1000ms | OpenAI GPT-4o-mini |
| TTS Generation | ~500ms | ElevenLabs |
| Lip-sync Video | ~1000ms | MuseTalk |
| **Total** | **~3s** | - |

---

## Error Handling Strategy

### Retry Configuration

```python
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0   # exponential backoff
```

### Error Types

#### OpenAI Errors
- `OpenAIError` - Base exception for all OpenAI errors
- `APIError` - API-level errors (500, 503)
- `RateLimitError` - Rate limit exceeded
- `AuthenticationError` - Invalid API key
- `Timeout` - Request timeout

#### ElevenLabs Errors
- `APIError` - General API errors
- `RateLimitError` - Character limit exceeded
- `AuthenticationError` - Invalid API key
- Network errors - Connection issues

### Fallback Strategies

1. **OpenAI Chat Failure**:
   - Return generic error message
   - Maintain conversation context
   - Log error for monitoring

2. **OpenAI Whisper Failure**:
   - Return empty transcript
   - Skip processing pipeline
   - Notify user of transcription failure

3. **ElevenLabs TTS Failure**:
   - Retry with exponential backoff
   - After max retries, fail gracefully
   - Send error message to client

---

## Configuration

### Environment Variables

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...
OPENAI_TIMEOUT=30.0

# ElevenLabs Configuration
ELEVENLABS_API_KEY=sk_...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
ELEVENLABS_MODEL=eleven_multilingual_v2
ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128
```

### Voice Configuration

Default voice: **Rachel** (`21m00Tcm4TlvDq8ikWAM`)

To change voice:
1. Browse voices at [ElevenLabs Voice Library](https://elevenlabs.io/voice-library)
2. Get voice ID
3. Update `ELEVENLABS_VOICE_ID` in `.env`

---

## Monitoring & Logging

### Performance Tracking

```python
# Metrics tracked per request
metrics = {
    "transcription_time_ms": 500,
    "llm_response_time_ms": 1000,
    "tts_generation_time_ms": 500,
    "total_latency_ms": 3000
}
```

### Logging Format

```
🎤 Transcribing audio (attempt 1/3)...
⏱️  Transcription: 500ms
🤖 Getting AI response (attempt 1/3)...
⏱️  LLM Response: 1000ms
🔊 Generating TTS audio with ElevenLabs streaming...
⏱️  TTS Generation: 500ms
✅ TTS audio generated (45678 bytes, 23 chunks)
```

### Error Logging

```
⚠️  OpenAI API attempt 1 failed: Rate limit exceeded, retrying in 1.0s...
❌ ElevenLabs TTS failed after 3 attempts: Connection timeout
```

---

## Best Practices

### 1. API Key Security
- ✅ Store in `.env` file
- ✅ Never commit to git
- ✅ Use environment variables
- ❌ Never hardcode in source

### 2. Rate Limiting
- ✅ Implement retry logic
- ✅ Use exponential backoff
- ✅ Monitor usage
- ✅ Set session limits

### 3. Error Handling
- ✅ Catch specific exceptions
- ✅ Provide fallback responses
- ✅ Log errors with context
- ✅ Track consecutive failures

### 4. Performance
- ✅ Use async operations
- ✅ Stream responses when possible
- ✅ Track latency metrics
- ✅ Optimize for TTFB (Time To First Byte)

### 5. User Experience
- ✅ Show status updates
- ✅ Handle errors gracefully
- ✅ Provide feedback
- ✅ Maintain conversation context

---

## Testing

### Unit Tests

```python
async def test_openai_chat():
    client = AsyncOpenAI(api_key="test_key")
    response = await client.chat.completions.create(...)
    assert response.choices[0].message.content

async def test_elevenlabs_tts():
    client = AsyncElevenLabs(api_key="test_key")
    audio = await client.text_to_speech.convert_as_stream(...)
    chunks = [chunk async for chunk in audio]
    assert len(chunks) > 0
```

### Integration Tests

```bash
# Test with real APIs
python -m pytest tests/test_api_integration.py

# Test error handling
python -m pytest tests/test_error_handling.py
```

---

## Troubleshooting

### Common Issues

**Issue**: `OpenAIError: Invalid API key`
- **Solution**: Check `OPENAI_API_KEY` in `.env`

**Issue**: `ElevenLabs rate limit exceeded`
- **Solution**: Check character usage, upgrade plan if needed

**Issue**: `Timeout errors`
- **Solution**: Increase timeout value, check network connection

**Issue**: `Audio quality poor`
- **Solution**: Use higher bitrate format (e.g., `mp3_44100_192`)

---

## References

### Official Documentation
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [ElevenLabs API Introduction](https://elevenlabs.io/docs/api-reference/introduction)
- [ElevenLabs Streaming Guide](https://elevenlabs.io/docs/api-reference/streaming)

### SDK Documentation
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [ElevenLabs Python SDK](https://github.com/elevenlabs/elevenlabs-python)

---

**Status**: Production Ready ✅
**Last Updated**: Phase 1.8 Complete
**Version**: 1.0.0

