# API Usage Examples

## Quick Reference Guide for OpenAI and ElevenLabs Integration

---

## OpenAI API Examples

### 1. Chat Completions (GPT-4o-mini)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key="your_api_key")

# Simple chat
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### 2. Speech-to-Text (Whisper)

```python
# Transcribe audio file
with open("audio.wav", "rb") as audio_file:
    transcript = await client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

print(transcript)
```

### 3. With Conversation History

```python
conversation_history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data."},
    {"role": "user", "content": "Can you tell me a joke instead?"}
]

response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=conversation_history,
    max_tokens=150,
    temperature=0.7
)
```

### 4. Error Handling

```python
from openai import OpenAIError

try:
    response = await client.chat.completions.create(...)
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
    # Handle error
```

---

## ElevenLabs API Examples

### 1. Basic Text-to-Speech

```python
from elevenlabs.client import AsyncElevenLabs

client = AsyncElevenLabs(api_key="your_api_key")

# Generate audio
audio = await client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
    text="Hello, world!",
    model_id="eleven_multilingual_v2"
)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(audio)
```

### 2. Streaming Text-to-Speech (Recommended)

```python
from elevenlabs import VoiceSettings

# Stream audio chunks
audio_stream = await client.text_to_speech.convert_as_stream(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    text="This is a streaming example.",
    model_id="eleven_multilingual_v2",
    voice_settings=VoiceSettings(
        stability=0.5,
        similarity_boost=0.75
    ),
    output_format="mp3_44100_128"
)

# Process chunks as they arrive
async for chunk in audio_stream:
    if chunk:
        # Send to client or save
        process_chunk(chunk)
```

### 3. Custom Voice Settings

```python
from elevenlabs import VoiceSettings

# More expressive voice
expressive_settings = VoiceSettings(
    stability=0.3,           # Lower = more expressive
    similarity_boost=0.8,    # Higher = more similar to original
    style=0.5,               # Moderate style exaggeration
    use_speaker_boost=True
)

# More stable voice
stable_settings = VoiceSettings(
    stability=0.8,           # Higher = more stable
    similarity_boost=0.5,
    style=0.0,
    use_speaker_boost=False
)
```

### 4. Different Output Formats

```python
# High quality (recommended)
audio = await client.text_to_speech.convert_as_stream(
    voice_id="voice_id",
    text="High quality audio",
    output_format="mp3_44100_128"  # 44.1kHz, 128kbps
)

# Premium quality
audio = await client.text_to_speech.convert_as_stream(
    voice_id="voice_id",
    text="Premium quality audio",
    output_format="mp3_44100_192"  # 44.1kHz, 192kbps
)

# Low bandwidth
audio = await client.text_to_speech.convert_as_stream(
    voice_id="voice_id",
    text="Low bandwidth audio",
    output_format="mp3_22050_32"  # 22.05kHz, 32kbps
)
```

---

## Combined Usage (Our Implementation)

### Complete Pipeline Example

```python
import asyncio
from openai import AsyncOpenAI
from elevenlabs.client import AsyncElevenLabs
from elevenlabs import VoiceSettings

async def process_user_speech(audio_file_path: str):
    # Initialize clients
    openai_client = AsyncOpenAI(api_key="your_openai_key")
    elevenlabs_client = AsyncElevenLabs(api_key="your_elevenlabs_key")
    
    # Step 1: Speech-to-Text
    with open(audio_file_path, "rb") as audio_file:
        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    
    print(f"User said: {transcript}")
    
    # Step 2: Get AI response
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcript}
        ],
        max_tokens=150
    )
    
    ai_text = response.choices[0].message.content
    print(f"AI responds: {ai_text}")
    
    # Step 3: Text-to-Speech (streaming)
    audio_stream = await elevenlabs_client.text_to_speech.convert_as_stream(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        text=ai_text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            use_speaker_boost=True
        ),
        output_format="mp3_44100_128"
    )
    
    # Step 4: Stream audio chunks
    audio_chunks = []
    async for chunk in audio_stream:
        if chunk:
            audio_chunks.append(chunk)
            # Send to client for immediate playback
            yield chunk
    
    # Save complete audio
    full_audio = b''.join(audio_chunks)
    with open("response.mp3", "wb") as f:
        f.write(full_audio)

# Run the pipeline
asyncio.run(process_user_speech("user_audio.wav"))
```

---

## Error Handling Patterns

### Pattern 1: Retry with Exponential Backoff

```python
async def api_call_with_retry(max_retries=3):
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            result = await make_api_call()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts")
                raise
```

### Pattern 2: Graceful Degradation

```python
async def get_ai_response_safe(user_message: str) -> str:
    try:
        response = await openai_client.chat.completions.create(...)
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "I'm sorry, I'm having trouble processing that right now."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again."
```

### Pattern 3: Timeout Handling

```python
import asyncio

async def api_call_with_timeout(timeout_seconds=30):
    try:
        result = await asyncio.wait_for(
            make_api_call(),
            timeout=timeout_seconds
        )
        return result
    except asyncio.TimeoutError:
        print(f"API call timed out after {timeout_seconds}s")
        return None
```

---

## Performance Optimization Tips

### 1. Use Streaming for Lower Latency

```python
# ❌ Bad: Wait for entire response
audio = await client.text_to_speech.convert(text="Long text...")
# User waits for entire audio before hearing anything

# ✅ Good: Stream chunks
audio_stream = await client.text_to_speech.convert_as_stream(text="Long text...")
async for chunk in audio_stream:
    send_to_client(chunk)  # User hears audio immediately
```

### 2. Parallel API Calls (When Possible)

```python
# Run independent API calls in parallel
transcript_task = openai_client.audio.transcriptions.create(...)
other_task = some_other_api_call(...)

transcript, other_result = await asyncio.gather(
    transcript_task,
    other_task
)
```

### 3. Reuse Client Instances

```python
# ✅ Good: Reuse client
client = AsyncOpenAI(api_key="...")

for message in messages:
    response = await client.chat.completions.create(...)

# ❌ Bad: Create new client each time
for message in messages:
    client = AsyncOpenAI(api_key="...")  # Unnecessary overhead
    response = await client.chat.completions.create(...)
```

---

## Testing Examples

### Mock OpenAI API

```python
from unittest.mock import AsyncMock, patch

async def test_chat_completion():
    mock_response = AsyncMock()
    mock_response.choices = [
        AsyncMock(message=AsyncMock(content="Test response"))
    ]
    
    with patch('openai.AsyncOpenAI.chat.completions.create') as mock_create:
        mock_create.return_value = mock_response
        
        result = await get_ai_response("Test input")
        assert result == "Test response"
```

### Mock ElevenLabs API

```python
async def test_tts_streaming():
    async def mock_audio_stream():
        yield b"chunk1"
        yield b"chunk2"
        yield b"chunk3"
    
    with patch('elevenlabs.AsyncElevenLabs.text_to_speech.convert_as_stream') as mock_stream:
        mock_stream.return_value = mock_audio_stream()
        
        chunks = []
        async for chunk in generate_tts("Test text"):
            chunks.append(chunk)
        
        assert len(chunks) == 3
```

---

## Common Voice IDs (ElevenLabs)

| Voice Name | Voice ID | Description |
|------------|----------|-------------|
| Rachel | `21m00Tcm4TlvDq8ikWAM` | Young American female |
| Drew | `29vD33N1CtxCmqQRPOHJ` | Young American male |
| Clyde | `2EiwWnXFnvU5JabPnv8n` | Middle-aged American male |
| Paul | `5Q0t7uMcjvnagumLfvZi` | Middle-aged American male |
| Domi | `AZnzlk1XvdvUeBnXmlld` | Young American female |
| Dave | `CYw3kZ02Hs0563khs1Fj` | Young British male |

Find more at: https://elevenlabs.io/voice-library

---

## Rate Limits & Costs

### OpenAI
- **GPT-4o-mini**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens
- **Whisper**: ~$0.006 per minute of audio
- **Rate Limits**: Varies by tier (check dashboard)

### ElevenLabs
- **Free Tier**: 10,000 characters/month
- **Starter**: 30,000 characters/month (~$5)
- **Creator**: 100,000 characters/month (~$22)
- **Pro**: 500,000 characters/month (~$99)

---

## Troubleshooting

### OpenAI Issues

**Problem**: "Invalid API key"
```python
# Solution: Check environment variable
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None
```

**Problem**: "Rate limit exceeded"
```python
# Solution: Implement retry with backoff
await asyncio.sleep(60)  # Wait 1 minute
```

### ElevenLabs Issues

**Problem**: "Character limit exceeded"
```python
# Solution: Check usage
response = await client.text_to_speech.with_raw_response.convert(...)
char_cost = response.headers.get("x-character-count")
print(f"Characters used: {char_cost}")
```

**Problem**: "Audio quality poor"
```python
# Solution: Use higher quality format
output_format="mp3_44100_192"  # Instead of mp3_22050_32
```

---

## References

- [OpenAI API Docs](https://platform.openai.com/docs/api-reference)
- [ElevenLabs API Docs](https://elevenlabs.io/docs/api-reference/introduction)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [ElevenLabs Python SDK](https://github.com/elevenlabs/elevenlabs-python)

---

**Last Updated**: Phase 1.8 Complete ✅

