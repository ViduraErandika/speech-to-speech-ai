# Voice Assistant Implementation with Ultravox and Kokoro

This project implements a real-time voice assistant using Ultravox for speech recognition and response generation and Kokoro for text-to-speech output. It creates a complete voice interaction system that can handle spoken queries and provide audible responses.

## Features

- Real-time speech recognition with microphone input
- AI-powered responses using Ultravox and LLama models
- Natural-sounding text-to-speech using Kokoro TTS
- Conversation history tracking
- Interrupt handling (new audio input can interrupt current speech)
- Voice character customization

## Project Structure

- `conversation_history.py`: Class for tracking and managing conversation history
- `test_qwen.py`: Implementation using the Qwen2-Audio model
- `test_uv.py`: Implementation using the Ultravox model
- `test_uv_kokoro.py`: Complete implementation with Ultravox and Kokoro TTS
- `tts.py`: Kokoro streaming text-to-speech implementation

## Setup

### Prerequisites

Required Python packages are listed in `req.txt`:

```
transformers
numpy
SpeechRecognition
PyAudio
peft
kokoro>=0.9.2
sounddevice
ipython
```

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r req.txt
   ```
3. Ensure you have the appropriate audio input/output devices configured

## Usage

To run the complete voice assistant implementation:

```
python test_uv_kokoro.py
```

The system will:
1. Load the Ultravox model
2. Initialize the Kokoro TTS engine
3. Start recording from your microphone
4. Process your speech and generate responses
5. Output the responses through text-to-speech

You can customize the system prompt in `test_uv_kokoro.py` to change the assistant's behavior and knowledge base.

## Components

### ConversationHistory

Tracks the conversation between the user and the assistant, maintaining a history of inputs and responses for context-aware interactions.

### UltravoxRealtimeInterface

Handles audio recording, speech recognition, and response generation using either the Ultravox model or Qwen2-Audio model.

### KokoroStreamingTTS

Provides text-to-speech functionality with sentence-level streaming for natural-sounding output with minimal delay.

## Model Information

- **Ultravox**: `fixie-ai/ultravox-v0_5-llama-3_2-1b` - A speech-to-text-to-speech model
- **Qwen2-Audio**: `Qwen/Qwen2-Audio-7B-Instruct` - An alternative speech understanding model
- **Kokoro TTS**: A high-quality text-to-speech engine

## Customization

You can customize the assistant by modifying the system prompt in the `main()` function. The current implementation provides information about SLT broadband packages, but you can adapt it to any use case.

## Voice Options

The default voice is set to "af_bella", but you can change this in the `main()` function to use other available Kokoro voices.

## License

This project is provided as an example implementation and is meant for educational purposes.
