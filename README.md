# Voice Assistant

This project implements a simple voice assistant pipeline that records audio from the microphone, transcribes it using Whisper, generates a response using GPT-2, and converts the response into speech using Edge TTS.

## Features
- Records audio from the microphone.
- Uses Faster Whisper for speech-to-text transcription.
- Generates a response using the GPT-2 model.
- Converts the response into speech using Edge TTS.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system. You may need to install `portaudio` for `pyaudio` to work properly.

For Linux:
```bash
pip install -r requirements.txt
```
## Usage
Run the main script to start the voice assistant:
```bash
python app.py
```
## How It Works
- Record Audio: The script records audio input from the microphone.
- Transcribe Audio: The recorded audio is transcribed into text using Faster Whisper.
- Generate Response: The transcribed text is passed to GPT-2, which generates a short response.
- Convert to Speech: The generated response is converted into speech and saved as output.wav.

## Requirements
- Python 3.8+
- A working microphone
- Internet connection for model downloads

## Demo
https://github.com/user-attachments/assets/e8c5aeeb-e4ab-4ea2-b1f9-7482913bc4b6

