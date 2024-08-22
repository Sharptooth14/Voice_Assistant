import pyaudio
import wave
from faster_whisper import WhisperModel
import numpy as np
import edge_tts

from transformers import pipeline
import asyncio

#Take input from microphone

def record_audio(output_filename, duration=5, sample_rate=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    frames = []

    print("Recording...")
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

#Transcribe the given input

def transcribe_audio(filename="input_audio.wav"):
    model = WhisperModel("base", device="cpu")

    segments, info = model.transcribe(filename)

    transcribed_text = " ".join([segment.text for segment in segments])

    
    return transcribed_text

#Prompt the transcribed text to the llm 
def generate_response(transcribed_text):

    generator = pipeline("text-generation",device ="cpu", model="gpt2") 
    

    prompt = (
        "You are a voice assistant. You will answer the following given prompt in two sentences. "
        "Please respond to the following prompt: " + transcribed_text
    )
    response = generator(prompt, max_length=50, num_return_sequences=1)
    
    generated_text = response[0]['generated_text'][len(prompt):].strip()
    
    return generated_text

#convert the response into speech

async def text_to_speech(text):
    communicate = edge_tts.Communicate(text, 'en-US-AriaNeural')
    await communicate.save("output.wav")


def main():
    audio_file = "input_audio.wav"
    record_audio(audio_file, duration=3)

    transcribed_text = transcribe_audio()

    response_text = generate_response(transcribed_text)

    print("Transcribed text:", transcribed_text)

    print("Generated response:", response_text)

    asyncio.run(text_to_speech(response_text))

if __name__ == "__main__":
    main()