import torch
from time import sleep
import numpy as np
import speech_recognition as sr
import queue
import threading
import time
import os
from datetime import datetime, timedelta
from huggingface_hub import login
from transformers import AutoProcessor, AutoModelForSeq2SeqLM

class UltravoxRealtimeInterface:
    def __init__(self):
        """Initialize the Ultravox model and settings"""
        print("Loading Ultravox model... This may take a few minutes...")
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
        
        # CUDA check
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"Using device: {self.device}")

        # Load the Qwen2-Audio model and processor
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct").to(self.device)
        print("Model and processor loaded successfully!")
        
        # Audio settings
        self.sample_rate = 16000
        
        # Default system prompt
        self.default_prompt = "You are a friendly and helpful character. You love to answer questions for people."
        
        # Audio recording setup
        self.data_queue = queue.Queue()
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        self.recording = False
        self.source = None
        self.recording_thread = None
        
    def start_recording(self):
        """Start recording from microphone"""
        if self.recording:
            return "Already recording. Please stop first."
        
        try:
            self.source = sr.Microphone(sample_rate=self.sample_rate)
            
            with self.source as source:
                self.recorder.adjust_for_ambient_noise(source)
            
            self.recording = True
            
            def record_callback(_, audio):
                if self.recording:
                    data = audio.get_raw_data()
                    self.data_queue.put(data)
            
            self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=3)
            
            return "Recording started. Speak now..."
        except Exception as e:
            return f"Error starting recording: {str(e)}"
    
    def process_recording(self, system_prompt):
        try:
            # Process all collected audio
            if self.data_queue.empty():
                return "No audio detected. Please try again."
            
            # Collect all audio data from queue
            audio_data = b''.join(list(self.data_queue.queue))
            self.data_queue.queue.clear()
            
            # Convert to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Prepare the input for the model
            inputs = self.processor(
                audio=audio_np,
                sampling_rate=self.sample_rate,
                text=system_prompt if system_prompt else self.default_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            
            # Decode the output
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return response
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    

def main():
    # login() 
    app = UltravoxRealtimeInterface()
    system_prompt = "You are a friendly and helpful character. You love to answer questions for people."
    app.start_recording()
    phrase_time = None
    while True:
        try:
            now = datetime.utcnow()
            if not app.data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=3):
                    phrase_complete = True
                phrase_time = now
                print("Processing audio...")
                result = app.process_recording(system_prompt)
                print(result)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()