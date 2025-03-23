import transformers
import torch
from time import sleep
import numpy as np
import speech_recognition as sr
import queue
from datetime import datetime, timedelta
from tts import KokoroStreamingTTS
from conversation_history import ConversationHistory

class UltravoxRealtimeInterface:
    def __init__(self):
        """Initialize the Ultravox model and settings"""
        print("Loading Ultravox model... This may take a few minutes...")
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"CUDA is available: {torch.cuda.is_available()}")
        print(f"Using device: {self.device}")

        # Explicitly set dtype to torch.float16 for better compatibility
        self.pipe = transformers.pipeline(
            model='fixie-ai/ultravox-v0_5-llama-3_2-1b',
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")
        
        # Audio settings
        self.sample_rate = 16000
        
        # Default system prompt
        self.default_prompt = "You are a friendly and helpful character. You love to answer questions for people."
        
        # Audio recording setup
        self.data_queue = queue.Queue()
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 2000
        self.recorder.dynamic_energy_threshold = False
        self.recording = False
        self.source = None
        self.recording_thread = None
        
        self.conversation_history = ConversationHistory()
        
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
            
            # Prepare conversation turns
            turns = [
                {
                    "role": "system",
                    "content": system_prompt if system_prompt else self.default_prompt
                }
            ]

            turns.extend(self.conversation_history.get_history())
            
            # Get model response
            result = self.pipe(
                {   
                    'audio': audio_np,
                    'turns': turns,
                    'sampling_rate': self.sample_rate
                },
                max_new_tokens=100
            )
            
            if isinstance(result, str):
                model_response = result
            elif isinstance(result, list):
                model_response = result[0] if result else "No response generated"
            elif isinstance(result, dict):
                model_response = result.get('generated_text', "No response generated")
            else:
                model_response = str(result)
            
            # Add model response to history
            self.conversation_history.add_model_response(model_response)
            
            return model_response
            
        except Exception as e:
            return f"Error processing audio: {str(e)}"
    
    def stop_recording(self):
        """Stop recording from microphone"""
        if not self.recording:
            return "Not currently recording."
        
        self.recording = False
        self.data_queue.queue.clear()
        return "Recording stopped."


def main():
    # login() 
    app = UltravoxRealtimeInterface()
    tts = KokoroStreamingTTS()
    voice = "af_bella"
    system_prompt = (
    """You are an AI assistant with access to the following SLT broadband package details. Output should be only in Paragraph without special characters such as asterisks, underscores, or brackets. Keep the output clean and short.:
"""
)
    app.start_recording()
    tts.speak("Hi How can i help you today?", voice)
    # phrase_time = None
    try:
        while True:
            now = datetime.utcnow()
            print("Checking for audio...")
            if not app.data_queue.empty():
                # phrase_complete = False
                # if phrase_time and now - phrase_time > timedelta(seconds=3):
                #     phrase_complete = True
                # phrase_time = now
                print("Processing audio...")
                result = app.process_recording(system_prompt)
                print(result)
                tts.speak(result, voice)
                app.data_queue.queue.clear()
                sleep(0.25)
            else:
                sleep(0.25)
    except KeyboardInterrupt:
        print("Stopping recording...")
        app.stop_recording()
        print("Exiting...")


if __name__ == "__main__":
    main()