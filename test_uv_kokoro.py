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
        
        self.is_processing = False
        self.user_interrupted = False

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
                    # Set the user_interrupted flag when new audio is detected
                    if self.is_processing:
                        self.user_interrupted = True
            
            self.recorder.listen_in_background(self.source, record_callback, phrase_time_limit=10)
            
            return "Recording started. Speak now..."
        except Exception as e:
            return f"Error starting recording: {str(e)}"
    
    def process_recording(self, system_prompt):
        
        if self.is_processing:
            print("Already processing audio. Skipping...")
            return None
        
        self.is_processing = True
        try:
            # Process all collected audio
            if self.data_queue.empty():
                self.is_processing = False
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
        finally:
            self.is_processing = False
            self.user_interrupted = False
    
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
    SLTMobitel offers a variety of broadband packages tailored to different user needs. Below is an overview of their offerings:

    Trio Packages (Data, PEOTV, and Voice):

    Trio Vibe: 40 GB Anytime Data & 40 GB Anytime Upload, PEOTV with 75 channels, and unlimited calls. Monthly rental: Rs. 3,530.
    Trio Vibe Plus: 40 GB Anytime Data & 40 GB Anytime Upload, PEOTV with 90 channels, and unlimited calls. Monthly rental: Rs. 4,100.
    Trio Shine: 100 GB Anytime Data, 25 GB 7xFun Bundle, PEOTV with 75 channels, unlimited calls, 50 GB Eazy Storage, and unlimited anytime uploads. Monthly rental: Rs. 4,950.
    Data Packages:

    Higher Education 1: 4 GB standard data with 6 GB free data. Monthly rental: Rs. 490.
    Web Lite: 6 GB standard data with 9 GB free data. Monthly rental: Rs. 590.
    Entree: 7 GB anytime data. Monthly rental: Rs. 650.
    Web Starter: 11 GB standard data with 17 GB free data. Monthly rental: Rs. 950.
    Higher Education 2: 12 GB standard data with 18 GB free data. Monthly rental: Rs. 990.
    Any Joy: 22 GB anytime data. Monthly rental: Rs. 1,150.
    Web Pal: 18 GB standard data with 27 GB free data. Monthly rental: Rs. 1,250.
    Any Beat: 36 GB anytime data. Monthly rental: Rs. 1,550.
    Web Family Plus: 36 GB standard data with 54 GB free data. Monthly rental: Rs. 1,790.
    Any Flix: 53 GB anytime data. Monthly rental: Rs. 2,150.
    Unlimited LTE Flash 5: Daily 5 GB at standard LTE speed with unlimited access to Zoom, Meet, Teams, and SLT-MOBITEL Linked. Monthly rental: Rs. 2,350.
    Web Family Active: 45 GB standard data with 65 GB free data. Monthly rental: Rs. 2,390.
    LTE Unlimited 2: Unlimited data up to 2 Mbps speed. Monthly rental: Rs. 2,950.
    Any Blaze: 72 GB anytime data. Monthly rental: Rs. 2,990.
    Web Family Xtra: 60 GB standard data with 90 GB free data. Monthly rental: Rs. 3,090.
    ADSL Unlimited 2: Unlimited data up to 2 Mbps speed. Monthly rental: Rs. 3,190.
    Any Tide: 100 GB anytime data. Monthly rental: Rs. 3,890.
    Web Booster: 75 GB standard data with 110 GB free data. Monthly rental: Rs. 4,050.
    Unlimited LTE Flash 10: Daily 10 GB at standard LTE speed with unlimited access to Zoom, Meet, Teams, and SLT-MOBITEL Linked. Monthly rental: Rs. 4,090.
    Any Spike: 130 GB anytime data. Monthly rental: Rs. 5,050.
    LTE Unlimited 4: Unlimited data up to 4 Mbps speed. Monthly rental: Rs. 5,150.
    Web Pro: 95 GB standard data with 140 GB free data. Monthly rental: Rs. 5,650.
    ADSL Unlimited 4: Unlimited data up to 4 Mbps speed. Monthly rental: Rs. 5,790.
    Fibre Unlimited Flash 10: Daily 10 GB up to 300 Mbps (thereafter 1 Mbps) with unlimited access to Zoom, Meet, Teams, and SLT-MOBITEL Linked. Monthly rental: Rs. 5,990.
    LTE Unlimited 8: Unlimited data up to 8 Mbps speed. Monthly rental: Rs. 7,090.
    ADSL Unlimited 8: Unlimited data up to 8 Mbps speed. Monthly rental: Rs. 7,490.
    Any Storm: 200 GB anytime data. Monthly rental: Rs. 7,490.
    Fibre Unlimited 10: Unlimited data up to 20 Mbps download speed and 10 Mbps upload speed. Monthly rental: Rs. 7,750.
    Web Master: 170 GB standard data with 255 GB free data. Monthly rental: Rs. 9,790.
    Fibre Unlimited Flash 25: Daily 25 GB up to 300 Mbps (thereafter 1 Mbps) with unlimited access to Zoom, Meet, Teams, and SLT-MOBITEL Linked. Monthly rental: Rs. 10,390.
    Any Glam: 350 GB anytime data. Monthly rental: Rs. 12,990.
    Fibre Unlimited 25: Unlimited data up to 40 Mbps download speed and 20 Mbps upload speed. Monthly rental: Rs. 13,590.
    Web Champ: 270 GB standard data with 405 GB free data. Monthly rental: Rs. 15,090.
    Ultra Flash Elite: 40 GB per day at 1 Gbps* speed. Monthly rental: Rs. 16,500.
    Ultra Elite: 500 GB at 1 Gbps* speed with unlimited YouTube access. Monthly rental: Rs. 16,500.
    Any Delight: 500 GB anytime data. Monthly rental
    Instruction:
    When a user queries about any specific broadband package, speed, price, data, or additional features, answer concisely based on the provided details. If the query is incomplete or ambiguous, politely ask the user for clarification.   
    """
)
    app.start_recording()
    tts.speak("Hi How can i help you today?", voice)
    # phrase_time = None
    try:
        while True:
            # print("Checking for audio...")

            if app.user_interrupted and tts.is_currently_speaking():
                print("User interrupted! Stopping current speech...")
                tts.stop_speaking() 

            if not app.data_queue.empty() and not app.is_processing:
                print("Processing audio...")
                result = app.process_recording(system_prompt)
                if result:
                    print(result)
                    tts.speak(result, voice)
                    app.data_queue.queue.clear()
            
            # Short sleep to prevent CPU hogging
            sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping recording...")
        app.stop_recording()
        tts.stop_speaking()
        print("Exiting...")


if __name__ == "__main__":
    main()