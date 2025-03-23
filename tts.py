from kokoro import KPipeline
import sounddevice as sd
from threading import Thread
import torch
from queue import Queue

class KokoroStreamingTTS:
    """
    A class for streaming text-to-speech using Kokoro TTS.
    
    This class handles concurrent text-to-speech generation and audio playback,
    creating a seamless streaming experience with minimal delay.
    """
    
    def __init__(self, lang_code='a', device=None):
        """
        Initialize the KokoroStreamingTTS with the specified language code.
        
        Args:
            lang_code (str): Language code ('a' for American English)
            device (str): Device to use for inference ('cuda' or 'cpu')
        """
        # Initialize the pipeline
        self.pipeline = KPipeline(lang_code=lang_code)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device for TTS: {self.device}")
        else:
            self.device = device
            print(f"Using device for TTS: {self.device}")
            
        self.audio_queue = Queue()
        print("KokoroStreamingTTS initialized")
    
    def split_into_sentences(self, text):
        """
        Split text into sentences for more natural streaming.
        
        Args:
            text (str): The input text to split
            
        Returns:
            list: A list of sentence strings
        """
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
                
        if len(current.strip()) > 0:
            sentences.append(current.strip())
            
        return sentences
    
    def generate_audio(self, text, voice):
        """
        Thread function to generate audio from text.
        
        Args:
            text (str): The text to convert to speech
            voice (str): The voice to use
        """
        generator = self.pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
        
        for i, (gs, ps, audio) in enumerate(generator):
            self.audio_queue.put(audio)  # Add audio chunk to the queue
        self.audio_queue.put(None) 
        print("Audio generation complete")
    
    def play_audio(self):
        while True:
            audio = self.audio_queue.get()  # Get the next audio chunk
            if audio is None:  # Check if generation is complete
                break
            sd.play(audio, 24000, device=3)
            sd.wait()  # Wait for the current chunk to finish playing
            

    
    def speak(self, text, voice):
        generation_thread = Thread(target=self.generate_audio, args=(text, voice))
        generation_thread.start()

        # Start playing audio immediately
        self.play_audio()

        # Wait for the generation thread to finish
        generation_thread.join() 
        return True
    

# if __name__ == "__main__":
#     tts = KokoroStreamingTTS()
#     text = "Success is not merely a destination but a continuous journey of growth, resilience, and learning. It requires perseverance through challenges, adaptability to change, and an unwavering belief in one's abilities. Hard work and dedication pave the path, while failures serve as valuable lessons rather than setbacks. True success is not just measured by wealth or status but by the impact one makes on others and the fulfillment derived from one's pursuits. By setting clear goals, staying committed, and maintaining a positive mindset, anyone can achieve greatness. Ultimately, success is about striving for excellence while staying true to one's values."
#     voice = "af_bella"
#     tts.speak(text, voice)
#     print("Audio playback complete")