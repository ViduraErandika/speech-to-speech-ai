from kokoro import KPipeline
import sounddevice as sd
from threading import Thread, Event
import torch
from queue import Queue, Empty

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

        self.interrupt_event = Event()
        self.generation_thread = None
        self.playback_thread = None
        self.is_speaking = False
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
        try:
            generator = self.pipeline(text, voice=voice, speed=1, split_pattern=r'\n+')
            
            for i, (gs, ps, audio) in enumerate(generator):
                if self.interrupt_event.is_set():
                    print("Audio generation interrupted")
                    break
                self.audio_queue.put(audio)  # Add audio chunk to the queue
            
            if not self.interrupt_event.is_set():
                self.audio_queue.put(None)  # Mark end of generation
                print("Audio generation complete")
        except Exception as e:
            print(f"Error in audio generation: {str(e)}")
            if not self.interrupt_event.is_set():
                self.audio_queue.put(None) 
    
    def play_audio(self):
        try:
            self.is_speaking = True
            while not self.interrupt_event.is_set():
                try:
                    audio = self.audio_queue.get(timeout=0.5)  # Get with timeout for responsiveness
                    if audio is None:  # Check if generation is complete
                        break
                    sd.play(audio, 24000, device=3)
                    sd.wait()  # Wait for the current chunk to finish playing
                    self.audio_queue.task_done()
                except Empty:
                    continue  # No audio yet, continue checking
            
            if self.interrupt_event.is_set():
                # Clear the queue if interrupted
                if not self.audio_queue.qsize() == 0:
                    self.audio_queue.queue.clear()
        except Exception as e:
            print(f"Error in audio playback: {str(e)}")
        finally:
            self.is_speaking = False
            print("Audio playback stopped")
            
    def is_currently_speaking(self):
        """
        Check if speech is currently being generated or played.
        
        Returns:
            bool: True if speaking, False otherwise
        """
        return self.is_speaking
    
    def speak(self, text, voice):
        self.stop_speaking()
        
        # Reset the interrupt flag
        self.interrupt_event.clear()
        
        # Clear the queue
        if not self.audio_queue.qsize() == 0:
            self.audio_queue.queue.clear()
        
        # Start new generation and playback threads
        self.generation_thread = Thread(target=self.generate_audio, args=(text, voice))
        self.generation_thread.daemon = True
        self.generation_thread.start()
        
        self.playback_thread = Thread(target=self.play_audio)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        
        return True
    
    def stop_speaking(self):
        """
        Stop any ongoing speech generation and playback.
        
        Returns:
            bool: True if speech was stopped, False if nothing was playing
        """
        if self.is_speaking or (self.generation_thread and self.generation_thread.is_alive()):
            self.interrupt_event.set()  # Signal interruption
            
            # Stop any active playback
            sd.stop()
            
            # Wait for threads to clean up, but with a timeout
            if self.generation_thread and self.generation_thread.is_alive():
                self.generation_thread.join(timeout=0.5)
            
            if self.playback_thread and self.playback_thread.is_alive():
                self.playback_thread.join(timeout=0.5)
            
            self.is_speaking = False
            print("Speech stopped")
            return True
        return False
    

# if __name__ == "__main__":
#     tts = KokoroStreamingTTS()
#     text = "Success is not merely a destination but a continuous journey of growth, resilience, and learning. It requires perseverance through challenges, adaptability to change, and an unwavering belief in one's abilities. Hard work and dedication pave the path, while failures serve as valuable lessons rather than setbacks. True success is not just measured by wealth or status but by the impact one makes on others and the fulfillment derived from one's pursuits. By setting clear goals, staying committed, and maintaining a positive mindset, anyone can achieve greatness. Ultimately, success is about striving for excellence while staying true to one's values."
#     voice = "af_bella"
#     tts.speak(text, voice)
#     print("Audio playback complete")