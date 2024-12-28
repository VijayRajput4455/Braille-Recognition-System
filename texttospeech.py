
import pyttsx3

class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 1.0, voice_id: str = 'com.apple.speech.synthesis.voice.Alex'):
        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()
        
        # Set properties
        self.engine.setProperty('rate', rate)  # Speed of speech
        self.engine.setProperty('volume', volume)  # Volume level (0.0 to 1.0)
        
        # Set the voice (this is platform-dependent; you may need to adjust for Windows/Linux)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if voice.id == voice_id:
                self.engine.setProperty('voice', voice.id)
                break
    
    def speak(self, text: str) -> None:
        """
        Convert the given text to speech and play it.
        """
        self.engine.say(text)
        self.engine.runAndWait()

    def set_rate(self, rate: int) -> None:
        """
        Set the speech rate (words per minute).
        """
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """
        Set the speech volume level.
        """
        self.engine.setProperty('volume', volume)

    def set_voice(self, voice_id: str) -> None:
        """
        Set the voice for the speech.
        """
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if voice.id == voice_id:
                self.engine.setProperty('voice', voice.id)
                break

# Example usage
# if __name__ == "__main__":
#     tts = TextToSpeech(rate=150, volume=1.0, voice_id='com.apple.speech.synthesis.voice.Alex')
#     response = "Hello, my name is Vijay"
#     tts.speak(response)
