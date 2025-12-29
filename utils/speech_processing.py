import speech_recognition as sr
import pyttsx3
import threading
import queue
import time
import os
from typing import Optional, Callable
from gtts import gTTS
import pygame
import tempfile

class SpeechProcessor:
    """Speech-to-text and text-to-speech processing"""
    
    def __init__(self, use_gtts: bool = False):
        """
        Initialize speech processor
        
        Args:
            use_gtts: Whether to use Google Text-to-Speech (requires internet)
        """
        
        # Speech Recognition setup
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.microphone_available = True
        except Exception as e:
            print(f"⚠️ Microphone initialization failed: {str(e)}")
            print("   Speech-to-text will not be available")
            self.microphone = None
            self.microphone_available = False
        
        # Text-to-Speech setup
        self.use_gtts = use_gtts
        self.tts_engine = None
        
        if not use_gtts:
            try:
                self.tts_engine = pyttsx3.init()
                self._configure_tts_engine()
            except Exception as e:
                print(f"⚠️ pyttsx3 initialization failed: {str(e)}, falling back to gTTS")
                self.use_gtts = True
        
        # Initialize pygame for audio playback (for gTTS)
        if self.use_gtts:
            try:
                pygame.mixer.init()
            except Exception as e:
                print(f"⚠️ Pygame mixer initialization failed: {str(e)}")
        
        # Audio processing queue for background operations
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.is_processing = False
        
        # Calibrate microphone
        self._calibrate_microphone()
        
        print(f"✅ Speech processor initialized (TTS: {'gTTS' if use_gtts else 'pyttsx3'})")
    
    def _configure_tts_engine(self):
        """Configure pyttsx3 TTS engine settings"""
        if self.tts_engine is None:
            return
        
        try:
            # Set speech rate
            rate = self.tts_engine.getProperty('rate')
            self.tts_engine.setProperty('rate', rate - 50)  # Slower speech
            
            # Set volume
            self.tts_engine.setProperty('volume', 0.9)
            
            # Get available voices and set to female if available
            voices = self.tts_engine.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
        except Exception as e:
            print(f"⚠️ Error configuring TTS engine: {str(e)}")
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not self.microphone_available or self.microphone is None:
            return
            
        try:
            with self.microphone as source:
                print("🔧 Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                print("✅ Microphone calibrated")
        except Exception as e:
            print(f"⚠️ Microphone calibration failed: {str(e)}")
    
    def speech_to_text(self, 
                      timeout: float = 5.0, 
                      phrase_time_limit: float = 10.0,
                      language: str = "en-US") -> Optional[str]:
        """
        Convert speech to text using Google Speech Recognition
        
        Args:
            timeout: Time to wait for speech to start
            phrase_time_limit: Maximum time for a single phrase
            language: Language code for recognition
            
        Returns:
            Recognized text or None if recognition failed
        """
        if not self.microphone_available or self.microphone is None:
            print("❌ Microphone not available for speech recognition")
            return None
            
        try:
            with self.microphone as source:
                print("🎤 Listening for speech...")
                
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                print("🔄 Processing speech...")
                
                # Use Google Speech Recognition API
                try:
                    text = self.recognizer.recognize_google(audio, language=language)
                    print(f"✅ Speech recognized: {text}")
                    return text
                    
                except sr.UnknownValueError:
                    print("❌ Could not understand audio")
                    return None
                    
                except sr.RequestError as e:
                    print(f"❌ Speech recognition service error: {str(e)}")
                    
                    # Fallback to offline recognition if available
                    try:
                        text = self.recognizer.recognize_sphinx(audio)
                        print(f"✅ Offline recognition: {text}")
                        return text
                    except:
                        return None
                
        except sr.WaitTimeoutError:
            print("⏱️ Speech recognition timeout - no speech detected")
            return None
            
        except Exception as e:
            print(f"❌ Error in speech recognition: {str(e)}")
            return None
    
    def text_to_speech(self, text: str, async_mode: bool = False) -> bool:
        """
        Convert text to speech

        Args:
            text: Text to convert to speech
            async_mode: Whether to run TTS in background

        Returns:
            Success status
        """
        if not text or text.strip() == "":
            print("⚠️ TTS: Empty text provided")
            return False

        if async_mode:
            # Add to queue for background processing
            self.audio_queue.put(('tts', text))
            self._start_audio_thread()
            print(f"✅ TTS queued asynchronously: '{text}'")
            return True
        else:
            success = self._synthesize_speech(text)
            if success:
                print(f"✅ TTS completed synchronously: '{text}'")
            else:
                print(f"❌ TTS failed synchronously: '{text}'")
            return success
    
    def _synthesize_speech(self, text: str) -> bool:
        """Internal method to synthesize speech"""
        try:
            if self.use_gtts:
                print(f"🔄 Attempting TTS with gTTS: '{text}'")
                success = self._speak_with_gtts(text)
                if success:
                    print("✅ gTTS TTS successful")
                else:
                    print("❌ gTTS TTS failed")
                return success
            else:
                print(f"🔄 Attempting TTS with pyttsx3: '{text}'")
                success = self._speak_with_pyttsx3(text)
                if success:
                    print("✅ pyttsx3 TTS successful")
                else:
                    print("❌ pyttsx3 TTS failed, falling back to gTTS")
                    # Fallback to gTTS if pyttsx3 fails
                    self.use_gtts = True
                    return self._speak_with_gtts(text)

        except Exception as e:
            print(f"❌ Unexpected error in speech synthesis: {str(e)}")
            return False
    
    def _speak_with_pyttsx3(self, text: str) -> bool:
        """Speak using pyttsx3"""
        try:
            if self.tts_engine is None:
                print("❌ pyttsx3 TTS error: Engine not initialized")
                return False

            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True

        except Exception as e:
            print(f"❌ pyttsx3 TTS error: {str(e)}")
            return False
    
    def _speak_with_gtts(self, text: str) -> bool:
        """Speak using Google Text-to-Speech"""
        try:
            # Create TTS object
            tts = gTTS(text=text, lang='en', slow=False)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                tts.save(temp_file.name)
                temp_filename = temp_file.name

            # Play audio file
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass

            return True

        except Exception as e:
            print(f"❌ gTTS error: {str(e)}")
            return False
    
    def _start_audio_thread(self):
        """Start background audio processing thread"""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
            self.audio_thread.start()
    
    def _audio_worker(self):
        """Background worker for audio processing"""
        while True:
            try:
                # Get task from queue (blocking with timeout)
                task_type, data = self.audio_queue.get(timeout=1.0)

                if task_type == 'tts':
                    print(f"🔄 Processing async TTS: '{data}'")
                    success = self._synthesize_speech(data)
                    if success:
                        print("✅ Async TTS completed successfully")
                    else:
                        print("❌ Async TTS failed")

                self.audio_queue.task_done()

            except queue.Empty:
                # No tasks in queue, continue
                continue
            except Exception as e:
                print(f"❌ Audio worker error: {str(e)}")
    
    def continuous_listen(self, 
                         callback: Callable[[str], None],
                         stop_event: threading.Event,
                         phrase_time_limit: float = 5.0) -> None:
        """
        Continuously listen for speech and call callback with recognized text
        
        Args:
            callback: Function to call with recognized text
            stop_event: Event to signal when to stop listening
            phrase_time_limit: Maximum time for each phrase
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            while not stop_event.is_set():
                try:
                    with self.microphone as source:
                        # Listen for audio
                        audio = self.recognizer.listen(
                            source, 
                            timeout=1.0,
                            phrase_time_limit=phrase_time_limit
                        )
                    
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio)
                    if text and callback:
                        callback(text)
                        
                except sr.WaitTimeoutError:
                    # No speech detected, continue
                    continue
                    
                except sr.UnknownValueError:
                    # Could not understand audio
                    continue
                    
                except Exception as e:
                    print(f"⚠️ Continuous listen error: {str(e)}")
                    time.sleep(0.1)
        
        except Exception as e:
            print(f"❌ Error in continuous listening: {str(e)}")
    
    def set_tts_properties(self, rate: Optional[int] = None, 
                          volume: Optional[float] = None,
                          voice_index: Optional[int] = None):
        """
        Set TTS engine properties
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice_index: Index of voice to use
        """
        if self.tts_engine is None:
            return
        
        try:
            if rate is not None:
                self.tts_engine.setProperty('rate', rate)
            
            if volume is not None:
                self.tts_engine.setProperty('volume', max(0.0, min(1.0, volume)))
            
            if voice_index is not None:
                voices = self.tts_engine.getProperty('voices')
                if voices and 0 <= voice_index < len(voices):
                    self.tts_engine.setProperty('voice', voices[voice_index].id)
                    
        except Exception as e:
            print(f"❌ Error setting TTS properties: {str(e)}")
    
    def get_available_voices(self) -> list:
        """Get list of available TTS voices"""
        if self.tts_engine is None:
            return []
        
        try:
            voices = self.tts_engine.getProperty('voices')
            return [(i, voice.name) for i, voice in enumerate(voices)] if voices else []
        except:
            return []
    
    def stop_tts(self):
        """Stop current TTS playback"""
        try:
            if self.use_gtts and pygame.mixer.get_init():
                pygame.mixer.music.stop()
            elif self.tts_engine:
                self.tts_engine.stop()
        except Exception as e:
            print(f"❌ Error stopping TTS: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop TTS
            self.stop_tts()
            
            # Clean up TTS engine
            if self.tts_engine:
                del self.tts_engine
            
            # Clean up pygame
            if self.use_gtts:
                pygame.mixer.quit()
            
            print("✅ Speech processor cleaned up")
            
        except Exception as e:
            print(f"❌ Error during cleanup: {str(e)}")
