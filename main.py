import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import signal
import atexit
import gc
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
import nemo.collections.asr as nemo_asr
import torch
import time

# --- Configuration ---
SAMPLE_RATE = 16000
CHANNELS = 1
CHARACTERS_PER_SECOND = 150  # Typing speed for streaming effect (set to 0 for instant)

class PushToTalkApp:
    def __init__(self):
        print("Loading Model... (this may take a minute)")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        self.keyboard_controller = KeyboardController()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU.")
        else:
            print("WARNING: Running on CPU.")

        self.recording = False
        self.audio_data = []
        self.stream = None
        # Track currently pressed keys
        self.pressed_keys = set()

    def start_recording(self):
        if not self.recording:
            print("\n🔴 Recording... (Release keys to transcribe)")
            self.recording = True
            self.audio_data = []
            self.stream = sd.InputStream(samplerate=SAMPLE_RATE, 
                                         channels=CHANNELS, 
                                         callback=self.audio_callback)
            self.stream.start()

    def stop_recording(self):
        if self.recording:
            print("Processing...")
            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            if not self.audio_data:
                print("No audio recorded.")
                return

            full_audio = np.concatenate(self.audio_data, axis=0)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav.write(tmp_file.name, SAMPLE_RATE, full_audio)
                tmp_path = tmp_file.name

            try:
                transcriptions = self.model.transcribe([tmp_path])
                result = transcriptions[0]
                # Extract just the text string from the Hypothesis object
                if hasattr(result, 'text'):
                    text = result.text
                else:
                    text = str(result)
                
                print(f"🦜 Transcribed: {text}")
                
                # Small delay to allow focus to return to the text field
                time.sleep(0.1)
                
                # Type the transcribed text into the focused window with streaming effect
                self._type_with_streaming(text)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            print("\nReady. Hold [Ctrl + Space] to speak.")

    def _type_with_streaming(self, text):
        """Type text with a streaming/typewriter effect."""
        if CHARACTERS_PER_SECOND <= 0:
            # Instant typing if set to 0 or negative
            self.keyboard_controller.type(text)
        else:
            delay = 1.0 / CHARACTERS_PER_SECOND
            for char in text:
                self.keyboard_controller.type(char)
                time.sleep(delay)

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def on_press(self, key):
        self.pressed_keys.add(key)
        
        # Check if Space AND (Left Ctrl OR Right Ctrl) are pressed
        if keyboard.Key.space in self.pressed_keys and \
           (keyboard.Key.ctrl_l in self.pressed_keys or keyboard.Key.ctrl_r in self.pressed_keys):
            self.start_recording()

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key)
        except KeyError:
            pass # Key wasn't in the set, ignore
        
        # If we release Space OR Ctrl, stop recording
        if key == keyboard.Key.space or \
           key == keyboard.Key.ctrl_l or \
           key == keyboard.Key.ctrl_r:
            self.stop_recording()
            
        if key == keyboard.Key.esc:
            return False
    
    def cleanup(self):
        """Clean up GPU memory and resources."""
        print("\nCleaning up...")
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except:
                pass
        
        # Delete model and clear CUDA cache
        if hasattr(self, 'model'):
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print("Cleanup complete.")

if __name__ == "__main__":
    app = None
    
    def signal_handler(signum, frame):
        """Handle termination signals."""
        if app is not None:
            app.cleanup()
        exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        app = PushToTalkApp()
        
        # Register cleanup on normal exit
        atexit.register(app.cleanup)
        
        print("\nReady! Hold down [Ctrl + Space] to record. Press [Esc] to quit.")
        
        with keyboard.Listener(on_press=app.on_press, on_release=app.on_release) as listener:
            listener.join()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if app is not None:
            app.cleanup()