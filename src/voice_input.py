import sys
import os
import logging
import threading
import queue
import time
import json
import re
import gc
import gc
if sys.platform == 'win32':
    import winreg

# Disable Symlinks for Windows (Fixes WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Configure logging immediately
logging.basicConfig(
    filename='wispr_debug.log',
    filemode='w',
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

def log_print(msg, **kwargs):
    logging.info(msg)
    print(msg, **kwargs)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

log_print("Starting Wispr Voice Input (English Edition)...")

try:
    log_print("Importing sounddevice...")
    import sounddevice as sd
    log_print("Importing numpy...")
    import numpy as np
    log_print("Importing torch...")
    import torch
    
    log_print("Importing faster_whisper...")
    from faster_whisper import WhisperModel
    
    log_print("Importing pynput...")
    from pynput import keyboard
    
    log_print("Importing pystray...")
    import pystray
    from PIL import Image, ImageDraw
    
    # Windows Sound
    try:
        import winsound
    except ImportError:
        winsound = None

    log_print("Importing llama_cpp...")
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
    
    log_print("Importing pyperclip...")
    import pyperclip
    
    log_print("All imports successful.")
except Exception as e:
    log_print(f"CRITICAL IMPORT ERROR: {e}")
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"Critical Initialization Error:\n\n{e}", "Wispr Local Error", 0x10)
    sys.exit(1)

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 512 
VAD_THRESHOLD = 0.5 
SILENCE_DURATION_MS = 2000
MIN_SPEECH_DURATION_MS = 250
SPEECH_PAD_MS = 500

# Models
WHISPER_SIZE = "distil-medium.en" # Fast, English Only, Accurate
# WHISPER_SIZE = "base.en" # Alternative smaller model

# Llama 3.2 3B Instruct - Best for formatting/instruction following in small size
GRAMMAR_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
GRAMMAR_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


class SoundManager:
    def __init__(self, enabled=True):
        self.enabled = enabled and (winsound is not None)

    def play_start(self):
        if self.enabled:
            threading.Thread(target=winsound.Beep, args=(880, 150), daemon=True).start()

    def play_stop(self):
        if self.enabled:
            threading.Thread(target=winsound.Beep, args=(440, 150), daemon=True).start()

    def play_error(self):
        if self.enabled:
            threading.Thread(target=winsound.Beep, args=(200, 500), daemon=True).start()


class GrammarChecker:
    def __init__(self, custom_dictionary=None, dictation_prompt=None, command_prompt=None):
        self.model = None
        self.custom_dictionary = custom_dictionary or []
        self.loading_error = None
        self.dictation_prompt = dictation_prompt
        self.command_prompt = command_prompt
        self.icon = None # Placeholder

    def load_model(self):
        if self.model:
            return

        # 1. Check Local "models" folder (Offline Mode)
        local_model_path = os.path.join(os.getcwd(), "models", GRAMMAR_FILE)
        if os.path.exists(local_model_path):
            log_print(f"Found local model: {local_model_path}")
            model_path = local_model_path
        else:
            # 2. Check/Download from Hugging Face
            log_print(f"Loading Grammar Model ({GRAMMAR_REPO})...", end="", flush=True)
            try:
                # Check if model exists in cache (approximate check)
                try:
                    hf_hub_download(repo_id=GRAMMAR_REPO, filename=GRAMMAR_FILE, local_files_only=True)
                except Exception:
                    log_print("Grammar Model not found locally. Downloading... (This may take time)")
                    if self.icon:
                        self.icon.notify("Downloading Grammar Model (2GB)... Please wait.", "Wispr Setup")
                
                model_path = hf_hub_download(repo_id=GRAMMAR_REPO, filename=GRAMMAR_FILE)
            except Exception as e:
                log_print(f"\nError downloading model: {e}")
                self.loading_error = str(e)
                if self.icon:
                     self.icon.notify("Error: Cloud not download model. Check internet or place in 'models' folder.", "Wispr Error")
                return

        try:
            # CPU Fallback for Llama
            n_gpu = -1 if torch.cuda.is_available() else 0
            
            self.model = Llama(
                model_path=model_path, 
                n_ctx=2048, 
                n_gpu_layers=n_gpu, 
                verbose=False
            )
            log_print(f"Done. (GPU Layers: {n_gpu})")
        except Exception as e:
            log_print(f"\nError loading Grammar Model: {e}")
            self.loading_error = str(e)

    def correct(self, text, is_command=False):
        if not self.model or not text.strip():
            return text
            
        try:
            dict_str = ", ".join(self.custom_dictionary)
            dict_prompt = f"\nHints: {dict_str}" if dict_str else ""

            if is_command:
                # Agent Mode
                system_prompt = self.command_prompt or (
                    "You are Wispr, an intelligent assistant. Execute the user's instruction perfectly. "
                    "Output ONLY the result. Do not chat."
                )
                user_content = text
            else:
                # Cleanup Mode (English Only)
                if self.dictation_prompt:
                    system_prompt = self.dictation_prompt.replace("{dict}", dict_prompt)
                else:
                    system_prompt = (
                        "You are a strict text editing engine. Your ONLY task is to rewrite the user's text to be grammatically correct and better formatted."
                        "\n\nRULES:"
                        "\n1. Output ONLY the corrected text. Do NOT converse. Do NOT say 'Here is the corrected text'."
                        "\n2. If the input is a question, correct the grammar of the question. Do NOT answer it."
                        "\n3. If the input is a list, format it as a markdown bulleted list."
                        "\n4. Maintain the original meaning."
                        f"{dict_prompt}"
                    )
                user_content = f"Input Text: {text}\n\nCorrected Text:"

            # Llama 3 Prompt Format
            # <|start_header_id|>system<|end_header_id|>\n\n...<|eot_id|>
            # <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
            # <|start_header_id|>assistant<|end_header_id|>\n\n
            
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            
            output = self.model(
                prompt, 
                max_tokens=1024,
                stop=["<|eot_id|>"], 
                echo=False,
                temperature=0.3,
            )
            return output['choices'][0]['text'].strip()
        except Exception as e:
            log_print(f"Grammar Check Error: {e}")
            return text

    def unload_model(self):
        if self.model:
            del self.model
            self.model = None
            log_print("Grammar Model Unloaded.")

class VoiceInputApp:
    def __init__(self):
        log_print("Initializing Voice Input Application...")
        
        self.keyboard_controller = keyboard.Controller()
        
        # Load Config
        self.hotkey = keyboard.Key.f8 # Default
        self.sound_enabled = True
        self.custom_dictionary = []
        self.dictation_prompt = None
        self.command_prompt = None
        self.load_config()
        
        self.sound_manager = SoundManager(self.sound_enabled)
        
        # State
        self.q = queue.Queue()
        self.audio_buffer = [] 
        self.is_listening = False
        self.is_speaking = False
        self.running = True
        self.stream = None
        self.mic_active = False
        
        self.models_ready = False
        self.loading_status = "Initializing..."
        
        # Initialize Placeholders
        self.grammar_checker = GrammarChecker(
            self.custom_dictionary, 
            dictation_prompt=self.dictation_prompt, 
            command_prompt=self.command_prompt
        )
        self.grammar_checker.icon = None # Will assign later
        self.vad_model = None
        self.asr_model = None
        self.vad_iterator = None
        
        # Tray Icon (placeholder)
        self.icon = None

        # VRAM Saver State
        self.last_activity_time = time.time()
        self.heavy_models_loaded = False
        self.model_lock = threading.Lock()
        self.vram_timeout = 60 # Seconds before unloading

        # Start loading threads
        threading.Thread(target=self.initial_load, daemon=True).start()

    def initial_load(self):
        self.loading_status = "Loading VAD..."
        self.update_tray_tooltip()
        self.load_vad()
        
        # We load heavy models initially so it's ready for first use, 
        # then let the saver handle unloading if unused.
        self.load_heavy_models()

    def load_vad(self):
        # 1. Load VAD Model (Silero)
        log_print("Loading Silero VAD...", end="", flush=True)
        try:
            self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                  model='silero_vad',
                                                  force_reload=False,
                                                  onnx=False)
            (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = utils
            self.vad_iterator = self.VADIterator(self.vad_model, 
                                                 threshold=VAD_THRESHOLD, 
                                                 sampling_rate=SAMPLE_RATE, 
                                                 min_silence_duration_ms=SILENCE_DURATION_MS, 
                                                 speech_pad_ms=SPEECH_PAD_MS)
            log_print("Done.")
        except Exception as e:
            log_print(f"\nError loading VAD: {e}")
            self.loading_status = "Error Loading VAD"
            self.update_tray_tooltip()
            return

    def load_heavy_models(self):
        with self.model_lock:
            if self.heavy_models_loaded:
                return

            log_print("Loading Heavy Models (Wake up)...")
            self.loading_status = "Loading Models..."
            self.update_tray_tooltip()

            # Load Grammar
            self.grammar_checker.load_model()

            # Load Faster-Whisper
            # Load Faster-Whisper
            log_print(f"Loading Faster-Whisper ({WHISPER_SIZE})...", end="", flush=True)
            
            # Simple check if model exists in cache (limited, but helpful)
            # Faster-whisper handles downloads internally, but we can warn user if it might be slow.
            if self.icon:
                 self.icon.title = "Wispr: Loading AI Models..."
            
            try:
                if torch.cuda.is_available():
                    device_str = "cuda"
                    compute_type = "float16" 
                else:
                    device_str = "cpu"
                    compute_type = "int8" # CPU usually needs int8 for speed or int8_float32

                self.asr_model = WhisperModel(
                    WHISPER_SIZE, 
                    device=device_str, 
                    compute_type=compute_type
                )
                log_print(f"Done. (Device: {device_str}, Compute: {compute_type})")
            except Exception as e:
                log_print(f"\nError loading Whisper: {e}")
                self.loading_status = "Error Loading ASR"
                self.update_tray_tooltip()
                return

            self.models_ready = True # VAD + Heavy loaded
            self.heavy_models_loaded = True
            self.loading_status = "Ready"
            self.update_status("READY")
            log_print(f"Acceleration Status: {'GPU ENABLED' if torch.cuda.is_available() else 'CPU MODE'}")
            self.sound_manager.play_start() 

    def unload_heavy_models(self):
        with self.model_lock:
            if not self.heavy_models_loaded:
                return
            
            log_print("Unloading Models (VRAM Saver)...")
            self.asr_model = None
            self.grammar_checker.unload_model()
            
            # Force Garbage Collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.heavy_models_loaded = False
            self.loading_status = "Idle (VRAM Free)"
            self.update_tray_tooltip()
            log_print("Models Unloaded. VRAM released.")

    def load_config(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
            else:
                base_path = os.getcwd()
                
            config_path = os.path.join(base_path, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    hotkey_str = config.get("hotkey", "f8").lower()
                    self.sound_enabled = config.get("sound_enabled", True)
                    self.custom_dictionary = config.get("custom_dictionary", [])
                    self.vram_timeout = config.get("vram_timeout", 60)
                    self.dictation_prompt = config.get("dictation_prompt", None)
                    self.command_prompt = config.get("command_prompt", None)
                    
                    # Model overrides
                    global WHISPER_SIZE, GRAMMAR_REPO, GRAMMAR_FILE
                    WHISPER_SIZE = config.get("whisper_model", WHISPER_SIZE)
                    GRAMMAR_REPO = config.get("grammar_repo", GRAMMAR_REPO)
                    GRAMMAR_FILE = config.get("grammar_file", GRAMMAR_FILE)
                    
                    if hotkey_str in keyboard.Key.__members__:
                        self.hotkey = keyboard.Key[hotkey_str]
                    else:
                        self.hotkey = keyboard.KeyCode.from_char(hotkey_str)
                    
                    log_print(f"Loaded Config - Hotkey: {self.hotkey}, Sound: {self.sound_enabled}, Timeout: {self.vram_timeout}s")
            else:
                log_print(f"config.json not found at {config_path}, using defaults")
        except Exception as e:
            log_print(f"Error loading config: {e}. Using default.")

    def create_icon(self, color):
        icon_path = resource_path("assets/icon.png")
        if os.path.exists(icon_path):
            try:
                base_image = Image.open(icon_path)
                # If we want to add a status overlay, we can do it here
                # For now, let's just return the image or a colored variant
                return base_image
            except Exception as e:
                log_print(f"Error loading icon file: {e}")

        # Fallback to dynamic drawing if file not found
        width = 64
        height = 64
        image = Image.new('RGB', (width, height), (0, 0, 0))
        dc = ImageDraw.Draw(image)
        dc.ellipse((4, 4, 60, 60), outline="white", width=2)
        dc.ellipse((14, 14, 50, 50), fill=color)
        return image

    def update_tray_tooltip(self):
        if self.icon:
            self.icon.title = f"Wispr: {self.loading_status}"

    def update_status(self, status):
        # status: READY, RECORDING, PROCESSING, ERROR, LOADING
        if not self.icon:
            return
            
        if not self.models_ready:
            self.icon.icon = self.create_icon("grey")
            self.update_tray_tooltip()
            return

        if status == "READY":
            self.icon.icon = self.create_icon("cyan")
            self.icon.title = "Wispr: Ready (F8)"
        elif status == "RECORDING":
            self.icon.icon = self.create_icon("red")
            self.icon.title = "Wispr: Listening..."
        elif status == "PROCESSING":
            self.icon.icon = self.create_icon("yellow")
            self.icon.title = "Wispr: Processing..."
        elif status == "ERROR":
            self.icon.icon = self.create_icon("orange")
            self.icon.title = "Wispr: Error/No Mic"

    def audio_callback(self, indata, frames, time, status):
        if self.running and self.mic_active and self.models_ready:
            self.q.put(indata.copy())

    def on_press(self, key):
        if key == self.hotkey:
            # Wake up detection
            if not self.heavy_models_loaded:
                log_print("Wake up detected. Pre-loading models...")
                threading.Thread(target=self.load_heavy_models, daemon=True).start()

            if not self.vad_model: # VAD must be loaded at least
                return

            if not self.mic_active:
                log_print("Ignored F8: No Microphone Active")
                self.sound_manager.play_error()
                return
                
            if not self.is_listening:
                self.start_listening()
            else:
                self.stop_listening()

    def start_listening(self):
        self.last_activity_time = time.time()
        log_print("\n[Start Listening]", flush=True)
        self.sound_manager.play_start()
        self.is_listening = True
        self.is_speaking = False
        self.audio_buffer = []
        if self.vad_iterator:
            self.vad_iterator.reset_states()
        self.update_status("RECORDING")

    def stop_listening(self):
        log_print(" [Stopped]", flush=True)
        self.sound_manager.play_stop()
        self.is_listening = False
        self.update_status("PROCESSING")
        
        if len(self.audio_buffer) > 0:
            audio_segment = np.array(self.audio_buffer)
            self.transcribe(audio_segment)
        else:
             self.update_status("READY")
             
        self.audio_buffer = []
        self.last_activity_time = time.time()

    def transcribe(self, audio_data):
        duration = len(audio_data) / SAMPLE_RATE
        if duration < (MIN_SPEECH_DURATION_MS / 1000):
            log_print(f" [Audio too short: {duration:.2f}s - Ignored]")
            self.update_status("READY")
            return

        max_amp = np.max(np.abs(audio_data))
        if max_amp < 0.001: 
            log_print(f" [Audio too quiet (Max Amp: {max_amp:.4f}) - Ignored]")
            self.update_status("READY")
            return

        # Ensure models are loaded before transcribing
        if not self.heavy_models_loaded:
            log_print("Waiting for models to load...")
            self.load_heavy_models()

        log_print(" Transcribing (Faster-Whisper)...", flush=True)
        t0 = time.time()
        try:
            # Faster-Whisper expects float32 array
            # No need for complex VAD merging here as we handle that upstream or pass single segment
            segments, info = self.asr_model.transcribe(
                audio_data.astype(np.float32), 
                beam_size=5,
                language="en", # Enforce English
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            raw_text = " ".join([segment.text for segment in segments]).strip()
            
            t1 = time.time()
            log_print(f" [ASR Time: {t1 - t0:.3f}s] Raw: {raw_text}")
            
            if not raw_text:
                self.update_status("READY")
                return

            # --- Logic for Command Mode (DISABLED FOR NOW) ---
            is_command = False
            command_text = raw_text
            
            # if raw_text.lower().startswith("wispr") or raw_text.lower().startswith("whisper"):
            #     is_command = True
            #     command_text = re.sub(r'^(wispr|whisper)\s*,?\s*', '', raw_text, flags=re.IGNORECASE)
            #     log_print(f" [Command Mode Detected] Input: {command_text}")
            
            log_print(" Refining format (Llama 3.2)...")
            t2 = time.time()
            final_text = self.grammar_checker.correct(command_text, is_command=is_command)
            t3 = time.time()
            log_print(f" [Grammar Time: {t3 - t2:.3f}s]")
            
            log_print(f"Output: {final_text}")
            log_print(f" [Total Time: {t3 - t0:.3f}s]")
            
            try:
                self.paste_text(final_text)
            except Exception as e:
                log_print(f"Typing Error: {e}")
                self.sound_manager.play_error()
                
        except Exception as e:
            log_print(f"ASR Error: {e}")
            self.sound_manager.play_error()
        finally:
            self.update_status("READY")

    def paste_text(self, text):
        try:
            original_clipboard = pyperclip.paste()
            pyperclip.copy(text)
            time.sleep(0.05) 
            with self.keyboard_controller.pressed(keyboard.Key.ctrl):
                self.keyboard_controller.press('v')
                self.keyboard_controller.release('v')
            time.sleep(0.2) 
            pyperclip.copy(original_clipboard)
        except Exception as e:
            log_print(f"Paste Error: {e}")
            self.keyboard_controller.type(text)

    def start_audio_stream(self):
        try:
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE
            )
            self.stream.start()
            self.mic_active = True
            log_print("Microphone Stream Started.")
        except Exception as e:
            log_print(f"Microphone Error: {e}")
            self.mic_active = False
            self.update_status("ERROR")
            self.sound_manager.play_error()

    def processing_loop(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        self.start_audio_stream()
            
        while self.running:
            # VRAM Saver Check
            if self.heavy_models_loaded and not self.is_listening:
                if (time.time() - self.last_activity_time) > self.vram_timeout:
                    self.unload_heavy_models()

            try:
                try:
                    chunk = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                    
                if not self.is_listening:
                    continue
                    
                chunk = chunk.flatten()
                self.audio_buffer.extend(chunk)
                
                # Check VAD for Manual Toggle Feedback (Optional)
                if self.vad_iterator:
                    chunk_tensor = torch.from_numpy(chunk).float()
                    speech_dict = self.vad_iterator(chunk_tensor, return_seconds=True)
                    if speech_dict and 'start' in speech_dict and not self.is_speaking:
                         self.is_speaking = True
                            
            except Exception as e:
                log_print(f"Loop Error: {e}")
                time.sleep(1)

    def exit_action(self, icon, item):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        icon.stop()
        os._exit(0)

    def reconnect_action(self, icon, item):
        log_print("User requested audio reconnect...")
        self.start_audio_stream()

    def toggle_startup(self, icon, item):
        if sys.platform != 'win32':
            log_print("Auto-launch is only supported on Windows.")
            return

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        app_name = "WisprLocal"
        # If frozen, use executable path. If script, use python + script path (less reliable for auto-start without wrapper)
        # But user wants this for the built exe mainly.
        exe_path = sys.executable 
        
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_ALL_ACCESS)
            if item.checked: # Currently checked, so turn OFF
                 try:
                    winreg.DeleteValue(key, app_name)
                    log_print("Auto-launch disabled.")
                 except FileNotFoundError:
                    pass
            else: # Currently unchecked, so turn ON
                 winreg.SetValueEx(key, app_name, 0, winreg.REG_SZ, exe_path)
                 log_print(f"Auto-launch enabled. Path: {exe_path}")
            winreg.CloseKey(key)
        except Exception as e:
            log_print(f"Error checking startup status: {e}")

    def check_startup_status(self, item):
        if sys.platform != 'win32':
            return False
            
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        app_name = "WisprLocal"
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ)
            winreg.QueryValueEx(key, app_name)
            winreg.CloseKey(key)
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def run(self):
        # Setup Tray Menu
        menu = pystray.Menu(
            pystray.MenuItem('Run at Startup', self.toggle_startup, checked=self.check_startup_status),
            pystray.MenuItem('Reconnect Audio', self.reconnect_action),
            pystray.MenuItem('Exit', self.exit_action)
        )
        
        # Create Icon
        image = self.create_icon("cyan")
        self.icon = pystray.Icon("Wispr Voice Input", image, "Wispr: Initializing...", menu)
        
        # Pass icon to grammar checker for notifications immediately
        self.grammar_checker.icon = self.icon

        print("System Tray started. Check your taskbar.")
        
        # Start Processing Thread
        t = threading.Thread(target=self.processing_loop, daemon=True)
        t.start()
        
        # Run Icon (Native Loop)
        self.icon.run()

if __name__ == "__main__":
    app = VoiceInputApp()
    app.run()
