import sys
import os
import logging
import threading
import queue
import time
import json
import re
import gc
if sys.platform == 'win32':
    import winreg

# Disable Symlinks for Windows (Fixes WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Configure logging
# Default to console only unless PRIVOX_DEBUG=1 is set
log_level = logging.INFO
log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'

if os.environ.get("PRIVOX_DEBUG") == "1":
    logging.basicConfig(
        filename='privox_app.log',
        filemode='a',
        format=log_format,
        datefmt=log_datefmt,
        level=log_level,
        force=True
    )
else:
    # Console only logging - explicitly use StreamHandler to avoid any default file behavior
    logging.basicConfig(
        format=log_format,
        datefmt=log_datefmt,
        level=log_level,
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

# Silence noisy external loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Redirect stdout/stderr to logging to avoid 'NoneType' has no attribute 'write' in --noconsole mode
class LoggerWriter:
    def __init__(self, level):
        self.level = level
    def write(self, message):
        if message.strip():
            self.level(message.strip())
    def flush(self):
        pass

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

def log_print(msg, **kwargs):
    # Only print to stdout. sys.stdout is already redirected to logging.info
    # This prevents duplicate log entries.
    print(msg, **kwargs)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

log_print("Starting Privox...")

# --- 1. System Diagnostics & Path Prioritization ---
try:
    import sys
    import os
    import logging
    
    # We MUST ensure standard libraries are reachable BEFORE we clobber sys.path
    log_print(f"System Diagnostic - Python Interpreter: {sys.executable}")
    log_print(f"System Diagnostic - sys.prefix: {sys.prefix}")
    
    # Check for core modules
    try:
        import timeit
        import json
        import re
        log_print("System Diagnostic - Standard libraries verified.")
    except ImportError as e:
        log_print(f"CRITICAL SYSTEM ERROR: Standard library missing: {e}")
        # If standard libs are missing, something is wrong with the Python install.
        # We'll try to add the default Lib paths if we can guess them.
        lib_path = os.path.join(sys.prefix, "Lib")
        if os.path.exists(lib_path) and lib_path not in sys.path:
            sys.path.append(lib_path)
    
except Exception as e:
    print(f"Boot Error: {e}")

# Base Directory for models/libs
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if BASE_DIR.endswith('src'):
         BASE_DIR = os.path.dirname(BASE_DIR)

# Path Isolation: Ensure we only use the libraries we installed
lib_dir = os.path.join(BASE_DIR, "_internal_libs")
if os.path.exists(lib_dir):
    # CRITICAL: Disable user site-packages to prevent global packages from overriding our GPU libs
    import site
    site.ENABLE_USER_SITE = False
    # Also remove user site-packages if already in path
    user_site = site.getusersitepackages() if hasattr(site, 'getusersitepackages') else None
    if user_site:
        sys.path = [p for p in sys.path if not p.startswith(user_site)]
    
    # Scrub any existing _internal_libs from path to avoid 3.12/3.13 pollution
    sys.path = [p for p in sys.path if "_internal_libs" not in p]
    # We insert at the BEGINNING to prioritize our bundled GPU libs over system libs
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    # Windows DLL paths for CUDA
    if sys.platform == 'win32':
        added_dlls = False
        for root, dirs, files in os.walk(lib_dir):
            if 'bin' in dirs:
                bin_path = os.path.normpath(os.path.join(root, 'bin'))
                if any(f.lower().endswith('.dll') for f in os.listdir(bin_path)):
                    try:
                        os.add_dll_directory(bin_path)
                        added_dlls = True
                    except: pass
        if added_dlls:
             logging.info("NVIDIA/CUDA DLL paths added to environment.")

try:
    log_print("Importing core utilities...")
    import sounddevice as sd
    import numpy as np
    from pynput import keyboard
    import pystray
    from PIL import Image, ImageDraw
    import pyperclip
    from huggingface_hub import hf_hub_download, snapshot_download
    
    # Global Torch Import (Essential for multi-threaded access)
    import torch
    
    # Windows Sound
    try:
        import winsound
    except ImportError:
        winsound = None
    
    log_print("Core imports successful.")
except Exception as e:
    import traceback
    log_print(f"CRITICAL UTILITY IMPORT ERROR: {e}")
    log_print(traceback.format_exc())
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"Critical Utility Initialization Error:\n\n{e}\n\nCheck privox_app.log for details.", "Privox Error", 0x10)
    sys.exit(1)

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 512 
VAD_THRESHOLD = 0.5 
SILENCE_DURATION_MS = 2000
MIN_SPEECH_DURATION_MS = 250
SPEECH_PAD_MS = 500

# Models
WHISPER_SIZE = "turbo" # Default
WHISPER_REPO = "deepdml/faster-whisper-large-v3-turbo-ct2"

# Llama 3.2 3B Instruct
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
        local_model_path = os.path.join(BASE_DIR, "models", GRAMMAR_FILE)
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
                        self.icon.notify("Downloading Grammar Model (2GB)... Please wait.", "Privox Setup")
                
                model_path = hf_hub_download(repo_id=GRAMMAR_REPO, filename=GRAMMAR_FILE)
            except Exception as e:
                log_print(f"\nError downloading model: {e}")
                self.loading_error = str(e)
                if self.icon:
                     self.icon.notify("Error: Cloud not download model. Check internet or place in 'models' folder.", "Privox Error")
                return

        try:
            # Heavy Import: Llama
            log_print("Importing llama_cpp...")
            from llama_cpp import Llama

            # CPU Fallback for Llama - Safer in bundled environments
            # Try GPU first if available, otherwise fallback to CPU (0)
            is_gpu = torch.cuda.is_available()
            n_gpu = -1 if is_gpu else 0
            
            try:
                self.model = Llama(
                    model_path=model_path, 
                    n_ctx=2048, 
                    n_gpu_layers=n_gpu, 
                    verbose=False
                )
            except Exception as e:
                log_print(f"Failed to load Llama with GPU ({n_gpu}), falling back to CPU (0): {e}")
                self.model = Llama(
                    model_path=model_path, 
                    n_ctx=2048, 
                    n_gpu_layers=0, 
                    verbose=False
                )
            
            log_print(f"Done. (GPU Acceleration: {'ENABLED' if is_gpu else 'DISABLED'})")
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
                    "You are Privox, an intelligent assistant. Execute the user's instruction perfectly. "
                    "Output ONLY the result. Do not chat."
                )
                user_content = text
            else:
                # Cleanup Mode (English Only)
                if self.dictation_prompt:
                    system_prompt = self.dictation_prompt.replace("{dict}", dict_prompt)
                else:
                    system_prompt = (
                        "You are a strict text editing engine. Your ONLY task is to rewrite the input text to be grammatically correct, better formatted, and professionally polished. "
                        "Preserve the original language (English or Traditional Chinese/Cantonese)."
                        "\n\nRULES:"
                        "\n1. Output ONLY the corrected text. Do NOT converse. Do NOT say 'Here is the corrected text'."
                        "\n2. FIX CAPITALIZATION: Ensure the first letter of every sentence is capitalized."
                        "\n3. FIX PUNCTUATION: Ensure every sentence ends with appropriate punctuation (., ?, or !)."
                        "\n4. If the input is a question, correct the grammar of the question. Do NOT answer it."
                        "\n5. FORMATTING: Use paragraphs for natural speech or narrative. Use markdown bulleted lists ONLY when the content is clearly a series of distinct items, a list of steps, or a shopping list."
                        "\n6. Maintain the original meaning and language."
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
        self.auto_stop_enabled = True
        self.silence_timeout_ms = 10000
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
        self.ui_state = "LOADING"
        
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
        self.pending_wakeup = False # Auto-start recording after loading?

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
                                                 min_silence_duration_ms=self.silence_timeout_ms, 
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
            log_print(f"Loading Faster-Whisper ({WHISPER_SIZE})...")
            
            try:
                # 1. Path Diagnostics
                local_whisper = os.path.join(BASE_DIR, "models", f"whisper-{WHISPER_SIZE}")
                log_print(f"ASR Diagnostic - BASE_DIR: {BASE_DIR}")
                log_print(f"ASR Diagnostic - Model path: {local_whisper}")
                
                if os.path.exists(local_whisper):
                    log_print(f"ASR Diagnostic - Folder contents: {os.listdir(local_whisper)}")
                    bin_file = os.path.join(local_whisper, "model.bin")
                    if os.path.exists(bin_file):
                         log_print(f"ASR Diagnostic - model.bin size: {os.path.getsize(bin_file) / (1024*1024):.1f} MB")
                
                # 2. Initialization Diagnostics
                from faster_whisper import WhisperModel
                log_print(f"ASR Diagnostic - Torch Path: {torch.__file__}")
                is_gpu = torch.cuda.is_available()
                log_print(f"ASR Diagnostic - CUDA available: {is_gpu}")
                
                if is_gpu:
                    log_print(f"ASR Diagnostic - CUDA Device: {torch.cuda.get_device_name(0)}")
                
                device_str = "cuda" if is_gpu else "cpu"
                compute_type = "float16" if is_gpu else "int8"

                # 3. Initialization with Fallback
                model_path = local_whisper if os.path.exists(os.path.join(local_whisper, "model.bin")) else WHISPER_REPO
                try:
                    log_print(f"ASR Diagnostic - Initializing WhisperModel on {device_str}...")
                    self.asr_model = WhisperModel(model_path, device=device_str, compute_type=compute_type)
                except Exception as e_gpu:
                    if is_gpu:
                        log_print(f"ASR Warning - CUDA failed ({e_gpu}). Falling back to CPU...")
                        self.asr_model = WhisperModel(model_path, device="cpu", compute_type="int8")
                    else:
                        raise e_gpu

                log_print(f"WhisperModel initialized successfully.")
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                log_print(f"ASR SETUP ERROR: {e}")
                log_print(f"Traceback: {error_details}")
                self.loading_status = "Error Loading ASR"
                self.update_tray_tooltip()
                return

            self.models_ready = True # VAD + Heavy loaded
            self.heavy_models_loaded = True
            self.loading_status = "Ready"
            self.update_status("READY")
            log_print(f"Acceleration Status: {'GPU ENABLED' if torch.cuda.is_available() else 'CPU MODE'}")
            
            # Reset activity timer so we don't immediately unload
            self.last_activity_time = time.time()
            self.sound_manager.play_start()
            
            # Auto-Start Recording if we woke up from F8
            if self.pending_wakeup:
                self.pending_wakeup = False
                # Must call from main thread logic ideally, but start_listening is thread-safe enough
                # We need a tiny delay to ensure sound finishes playing or doesn't overlap
                time.sleep(0.1) 
                self.start_listening()
  

    def unload_heavy_models(self):
        with self.model_lock:
            if not self.heavy_models_loaded:
                return
            
            idle_time = time.time() - self.last_activity_time
            log_print(f"Unloading Models (VRAM Saver - Idle for {idle_time:.1f}s)...")
            self.asr_model = None
            self.grammar_checker.unload_model()
            
            # Force Garbage Collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.heavy_models_loaded = False
            self.loading_status = "Idle (VRAM Free)"
            self.update_tray_tooltip()
            self.update_status("SLEEP") # Trigger flat line animation
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
                    self.auto_stop_enabled = config.get("auto_stop_enabled", True)
                    self.silence_timeout_ms = config.get("silence_timeout_ms", 10000)
                    self.custom_dictionary = config.get("custom_dictionary", [])
                    self.vram_timeout = config.get("vram_timeout", 60)
                    self.dictation_prompt = config.get("dictation_prompt", None)
                    self.command_prompt = config.get("command_prompt", None)
                    
                    # Model overrides
                    global WHISPER_SIZE, GRAMMAR_REPO, GRAMMAR_FILE
                    old_whisper = WHISPER_SIZE
                    WHISPER_SIZE = config.get("whisper_model", WHISPER_SIZE)
                    GRAMMAR_REPO = config.get("grammar_repo", GRAMMAR_REPO)
                    GRAMMAR_FILE = config.get("grammar_file", GRAMMAR_FILE)
                    
                    # Verify model folder exists if overridden
                    if WHISPER_SIZE != old_whisper:
                        model_path = os.path.join(BASE_DIR, "models", f"whisper-{WHISPER_SIZE}")
                        if not os.path.exists(model_path):
                            log_print(f"WARNING: Configured model '{WHISPER_SIZE}' not found at {model_path}. Transcription may fail.")
                    
                    if hotkey_str in keyboard.Key.__members__:
                        self.hotkey = keyboard.Key[hotkey_str]
                    else:
                        self.hotkey = keyboard.KeyCode.from_char(hotkey_str)
                    
                    log_print(f"Loaded Config - Hotkey: {self.hotkey}, Sound: {self.sound_enabled}, Timeout: {self.vram_timeout}s, Model: {WHISPER_SIZE}")
            else:
                log_print(f"config.json not found at {config_path}, using defaults")
        except Exception as e:
            log_print(f"Error loading config: {e}. Using default.")



    def update_tray_tooltip(self):
        if self.icon:
            self.icon.title = f"Privox: {self.loading_status}"

    def update_status(self, status):
        # status: READY, RECORDING, PROCESSING, ERROR, LOADING, SLEEP
        self.ui_state = status
        
        # Immediate text update (icon handled by loop or here if static)
        if not self.icon: return

        if status == "READY":
            self.icon.title = "Privox: Ready (F8)"
        elif status == "RECORDING":
            self.icon.title = "Privox: Listening..."
        elif status == "PROCESSING":
            self.icon.title = "Privox: Processing..."
        elif status == "ERROR":
            self.icon.title = "Privox: Error/No Mic"
        elif status == "SLEEP":
             self.icon.title = "Privox: Sleeping (VRAM Saver Active)"

    def animation_loop(self):
        frame = 0
        while self.running:
            try:
                if not self.icon:
                    time.sleep(1)
                    continue

                # Default static icon path
                icon_path = resource_path("assets/icon.png")
                base_img = None
                if os.path.exists(icon_path):
                     base_img = Image.open(icon_path).convert("RGBA")
                
                # If static state, just ensure base icon is set once (to avoid cpu usage)
                # But here we want custom static states too (e.g. ready = normal)
                
                new_icon = None
                
                if self.ui_state == "RECORDING":
                    # Waveform Animation
                    new_icon = self.draw_waveform(frame, base_img)
                    frame += 1
                    time.sleep(0.08) # 12fps
                    
                elif self.ui_state == "PROCESSING":
                    # Spinner Animation
                    new_icon = self.draw_spinner(frame, base_img)
                    frame += 1
                    time.sleep(0.08)
                    
                elif self.ui_state == "SLEEP":
                    # Flat Line (Static or slow pulse?) -> Let's do static flat line
                    # Only update if current icon is not already it? 
                    # Simpler to re-draw for now, optimization later if needed.
                    new_icon = self.draw_flat_line(base_img)
                    time.sleep(0.5) # Slow update
                    
                elif self.ui_state == "ERROR":
                    # Error Dot (Static - Monotone White)
                    # Maybe draw an "!" or solid circle
                    if base_img:
                        new_icon = base_img.copy()
                        d = ImageDraw.Draw(new_icon)
                        # Draw "!" or white dot
                        d.ellipse((48, 48, 60, 60), fill="white", outline="white")
                    time.sleep(0.5)
                    
                else: # READY or LOADING
                    # Just the base icon (Normal)
                    # Maybe clear any overlays
                    if base_img: new_icon = base_img
                    time.sleep(0.5)

                if new_icon:
                    self.icon.icon = new_icon
                    
            except Exception as e:
                log_print(f"Anim Error: {e}")
                time.sleep(1)

    def draw_waveform(self, frame, base_img):
        # Draw dynamic waveform bars on top of base or instead of?
        # User said "waveform as icon". Our icon IS a waveform.
        # So we should animate the bars of the icon itself? 
        # But we loaded a PNG. We can't easily animate components of a flat PNG.
        # OPTION: Draw the waveform procedurally from scratch (like generate_icon.py).
        
        size = (64, 64)
        bg_color = (25, 25, 35, 255) 
        bar_color = (255, 255, 255, 255)
        
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # 4 Bars
        import random
        import math
        
        bar_w = 8
        gap = 4
        total_w = (4 * bar_w) + (3 * gap)
        start_x = (64 - total_w) // 2
        center_y = 32
        
        # Animate heights based on sine wave + noise
        for i in range(4):
            # Phase shift for each bar
            # Time varying
            t = frame * 0.5
            
            # Base height + sin wave
            # random noise to look like voice
            noise = random.randint(-5, 5)
            h = 20 + int(15 * math.sin(t + i)) + noise
            h = max(4, min(60, h))
            
            x = start_x + i * (bar_w + gap)
            y1 = center_y - (h // 2)
            y2 = center_y + (h // 2)
            
            draw.rectangle((x, y1, x+bar_w, y2), fill=bar_color)
            
        # Monotone: No Red Dot
        # draw.ellipse((50, 50, 60, 60), fill="#ff4444", outline="white")
            
        return img

    def draw_spinner(self, frame, base_img):
        # Processing: Rotating Circle (Monotone White)
        size = (64, 64)
        bg_color = (25, 25, 35, 255) 
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw Arc
        start_angle = (frame * 30) % 360
        end_angle = (start_angle + 270) % 360
        
        draw.arc((12, 12, 52, 52), start=start_angle, end=end_angle, fill="white", width=4)
        
        return img

    def draw_flat_line(self, base_img):
        # Sleep Mode (Monotone White)
        size = (64, 64)
        bg_color = (25, 25, 35, 255) 
        img = Image.new("RGBA", size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Flat Line
        draw.line((10, 32, 54, 32), fill="white", width=3) # Changed from grey to white for visibility
        
        return img

    def audio_callback(self, indata, frames, time, status):
        if self.running and self.mic_active and self.models_ready:
            self.q.put(indata.copy())

    def on_press(self, key):
        if key == self.hotkey:
            # Wake up detection
            if not self.heavy_models_loaded:
                log_print("Wake up detected. Pre-loading models...")
                self.pending_wakeup = True # Auto-start recording when ready
                threading.Thread(target=self.load_heavy_models, daemon=True).start()
                return # Don't fall through to error message

            if not self.vad_model or self.asr_model is None:
                log_print("Ignored F8: Models not fully loaded.")
                self.sound_manager.play_error()
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
        try:
            duration = len(audio_data) / SAMPLE_RATE
            max_amp = np.max(np.abs(audio_data))
            rms = np.sqrt(np.mean(audio_data**2))
            
            log_print(f"\n--- Transcription Diagnostic ---")
            log_print(f"Audio Stats - Duration: {duration:.2f}s, Max Amp: {max_amp:.4f}, RMS: {rms:.4f}")
            
            if duration < (MIN_SPEECH_DURATION_MS / 1000):
                log_print(f" [Audio too short - Ignored]")
                self.update_status("READY")
                return

            if max_amp < 0.001: 
                log_print(f" [Audio too quiet - Ignored]")
                self.update_status("READY")
                return

            # Ensure models are loaded before transcribing
            if not self.heavy_models_loaded:
                log_print("Waiting for models to load...")
                self.load_heavy_models()
                if not self.asr_model:
                     log_print("ASR Model still missing after lazy load attempt.")
                     self.update_status("READY")
                     return

            log_print(f" Transcribing Using Model: {WHISPER_SIZE}...", flush=True)
            t0 = time.time()
            
            # Faster-Whisper
            segments, info = self.asr_model.transcribe(
                audio_data.astype(np.float32), 
                beam_size=5,
                language=None, # Auto-detect
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            log_print(f" ASR Result - Language Detected: {info.language} ({info.language_probability:.2f})")
            
            # Collect segments and log each one
            results = []
            for segment in segments:
                log_print(f"  Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] '{segment.text}'")
                results.append(segment.text)
            
            raw_text = " ".join(results).strip()
            
            t1 = time.time()
            log_print(f" [ASR Total Time: {t1 - t0:.3f}s] Joined Result: '{raw_text}'")
            
            if not raw_text:
                log_print(" [Empty Transcription Result]")
                self.update_status("READY")
                return

            # --- Logic for Command Mode (DISABLED FOR NOW) ---
            is_command = False
            command_text = raw_text
            
            # if raw_text.lower().startswith("privox"):
            #     is_command = True
            #     command_text = re.sub(r'^(privox)\s*,?\s*', '', raw_text, flags=re.IGNORECASE)
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
            # Diagnostic: Log input device
            try:
                device_info = sd.query_devices(kind='input')
                log_print(f"Audio Diagnostic - Using Default Input: {device_info.get('name')} (Channels: {device_info.get('max_input_channels')})")
            except Exception as de:
                log_print(f"Audio Diagnostic - Could not query input device: {de}")

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
                    if speech_dict:
                        if 'start' in speech_dict and not self.is_speaking:
                             self.is_speaking = True
                        if 'end' in speech_dict and self.is_listening and self.auto_stop_enabled:
                             log_print(" [Auto-Stop Detected]")
                             self.stop_listening()
                            
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
        app_name = "Privox"
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
        app_name = "Privox"
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
        image = self.draw_flat_line(None)
        self.icon = pystray.Icon("Privox", image, "Privox: Initializing...", menu)
        
        # Pass icon to grammar checker for notifications immediately
        self.grammar_checker.icon = self.icon

        print("System Tray started. Check your taskbar.")
        
        # Start Threads
        threading.Thread(target=self.processing_loop, daemon=True).start()
        threading.Thread(target=self.animation_loop, daemon=True).start()
        
        # Run Icon (Native Loop)
        self.icon.run()

if __name__ == "__main__":
    app = VoiceInputApp()
    app.run()
