import sys
import os

# --- 0. Hard Environment Isolation (MUST BE FIRST) ---
os.environ["PYTHONNOUSERSITE"] = "1"
import site
site.ENABLE_USER_SITE = False

import logging
import threading
import queue
import time
import json
import re
import gc
import concurrent.futures
if sys.platform == 'win32':
    import winreg
    import ctypes
    from ctypes import wintypes

def setup_logging():
    # Determine BASE_DIR early
    if getattr(sys, 'frozen', False):
        # We want the log to be in the same folder as the app for portability/custom paths
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.exists(base_dir):
        try: os.makedirs(base_dir, exist_ok=True)
        except: pass

    # Configure logging to always write to file in AppData
    log_file = os.path.join(base_dir, 'privox_app.log')
    log_level = logging.INFO
    log_format = '%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s'
    log_datefmt = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=log_format,
        datefmt=log_datefmt,
        level=log_level,
        force=True,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if sys.stdout else logging.NullHandler()
        ]
    )
    
    # Redirect stdout/stderr
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

# Silence noisy external loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Initialize logging IMMEDIATELY to catch import errors
setup_logging()

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
    # For custom install paths, we use the EXE directory
    BASE_DIR = os.path.dirname(os.path.normpath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Simplified Path Logic: Trust Pixi environment but handle DLLs if needed
if sys.platform == 'win32':
    # Ensure CUDA DLLs from pixi env are reachable (usually in .pixi/envs/default/bin)
    pixi_bin = os.path.join(BASE_DIR, ".pixi", "envs", "default", "bin")
    if os.path.exists(pixi_bin):
        try:
            os.add_dll_directory(pixi_bin)
            logging.info(f"Added Pixi bin to DLL directory: {pixi_bin}")
        except Exception as e:
            logging.warning(f"Failed to add Pixi bin to DLL directory: {e}")

try:
    log_print("Importing core utilities...")
    log_print(f"DEBUG: sys.path is: {sys.path}")

    import sounddevice as sd
    import numpy as np
    from pynput import keyboard
    import pystray
    from PIL import Image, ImageDraw
    import pyperclip
    from huggingface_hub import hf_hub_download, snapshot_download
    
    # Global Torch Import (Essential for multi-threaded access)
    import torch
    
    log_print(f"--- TORCH DIAGNOSTICS ---")
    log_print(f"Python Version: {sys.version}")
    log_print(f"Torch Version: {torch.__version__}")
    log_print(f"Torch Path: {getattr(torch, '__file__', 'Unknown')}")
    log_print("Importing Llama components...")
    from llama_cpp import Llama
    log_print("Llama import successful.")
    log_print(f"CUDA Available: {torch.cuda.is_available()}")
    log_print(f"CUDA Version: {torch.version.cuda}")
    log_print(f"CuDNN Version: {torch.backends.cudnn.version()}")
    if torch.cuda.is_available():
        log_print(f"Current Device: {torch.cuda.get_device_name(0)}")
    else:
        log_print("CUDA NOT AVAILABLE. Possible reasons:")
        log_print("1. CPU version of Torch installed (check version above)")
        log_print("2. Missing CUDA DLLs in PATH")
        log_print("3. GPU driver issues")
        
        # Native Popup for visibility
        if sys.platform == 'win32' and not getattr(sys, 'frozen', False):
             # Only show popup in dev mode for now to avoid annoying users, 
             # OR we can show it if we explicitly want GPU.
             pass
    log_print(f"-------------------------")
    
    # Windows Sound
    try:
        import winsound
    except ImportError:
        winsound = None
    
    log_print("Core imports successful.")
except Exception as e:
    import traceback
    err_stack = traceback.format_exc()
    log_print(f"CRITICAL UTILITY IMPORT ERROR: {e}")
    log_print(err_stack)
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, f"Privox Import Error:\n\n{e}\n\nTraceback:\n{err_stack[:500]}...", "Privox Fatal Error", 0x10)
    sys.exit(1)

# --- Configuration ---
SAMPLE_RATE = 16000
BLOCK_SIZE = 512 
VAD_THRESHOLD = 0.5 
SILENCE_DURATION_MS = 2000
MIN_SPEECH_DURATION_MS = 250
SPEECH_PAD_MS = 500

# Models
# Models
WHISPER_SIZE = "large-v3-turbo-cantonese" # Optimized for Hong Kong Code-switching
WHISPER_REPO = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english-ct2"
ASR_BACKEND = "whisper" # Default: whisper or sensevoice

# Llama 3.2 3B Instruct
GRAMMAR_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
GRAMMAR_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"


class SoundManager:
    def __init__(self, enabled=True):
        self.enabled = enabled and (winsound is not None)
        self.lock = threading.Lock()

    def _play(self, freq, duration):
        if self.enabled:
            # Using a lock ensures beeps don't collide if triggered rapidly
            try:
                with self.lock:
                    winsound.Beep(freq, duration)
            except Exception as e:
                log_print(f"Sound Error: {e}")

    def play_start(self):
        if self.enabled:
            threading.Thread(target=self._play, args=(1000, 200), daemon=True).start()

    def play_stop(self):
        if self.enabled:
            threading.Thread(target=self._play, args=(750, 200), daemon=True).start()

    def play_error(self):
        if self.enabled:
            threading.Thread(target=self._play, args=(400, 500), daemon=True).start()


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
            try:
                import llama_cpp
                log_print(f"llama-cpp-python Version: {getattr(llama_cpp, '__version__', 'Unknown')}")
                # Log the system info string - this tells us if CUDA is actually compiled in
                sys_info = llama_cpp.llama_print_system_info()
                log_print(f"llama-cpp-python System Info: {sys_info}")
                if "CUDA = 1" in str(sys_info) or "BLAS = 1" in str(sys_info):
                    log_print("GPU Backend detected in llama-cpp-python.")
                else:
                    log_print("WARNING: llama-cpp-python appears to be CPU-ONLY.")
            except ImportError as ie:
                log_print(f"CRITICAL: Failed to import llama_cpp: {ie}")
                raise ie
                
            from llama_cpp import Llama

            # Assertive GPU Offloading for Llama 3.2 3B
            is_gpu = torch.cuda.is_available()
            # If GPU is available, offload EVERYTHING (approx 28-33 layers for 3B)
            # Setting a very high number (99) ensures all layers are offloaded
            n_gpu = 99 if is_gpu else 0
            
            try:
                log_print(f"Loading Llama (GPU={is_gpu}, request_layers={n_gpu})...")
                self.model = Llama(
                    model_path=model_path, 
                    n_ctx=2048, 
                    n_gpu_layers=n_gpu, 
                    verbose=False,
                    n_threads=os.cpu_count() // 2 # Use half cores for safety if on CPU
                )
            except Exception as e:
                log_print(f"GPU Load failed, falling back to CPU: {e}")
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
            if sys.platform == 'win32':
                 ctypes.windll.user32.MessageBoxW(0, f"Error loading Grammar Model (Llama):\n\n{e}\n\nCheck logs for details.", "Privox Model Error", 0x10)

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
                        "You are an expert editor specializing in Hong Kong style 'Kongish' (mixed Cantonese and English). "
                        "Your goal is to make the input text clean and readable while strictly preserving the natural, informal Cantonese flavor. "
                        "Do NOT convert Cantonese into formal written Chinese. Do NOT over-edit oral Cantonese expressions."
                        "\n\nCRITICAL RULES:"
                        "\n1. FIX ENGLISH: Correct grammar and spelling within English parts."
                        "\n2. FIX TYPOS: Correct obvious Cantonese homophone typos."
                        "\n3. PUNCTUATION: Use appropriate punctuation (，、。？！) to make thoughts clear."
                        "\n4. KONGISH BALANCE: Keep the original mix of Cantonese and English exactly as provided."
                        "\n5. SEQUENCE: Strictly preserve the original word order and sequence. Do NOT rearrange phrases."
                        "\n6. NO CONVERSATION: Output ONLY the corrected text. No explanations."
                        "\n7. Maintain the speaker's original tone and informal flow."
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
        
        # Hotkey support
        self.hotkey = keyboard.Key.f8 # Default key

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
        """Concurrent loading of ASR and Grammar models to minimize wake-up latency."""
        with self.model_lock:
            if self.heavy_models_loaded:
                return

            log_print("Loading Heavy Models (Wake up)...")
            self.loading_status = "Loading Models..."
            self.update_tray_tooltip()

            def load_grammar():
                try:
                    self.grammar_checker.load_model()
                    return True
                except Exception as e:
                    log_print(f"Parallel Load Error (Grammar): {e}")
                    return False

            def load_asr():
                try:
                    is_gpu = torch.cuda.is_available()
                    device_str = "cuda" if is_gpu else "cpu"
                    
                    if ASR_BACKEND == "sensevoice":
                        sense_dir = os.path.join(BASE_DIR, "models", "SenseVoiceSmall")
                        log_print(f"ASR Diagnostic - Initializing SenseVoiceSmall on {device_str}...")
                        from funasr import AutoModel
                        self.asr_model = AutoModel(
                            model=sense_dir if os.path.exists(sense_dir) else "iic/SenseVoiceSmall",
                            device=device_str,
                            disable_update=True
                        )
                        log_print(f"SenseVoice initialized successfully.")
                    else:
                        compute_type = "float16" if is_gpu else "int8"
                        from faster_whisper import WhisperModel
                        local_whisper = os.path.join(BASE_DIR, "models", f"whisper-{WHISPER_SIZE}")
                        model_path = local_whisper if os.path.exists(os.path.join(local_whisper, "model.bin")) else WHISPER_REPO
                        log_print(f"ASR Diagnostic - Initializing WhisperModel ({WHISPER_SIZE}) on {device_str}...")
                        self.asr_model = WhisperModel(model_path, device=device_str, compute_type=compute_type)
                        log_print(f"WhisperModel initialized successfully.")
                    return True
                except Exception as e:
                    log_print(f"Parallel Load Error (ASR): {e}")
                    self.loading_status = "Error Loading ASR"
                    return False

            # Use ThreadPoolExecutor for concurrent model loading
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(load_grammar), executor.submit(load_asr)]
                results = [f.result() for f in futures]

            if not all(results):
                log_print("CRITICAL: One or more models failed to load.")
                self.update_tray_tooltip()
                return

            self.models_ready = True
            self.heavy_models_loaded = True
            self.loading_status = "Ready"
            self.update_status("READY")
            
            # Reset activity timer so we don't immediately unload
            self.last_activity_time = time.time()
            
            # If this was a manual F8 wakeup, we immediately start listening. 
            # Otherwise (initial load), we play the 'Ready' sound.
            if self.pending_wakeup:
                log_print("Pending Wakeup found. Auto-starting recording...")
                self.pending_wakeup = False
                # Tiny delay to ensure UI updates and avoid race conditions with sound manager
                time.sleep(0.1) 
                self.start_listening()
            else:
                self.sound_manager.play_start()
  

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
                    # Model overrides
                    global WHISPER_SIZE, WHISPER_REPO, GRAMMAR_REPO, GRAMMAR_FILE, ASR_BACKEND
                    old_whisper = WHISPER_SIZE
                    WHISPER_SIZE = config.get("whisper_model", WHISPER_SIZE)
                    WHISPER_REPO = config.get("whisper_repo", WHISPER_REPO)
                    GRAMMAR_REPO = config.get("grammar_repo", GRAMMAR_REPO)
                    GRAMMAR_FILE = config.get("grammar_file", GRAMMAR_FILE)
                    ASR_BACKEND = config.get("asr_backend", "whisper")
                    
                    # Verify model folder exists if overridden
                    if WHISPER_SIZE != old_whisper:
                        model_path = os.path.join(BASE_DIR, "models", f"whisper-{WHISPER_SIZE}")
                        if not os.path.exists(model_path):
                            log_print(f"WARNING: Configured model '{WHISPER_SIZE}' not found at {model_path}. Transcription may fail.")
                    
                    # Simple hotkey support (single key only)
                    self.hotkey = keyboard.Key.f8
                    if hotkey_str in keyboard.Key.__members__:
                        self.hotkey = keyboard.Key[hotkey_str]
                    elif len(hotkey_str) == 1:
                        self.hotkey = keyboard.KeyCode.from_char(hotkey_str)
                    else:
                        if hotkey_str.upper() in keyboard.Key.__members__:
                            self.hotkey = keyboard.Key[hotkey_str.upper()]
                        else:
                            log_print(f"Unknown hotkey: {hotkey_str}. Using F8.")
                    
                    log_print(f"Loaded Config - Hotkey: {hotkey_str}, Sound: {self.sound_enabled}, Timeout: {self.vram_timeout}s, Model: {WHISPER_SIZE}, Backend: {ASR_BACKEND}")
            else:
                log_print(f"config.json not found at {config_path}, using defaults")
        except Exception as e:
            log_print(f"Error loading config: {e}. Using default.")



    def update_tray_tooltip(self):
        if self.icon:
            gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
            self.icon.title = f"Privox: {self.loading_status} ({gpu_status})"

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
                return

            if not self.vad_model or self.asr_model is None:
                log_print("Ignored Hotkey: Models not fully loaded.")
                self.sound_manager.play_error()
                return

            if not self.mic_active:
                log_print("Ignored Hotkey: No Microphone Active")
                self.sound_manager.play_error()
                return
                
            if not self.is_listening:
                self.start_listening()
            else:
                self.stop_listening()

    def on_release(self, key):
        pass

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
            # Run transcription in a separate thread so we don't block the keyboard listener!
            threading.Thread(target=self.transcribe, args=(audio_segment,), daemon=True).start()
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

            log_print(f" Transcribing Using Backend: {ASR_BACKEND} (Model: {WHISPER_SIZE if ASR_BACKEND == 'whisper' else 'SenseVoiceSmall'})...", flush=True)
            t0 = time.time()
            
            raw_text = ""
            if ASR_BACKEND == "sensevoice":
                # SenseVoice/funasr
                results = self.asr_model.generate(
                    input=audio_data.astype(np.float32),
                    cache={},
                    language="auto", # SenseVoice handles LID well
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
                
                # funasr output is a list of dicts: [{'text': '...', 'key': '...'}]
                if results and len(results) > 0:
                    raw_text = results[0].get('text', '')
                    # Clean up emotion/event tags like <|HAPPY|>, <|ENTHUSIASTIC|>, etc.
                    raw_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
                
                log_print(f" SenseVoice Result - Raw: '{raw_text}'")
            else:
                # Faster-Whisper
                segments, info = self.asr_model.transcribe(
                    audio_data.astype(np.float32), 
                    beam_size=5,
                    language="yue", # Force Cantonese + English support
                    initial_prompt="這是一段廣東話同英文混合嘅錄音。It contains Cantonese and English mixed together.",
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                log_print(f" ASR Result - Language Detected: {info.language} ({info.language_probability:.2f})")
                
                # Collect segments and log each one
                seg_results = []
                for segment in segments:
                    log_print(f"  Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] '{segment.text}'")
                    seg_results.append(segment.text)
                
                raw_text = " ".join(seg_results).strip()
            
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
            # original_clipboard = pyperclip.paste() # DISABLED: User wants to keep text for manual paste
            pyperclip.copy(text)
            time.sleep(0.05) 
            with self.keyboard_controller.pressed(keyboard.Key.ctrl):
                self.keyboard_controller.press('v')
                self.keyboard_controller.release('v')
            time.sleep(0.2) 
            # pyperclip.copy(original_clipboard) # DISABLED
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
        self.start_audio_stream()
            
        while self.running:
            # VRAM Saver Check
            if self.heavy_models_loaded and not self.is_listening and self.ui_state != "PROCESSING":
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
        icon.visible = False
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
        # Single Instance Mutex Check (Windows)
        self.mutex_handle = None
        if sys.platform == 'win32':
            mutex_name = "Global\\Privox_SingleInstance_Mutex"
            self.mutex_handle = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
            last_error = ctypes.windll.kernel32.GetLastError()
            
            # ERROR_ALREADY_EXISTS = 183
            if last_error == 183:
                log_print("Another instance of Privox is already running. Exiting.")
                if self.mutex_handle:
                    ctypes.windll.kernel32.CloseHandle(self.mutex_handle)
                
                # POPUP to let user know why it's invisible
                ctypes.windll.user32.MessageBoxW(0, "Privox is already running in the background.\nCheck your system tray or Task Manager.", "Privox", 0x40)
                sys.exit(0)
            
            log_print("Acquired single-instance mutex.")

        # Setup Tray Menu
        menu = pystray.Menu(
            pystray.MenuItem('Run at Startup', self.toggle_startup, checked=self.check_startup_status),
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
        
        # Start Keyboard Listener
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        
        # Run Icon (Native Loop)
        log_print("Starting Tray Icon Loop...")
        # ctypes.windll.user32.MessageBoxW(0, "About to show Tray Icon. Click OK to continue.", "Privox Debug", 0x40)
        self.icon.run()

if __name__ == "__main__":
    try:
        # 3. Early GPU Check
        import torch
        gpu_detected = torch.cuda.is_available()
        
        logging.info("--- VoiceInputApp Startup ---")
        logging.info(f"Python Executable: {sys.executable}")
        logging.info(f"sys.path: {sys.path}")
        logging.info(f"GPU Support Detected: {gpu_detected}")
        if gpu_detected:
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("GPU NOT DETECTED early. Processing may be slow.")
        
        app = VoiceInputApp()
        app.run()
    except Exception as e:
        import traceback
        err_msg = f"Fatal Error on Startup:\n\n{e}\n\n{traceback.format_exc()}"
        logging.error(err_msg)
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, f"Privox failed to start:\n\n{e}\n\nCheck privox_app.log for details.", "Privox Fatal Error", 0x10)
        sys.exit(1)
