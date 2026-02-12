import os
import sys
import subprocess
import threading
import time
import logging
import importlib

# Ensure we are in the same directory as the executable
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(sys.executable)
    os.chdir(EXE_DIR)
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    # If in src/, go up to root
    if EXE_DIR.endswith('src'):
        EXE_DIR = os.path.dirname(EXE_DIR)
    os.chdir(EXE_DIR)

# Set up logging for bootstrap in the EXE_DIR
logging.basicConfig(
    filename=os.path.join(EXE_DIR, 'wispr_bootstrap.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_info(msg):
    logging.info(msg)
    print(msg)

def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = EXE_DIR
    return os.path.join(base_path, relative_path)

def is_torch_available():
    importlib.invalidate_caches()
    try:
        # Use find_spec to avoid actually loading it if it's broken
        import importlib.util
        spec = importlib.util.find_spec("torch")
        return spec is not None
    except:
        return False

def show_tray_feedback():
    """ Show a tray icon while installation is running """
    try:
        import pystray
        from PIL import Image, ImageDraw
        
        def create_image():
            # Generate a simple red icon to indicate "Busy/Setup"
            image = Image.new('RGB', (64, 64), color='red')
            draw = ImageDraw.Draw(image)
            draw.ellipse((16, 16, 48, 48), fill='white')
            return image

        icon = pystray.Icon("Wispr Setup", create_image(), "Wispr: Installing AI Libraries...")
        
        # Run icon in its own thread
        threading.Thread(target=icon.run, daemon=True).start()
        return icon
    except Exception as e:
        log_info(f"Could not show tray feedback: {e}")
        return None

def install_dependencies():
    lib_dir = os.path.join(EXE_DIR, "_internal_libs")
    if not os.path.exists(lib_dir):
        os.makedirs(lib_dir)

    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)

    if is_torch_available():
        log_info("Required libraries already present.")
        return True

    # Lock file check
    lock_file = os.path.join(EXE_DIR, "wispr_setup.lock")
    if os.path.exists(lock_file):
        log_info("Setup already in progress (lock file found). Exiting this instance.")
        return False

    # Create lock file
    with open(lock_file, 'w') as f:
        f.write(str(os.getpid()))

    log_info("Missing core AI libraries (PyTorch/CUDA). Prompting user...")
    
    # Prompt user
    if getattr(sys, 'frozen', False):
        try:
            import ctypes
            res = ctypes.windll.user32.MessageBoxW(0, 
                "Wispr Initial Setup:\n\nDetailed AI libraries (PyTorch/CUDA) are needed and will be downloaded now.\n"
                "Size: ~2GB. This happens only once.\n\n"
                "A red icon will appear in your tray during setup.\n"
                "Continue?", 
                "Wispr Setup", 0x41) # 0x41 = MB_ICONINFORMATION | MB_OKCANCEL
            if res != 1: # 1 is IDOK
                log_info("User cancelled setup.")
                os.remove(lock_file)
                return False
        except:
            pass

    # Show tray feedback
    icon = show_tray_feedback()

    try:
        log_info("Starting download via pip...")
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "--target", lib_dir,
            "torch", "torchaudio", "nvidia-cudnn-cu12", "nvidia-cublas-cu12",
            "--no-cache-dir", "--quiet"
        ]
        
        log_info(f"Running: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        
        # Log pip output in real-time
        for line in process.stdout:
            log_info(f"Pip: {line.strip()}")
        
        process.wait()
        
        if icon:
            icon.stop()

        if process.returncode == 0:
            log_info("Installation successful.")
            os.remove(lock_file)
            return True
        else:
            log_info(f"Installation failed with code {process.returncode}")
            os.remove(lock_file)
            return False
            
    except Exception as e:
        log_info(f"Error during installation: {e}")
        if icon:
            icon.stop()
        if os.path.exists(lock_file):
            os.remove(lock_file)
        return False

def main():
    log_info(f"Wispr Bootstrap Launcher starting... (PID: {os.getpid()})")
    
    # Check if we should exit early (already running or success)
    success = install_dependencies()
    
    if success:
        log_info("Launching Wispr main application...")
        # Add the lib dir to path for the main app
        lib_dir = os.path.join(EXE_DIR, "_internal_libs")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
            
        try:
            # We need to import voice_input AFTER we added lib_dir to path
            # and ensured torch exists there.
            import voice_input
            app = voice_input.VoiceInputApp()
            app.run()
        except Exception as e:
            log_info(f"CRITICAL LAUNCH ERROR: {e}")
            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.user32.MessageBoxW(0, f"App Launch Failed:\n\n{e}", "Wispr Error", 0x10)
    else:
        # If we failed but it wasn't a Cancel, show error
        # (Success is only return if is_torch_available() or successful pip)
        pass

if __name__ == "__main__":
    main()
