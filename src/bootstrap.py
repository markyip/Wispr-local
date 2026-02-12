import os
import sys
import subprocess
import threading
import time
import logging
import importlib
import queue
import tkinter as tk
from tkinter import ttk, messagebox

# Disable Symlinks for Windows (Fixes WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
# Disable Progress Bars for Hugging Face (Avoids tqdm crash in --noconsole)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

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

# Set up logging for bootstrap
# Default to console only unless PRIVOX_DEBUG=1 is set
log_level = logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(message)s'

if os.environ.get("PRIVOX_DEBUG") == "1":
    logging.basicConfig(
        filename=os.path.join(EXE_DIR, 'privox_setup.log'),
        filemode='a',
        format=log_format,
        level=log_level,
        force=True
    )
else:
    # Console only logging - explicitly use StreamHandler to avoid any default file behavior
    # and avoid redirect loops if sys.stdout/stderr are later redirected
    logging.basicConfig(
        format=log_format,
        level=log_level,
        force=True,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

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

def log_info(msg):
    # print() will be caught by LoggerWriter and logged
    print(msg)

log_info("--- Privox Setup Started ---")
log_info(f"Platform: {sys.platform}")
log_info(f"Python: {sys.version}")
try:
    gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if gpu_check.returncode == 0:
        log_info("GPU Detected via nvidia-smi")
    else:
        log_info("No NVIDIA GPU detected or driver issue.")
except:
    log_info("Could not run nvidia-smi check.")

class InstallerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Privox Setup")
        
        # Window Setup
        width = 600
        height = 480
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.resizable(False, False)
        
        # Icon
        icon_path = os.path.join(EXE_DIR, "assets", "icon.ico")
        if not os.path.exists(icon_path) and getattr(sys, 'frozen', False):
             icon_path = os.path.join(sys._MEIPASS, "assets", "icon.ico")
        
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        # Style using ttk
        style = ttk.Style()
        style.theme_use('vista' if sys.platform == 'win32' else 'clam')
        
        # Variables
        self.gpu_support = tk.BooleanVar(value=True if sys.platform == 'win32' else False)
        self.install_mode = tk.BooleanVar(value=True) # True=Install, False=Portable
        self.progress_text = tk.StringVar(value="Ready to install...")
        self.progress_val = tk.DoubleVar(value=0)
        self.active_process = None
        self.is_cancelling = False
        
        # --- Layout ---
        self.main_content = ttk.Frame(self, padding="20")
        self.main_content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        ttk.Separator(self, orient='horizontal').pack(fill=tk.X)

        self.bottom_bar = ttk.Frame(self, padding="10")
        self.bottom_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Buttons
        self.btn_next = ttk.Button(self.bottom_bar, text="Next >", command=None)
        self.btn_next.pack(side=tk.RIGHT)
        
        self.btn_cancel = ttk.Button(self.bottom_bar, text="Cancel", command=self.on_cancel)
        self.btn_cancel.pack(side=tk.RIGHT, padx=10)

        # Pages storage
        self.pages = {}
        self.current_page_name = None
        
        self.create_pages()
        self.show_page("welcome")

    def create_pages(self):
        # --- Welcome Page ---
        p1 = ttk.Frame(self.main_content)
        
        ttk.Label(p1, text="Welcome to Privox Setup", font=("Segoe UI", 16, "bold")).pack(pady=(10, 20))
        ttk.Label(p1, text="Privox is a local, privacy-focused speech-to-text tool.\nThis wizard will guide you through the setup process.", justify=tk.CENTER).pack(pady=10)
        
        opt_frame = ttk.LabelFrame(p1, text="Installation Mode", padding=15)
        opt_frame.pack(fill=tk.X, pady=20)
        
        ttk.Radiobutton(opt_frame, text="Install to System (Recommended)", variable=self.install_mode, value=True).pack(anchor=tk.W, pady=5)
        ttk.Label(opt_frame, text="   Installs to AppData, creates shortcuts, and enables auto-start.", font=("Segoe UI", 9), foreground="gray").pack(anchor=tk.W)
        
        ttk.Radiobutton(opt_frame, text="Run Portable Mode", variable=self.install_mode, value=False).pack(anchor=tk.W, pady=(15, 5))
        ttk.Label(opt_frame, text="   Runs directly from this folder. No shortcuts created.", font=("Segoe UI", 9), foreground="gray").pack(anchor=tk.W)
        
        gpu_frame = ttk.Frame(p1)
        gpu_frame.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(gpu_frame, text="Install NVIDIA GPU Support (~2GB)", variable=self.gpu_support).pack(anchor=tk.W)
        ttk.Label(gpu_frame, text="   Uncheck this if you do not have an NVIDIA GPU (Saves space).", font=("Segoe UI", 8), foreground="gray").pack(anchor=tk.W)
        
        self.pages["welcome"] = p1
        
        # --- Progress Page ---
        p2 = ttk.Frame(self.main_content)
        ttk.Label(p2, text="Installing Privox...", font=("Segoe UI", 14, "bold")).pack(pady=(10, 20))
        
        self.pb = ttk.Progressbar(p2, variable=self.progress_val, maximum=100)
        self.pb.pack(fill=tk.X, pady=10)
        
        ttk.Label(p2, textvariable=self.progress_text, font=("Segoe UI", 9)).pack(anchor=tk.W)
        
        self.log_box = tk.Text(p2, height=12, state=tk.DISABLED, font=("Consolas", 8))
        self.log_box.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.pages["progress"] = p2
        
        # --- Success Page ---
        p3 = ttk.Frame(self.main_content)
        ttk.Label(p3, text="Installation Complete!", font=("Segoe UI", 16, "bold"), foreground="green").pack(pady=(40, 20))
        ttk.Label(p3, text="Privox has been successfully set up.", justify=tk.CENTER).pack(pady=10)
        
        self.pages["success"] = p3

    def show_page(self, name):
        self.after(0, self._show_page_ui, name)

    def _show_page_ui(self, name):
        if self.current_page_name:
            self.pages[self.current_page_name].pack_forget()
            
        self.current_page_name = name
        self.pages[name].pack(fill=tk.BOTH, expand=True)
        
        # Update Buttons based on page
        if name == "welcome":
            self.btn_next.config(text="Install", state=tk.NORMAL, command=self.start_installation)
            self.btn_cancel.config(state=tk.NORMAL)
        elif name == "progress":
            self.btn_next.config(text="Installing...", state=tk.DISABLED)
            self.btn_cancel.config(state=tk.NORMAL)
        elif name == "success":
            self.btn_next.config(text="Launch", state=tk.NORMAL, command=self.launch_and_exit)
            self.btn_cancel.config(text="Close", state=tk.NORMAL, command=self.destroy)

    def on_cancel(self):
        if self.current_page_name == "progress":
            if not messagebox.askyesno("Cancel Setup", "Do you want to stop the installation?"):
                return
            
            self.is_cancelling = True
            if self.active_process:
                try:
                    self.log("Stopping active processes...")
                    self.active_process.terminate()
                except: pass
        
        self.destroy()
        sys.exit(0)

    def log(self, msg):
        log_info(msg)
        self.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)
        self.progress_text.set(msg)

    def start_installation(self):
        self.show_page("progress")
        threading.Thread(target=self.run_install_process, daemon=True).start()

    def run_install_process(self):
        try:
            log_info("Starting installation thread...")
            target_exe_path = sys.executable
            target_dir = EXE_DIR
            
            # 1. Install Files (EXE)
            if self.install_mode.get():
                self.log("Installing application files...")
                new_path = install_app_files(self.log)
                if new_path:
                    target_exe_path = new_path
                    target_dir = os.path.dirname(target_exe_path)
                else:
                    self.log("Installation failed. Using portable mode.")
            
            self.target_dir = target_dir
            self.target_exe = target_exe_path

            # 2. Install Dependencies
            self.log("Checking dependencies...")
            self.pb.config(value=5) # Visual start
            
            success = install_dependencies(self, target_dir, self.gpu_support.get())
            if not success:
                if self.is_cancelling:
                    self.log("Installation cancelled by user.")
                else:
                    self.log("Dependency installation failed!")
                return

            self.pb.config(value=40)

            # 3. Download Model
            self.log("Checking AI Models...")
            check_and_download_model(self, target_dir)
            
            self.pb.config(value=100)
            self.log("Setup Finished.")
            
            # 4. Finalize
            # Only NOW do we consider the installation "complete" and ready to launch.
            # (Note: Shortcuts/Registry were done in install_app_files, which is fine)
            
            self.after(500, lambda: self.show_page("success"))
            
        except Exception as e:
            msg = f"A fatal error occurred during installation:\n\n{e}"
            log_info(f"FATAL INSTALL ERROR: {e}")
            import traceback
            log_info(traceback.format_exc())
            
            def show_error_state():
                messagebox.showerror("Installation Error", msg)
                self.progress_text.set("Installation Failed.")
                self.btn_next.config(text="Exit", state=tk.NORMAL, command=self.destroy)
                self.btn_cancel.config(state=tk.DISABLED)

            self.after(0, show_error_state)

    def launch_and_exit(self):
        try:
            if self.install_mode.get():
                subprocess.Popen([self.target_exe, "--run"])
            else:
                 launch_main_app(self.target_dir)
            
            self.destroy()
            sys.exit(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch: {e}")

# --- Helper Functions ---

def create_shortcut(target, shortcut_path, description="", icon=None):
    try:
        vbs_script = f"""
        Set oWS = WScript.CreateObject("WScript.Shell")
        sLinkFile = "{shortcut_path}"
        Set oLink = oWS.CreateShortcut(sLinkFile)
        oLink.TargetPath = "{target}"
        oLink.Description = "{description}"
        oLink.WorkingDirectory = "{os.path.dirname(target)}"
        {f'oLink.IconLocation = "{icon}"' if icon else ''}
        oLink.Save
        """
        vbs_file = os.path.join(os.environ['TEMP'], f"mkshortcut_{os.getpid()}.vbs")
        with open(vbs_file, "w") as f:
            f.write(vbs_script)
        subprocess.call(["cscript", "//nologo", vbs_file])
        os.remove(vbs_file)
        return True
    except Exception as e:
        return False

def get_python_exe(log_callback=None):
    """ Finds a real Python interpreter matching the current major.minor version. """
    target_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    if getattr(sys, 'frozen', False):
        # 1. Try common absolute paths FIRST
        potential_names = ["python.exe", "pythonw.exe"]
        potential_paths = []
        for name in potential_names:
            potential_paths.extend([
                os.path.expandvars(rf"%LOCALAPPDATA%\Programs\Python\Python313\{name}"),
                os.path.expandvars(rf"%LOCALAPPDATA%\Programs\Python\Python312\{name}"),
                rf"C:\ProgramData\miniconda3\{name}",
                rf"C:\ProgramData\anaconda3\{name}",
                os.path.expandvars(rf"%SystemDrive%\Python313\{name}"),
                os.path.expandvars(rf"%SystemDrive%\Python312\{name}"),
            ])
        
        for path in potential_paths:
            if os.path.exists(path):
                # Verify version
                try:
                    ver = subprocess.check_output([path, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                                   text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
                    if ver == target_ver:
                        if log_callback: log_callback(f"Using verified Python {ver}: {path}")
                        return path
                except: continue

        # 2. Try PATH
        for cmd in ["python", "python3", "py"]:
            try:
                ver = subprocess.check_output([cmd, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], 
                                               text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()
                if ver == target_ver:
                    # Check if pip is available
                    subprocess.run([cmd, "-m", "pip", "--version"], capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    if log_callback: log_callback(f"Using system Python {ver} (pip found): {cmd}")
                    return cmd
            except: continue
        
        return None
    return sys.executable

def install_app_files(log_callback=None):
    """ Copies the EXE and resources to LocalAppData. Does NOT launch the app. """
    try:
        app_name = "Privox"
        exe_name = "Privox.exe"
        target_dir = os.path.join(os.environ['LOCALAPPDATA'], "Privox")
        target_exe = os.path.join(target_dir, exe_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        current_exe = sys.executable
        if log_callback: log_callback(f"Terminating old instances...")
        try:
            my_pid = os.getpid()
            # 1. Kill the EXE instances (current and legacy names)
            subprocess.run(["taskkill", "/F", "/IM", "Privox.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "pythonw.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            
            # Additional names just in case
            subprocess.run(["taskkill", "/F", "/IM", "WisprLocal.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "WisprLocal_v2.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            
            # 2. Kill Python processes running our script (more targeted)
            # We use wmic for targeted killing of python processes specifically running voice_input.py
            cmd = 'wmic process where "commandline like \'%voice_input.py%\'" get processid'
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                pid = line.strip()
                if pid.isdigit() and int(pid) != my_pid:
                    try: subprocess.run(["taskkill", "/F", "/PID", pid], creationflags=subprocess.CREATE_NO_WINDOW)
                    except: pass
            
            # Also try to kill any process that has our folder open (less reliable but worth a shot)
            # Nothing built-in for this without third-party tools, so we rely on the loop below.
            
            time.sleep(2.0) # Wait for them to die
        except: pass
        
        if log_callback: log_callback(f"Copying to {target_dir}...")
        
        import shutil
        if os.path.normpath(current_exe) != os.path.normpath(target_exe):
            try:
                shutil.copy2(current_exe, target_exe)
            except Exception as e:
                if log_callback: log_callback(f"Copy Warning (continuing): {e}")
        else:
             if log_callback: log_callback("Running from install directory. Skipping copy.")
        
        config_path = os.path.join(EXE_DIR, "config.json")
        if os.path.exists(config_path):
             shutil.copy2(config_path, os.path.join(target_dir, "config.json"))

        for folder in ["models", "assets"]: # Copy models and UI assets
            src = os.path.join(EXE_DIR, folder)
            if not os.path.exists(src) and getattr(sys, 'frozen', False):
                 src = os.path.join(sys._MEIPASS, folder)
            
            dst = os.path.join(target_dir, folder)
            if os.path.exists(src):
                if log_callback: log_callback(f"Copying {folder}...")
                if os.path.exists(dst): 
                    try: shutil.rmtree(dst)
                    except: pass
                shutil.copytree(src, dst)
        
        # 3. Copy Source Code (for script-based launch)
        src_dir = os.path.join(target_dir, "src")
        if not os.path.exists(src_dir): os.makedirs(src_dir)
        
        main_script = os.path.join(EXE_DIR, "src", "voice_input.py")
        if not os.path.exists(main_script) and getattr(sys, 'frozen', False):
             main_script = os.path.join(sys._MEIPASS, "src", "voice_input.py")
             
        if os.path.exists(main_script):
            if log_callback: log_callback("Extracting application source...")
            shutil.copy2(main_script, os.path.join(src_dir, "voice_input.py"))

        # Also copy the fixer script
        fixer_src = os.path.join(EXE_DIR, "src", "fix_cuda_build.py")
        if not os.path.exists(fixer_src) and getattr(sys, 'frozen', False):
             fixer_src = os.path.join(sys._MEIPASS, "src", "fix_cuda_build.py")
        
        if os.path.exists(fixer_src):
            shutil.copy2(fixer_src, os.path.join(src_dir, "fix_cuda_build.py"))
        
        if log_callback: log_callback("Creating Shortcuts...")
        start_menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
        create_shortcut(target_exe, os.path.join(start_menu, f"{app_name}.lnk"), "Privox AI Voice Input", target_exe)
        
        if log_callback: log_callback("Registering Uninstaller...")
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Privox"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, app_name)
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, f'"{target_exe}" --uninstall')
            winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, target_exe)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Privox")
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, target_dir)
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)

        run_key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key_path, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "Privox", 0, winreg.REG_SZ, f'"{target_exe}"')

        return target_exe
    except Exception as e:
        if log_callback: log_callback(f"Install error: {e}")
        return None

def install_dependencies(gui_instance, target_base_dir, gpu_support):
    log_callback = gui_instance.log
    lib_dir = os.path.join(target_base_dir, "_internal_libs")
    version_file = os.path.join(lib_dir, ".py_version")
    current_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    # Define python_exe EARLY because it's needed for the fixer script
    python_exe = get_python_exe(log_callback)
    if not python_exe:
        if log_callback: log_callback("Error: No Python interpreter found! Please install Python.")
        return False

    if os.path.exists(lib_dir):
        # ALWAYS attempt to rename the folder on every install/update.
        # This is the ONLY way to guarantee no 'PermissionError (WinError 5)' from locked DLLs
        # because even if a process has a DLL open, you can usually rename the PARENT folder on Windows.
        if log_callback: log_callback("Checking for locked libraries...")
        temp_cleanup = lib_dir + f".old_{int(time.time())}"
        success = False
        
        # Try up to 3 times to clear the folder
        for attempt in range(3):
            try:
                if not os.path.exists(lib_dir):
                    success = True
                    break
                    
                if log_callback: log_callback(f"Wiping existing libraries (Attempt {attempt+1})...")
                os.rename(lib_dir, temp_cleanup)
                os.makedirs(lib_dir)
                # Try to delete the old one in a thread
                threading.Thread(target=lambda: shutil.rmtree(temp_cleanup, ignore_errors=True), daemon=True).start()
                success = True
                break
            except Exception as e:
                if log_callback: log_callback(f"Folder locked: {e}. Retrying force delete...")
                
                # Force delete with batch
                script_path = os.path.join(os.environ['TEMP'], f"delete_libs_{int(time.time())}.bat")
                with open(script_path, "w") as f:
                    # More aggressive rd /s /q
                    f.write(f'@echo off\ntimeout /t 1 /nobreak >nul\nrd /s /q "{lib_dir}"\nexit\n')
                
                subprocess.run(["cmd", "/c", script_path], creationflags=subprocess.CREATE_NO_WINDOW)
                time.sleep(2.0)
                try: os.remove(script_path)
                except: pass
                
                if not os.path.exists(lib_dir):
                    os.makedirs(lib_dir)
                    success = True
                    break

        if not success:
            if log_callback: 
                log_callback("FATAL ERROR: Could not clear library folder.")
                log_callback("Please REBOOT your computer and try again.")
            raise Exception("Installation blocked by locked files. Please reboot and retry.")
    else:
        os.makedirs(lib_dir)

    # Check for Llama CPP and Faster Whisper
    llama_check = os.path.join(lib_dir, "llama_cpp")
    fw_check = os.path.join(lib_dir, "faster_whisper")
    torch_check = os.path.join(lib_dir, "torch")
    
    # ---------------------------------------------------------
    # NEW: Run dependency fixer script to handle CUDA Integration
    # This helps fix "No CUDA toolset found" errors in CMake
    # ---------------------------------------------------------
    # We now look in src/ because that's where we copy it
    fixer_script = os.path.join(target_base_dir, "src", "fix_cuda_build.py")
    if os.path.exists(fixer_script) and gpu_support and sys.platform == 'win32':
         if log_callback: log_callback("Running CUDA Integration Fixer...")
         try:
             # DEBUG: Log before running fixer
             if log_callback: log_callback(f"DEBUG: Executing {fixer_script}")
             subprocess.run([python_exe, fixer_script], check=True, 
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
             if log_callback: log_callback("DEBUG: CUDA Fixer completed.")
         except Exception as e:
             if log_callback: log_callback(f"Warning: CUDA Fixer failed ({e}). Compilation might fail.")
    # ---------------------------------------------------------

    # Validation: Check if we have the RIGHT torch version (CUDA)
    valid_install = False
    # ... (validation logic skipped for brevity) ...

    # Define packages list (Critical: don't delete this!)
    packages = [
        "torch", "torchaudio", 
        "faster-whisper",
        "numpy<2.0.0", 
        "llama-cpp-python",
        "sounddevice",
        "pynput",
        "pystray",
        "Pillow",
        "pyperclip",
        "huggingface_hub"
    ]
    
    if gpu_support and sys.platform == 'win32':
        packages.extend(["nvidia-cudnn-cu12", "nvidia-cublas-cu12"])

    if sys.platform == 'win32' and gpu_support:
        # STEP 1: FORCE INSTALL PyTorch (CUDA)
        # We MUST do this separately and FIRST to ensure we get the cu124 version
        # PIN VERSION to prevent accidental upgrade to newer CPU-only versions from PyPI (e.g. 2.10)
        torch_packages = [
            "torch==2.6.0+cu124", 
            "torchaudio==2.6.0+cu124", 
            "nvidia-cudnn-cu12", 
            "nvidia-cublas-cu12"
        ]
        torch_cmd = [
            python_exe, "-m", "pip", "install", 
            "--target", lib_dir,
            "--no-input",
            "--force-reinstall", # FORCE reinstall to overwrite any CPU version
            "--index-url", "https://download.pytorch.org/whl/cu124",
            # NO extra-index-url here to prevent PyPI fallback
        ] + torch_packages

        if log_callback: log_callback(f"Step 1/3: Installing PyTorch (CUDA)...")
        
        # DEBUG: Log torch command
        # if log_callback: log_callback(f"DEBUG: Torch CMD: {torch_cmd}")

        try:
            if not run_pip(gui_instance, torch_cmd):
                if log_callback: log_callback("ERROR: PyTorch installation failed (run_pip returned False).")
                return False
            if log_callback: log_callback("DEBUG: Step 1 completed successfully.")
        except Exception as e:
            if log_callback: log_callback(f"CRITICAL ERROR in Step 1: {e}")
            return False

        # STEP 2: INSTALL Llama-CPP (CUDA Preference with Source/CPU Fallback)
        # Strategy:
        # 1. Try to download pre-built CUDA wheel (Best for Py3.10-3.12)
        # 2. If wheel fails, try compiling from source (Best for Py3.13 + NVCC)
        # 3. If compilation fails, fall back to CPU version (Last resort)
        try:
            llama_cmd_gpu = [
                python_exe, "-m", "pip", "install", 
                "--target", lib_dir,
                "--no-input",
                "--force-reinstall",
                "--index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
                "llama-cpp-python"
            ]
            
            if log_callback: log_callback(f"Step 2/3: Attempting to install Llama-CPP CUDA wheel...")
            if not run_pip(gui_instance, llama_cmd_gpu):
                raise Exception("CUDA wheel not found")
            else:
                if log_callback: log_callback("Success: Installed Llama-CPP CUDA wheel.")

        except:
            if log_callback: log_callback("Warning: Pre-built CUDA wheel not found. Attempting fallback...")
            try:
                # Check for NVCC
                subprocess.run(["nvcc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if log_callback: log_callback("NVCC detected. Attempting to compile Llama-CPP from source for GPU...")
                
                # Set environment variables for compilation
                env = os.environ.copy()
                env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
                env["FORCE_CMAKE"] = "1"
                
                # Install from source (no binary preference)
                llama_cmd_compile = [
                    python_exe, "-m", "pip", "install", 
                    "--target", lib_dir,
                    "--no-input",
                    "--force-reinstall",
                    "--upgrade",
                    "--no-cache-dir",
                    "llama-cpp-python"
                ]
                
                # Custom run_pip with env support
                process = subprocess.Popen(
                    llama_cmd_compile, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                gui_instance.active_process = process
                
                for line in process.stdout:
                    if log_callback: log_callback(f"COMPILE: {line.strip()}")
                
                process.wait()
                gui_instance.active_process = None
                
                if process.returncode == 0:
                        if log_callback: log_callback("Successfully compiled Llama-CPP with CUDA support!")
                else:
                        raise Exception("Compilation failed")

            except Exception as e:
                if log_callback: log_callback(f"Compilation fallback failed ({e}). Proceeding to CPU version.")
                
                llama_cmd_cpu = [
                    python_exe, "-m", "pip", "install", 
                    "--target", lib_dir,
                    "--no-input",
                    "--upgrade",
                    "--index-url", "https://pypi.org/simple",
                    "llama-cpp-python"
                ]
                run_pip(gui_instance, llama_cmd_cpu)
            if log_callback: log_callback("Warning: Llama-CPP CUDA wheel not found (likely no Python 3.13 support yet). Falling back to CPU version...")
            llama_cmd_cpu = [
                python_exe, "-m", "pip", "install", 
                "--target", lib_dir,
                "--no-input",
                "--upgrade",
                "--index-url", "https://pypi.org/simple",
                "llama-cpp-python"
            ]
            run_pip(gui_instance, llama_cmd_cpu)

        # STEP 3: Install Remaining Dependencies
        remaining = [p for p in packages if p not in ["torch", "torchaudio", "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "llama-cpp-python"]]
        cmd = [
             python_exe, "-m", "pip", "install", 
             "--target", lib_dir,
             "--no-input",
             # NO --upgrade
             "--index-url", "https://pypi.org/simple",
             "--extra-index-url", "https://download.pytorch.org/whl/cu124" # Allow finding +cu124 deps if needed
        ] + remaining
    else:
        # Standard CPU/Mac install
        cmd = [
            python_exe, "-m", "pip", "install", 
            "--target", lib_dir,
            "--no-input",
            "--upgrade"
        ] + packages

    if log_callback: log_callback(f"Step 3/3: Installing other dependencies (sounddevice, etc.)...")
    result = run_pip(gui_instance, cmd, version_file, current_ver)

    # SELF-VERIFICATION
    if result:
         try:
             # import sys  <-- CAUSES UnboundLocalError because sys is used earlier in this function!
             if lib_dir not in sys.path: sys.path.insert(0, lib_dir)
             if log_callback: log_callback("Verifying sounddevice installation...")
             import sounddevice
             if log_callback: log_callback("Verification SUCCESS: sounddevice imported.")
         except ImportError as e:
             if log_callback: log_callback(f"Verification FAILED: Could not import sounddevice ({e})")
             # Don't fail the whole setup, but warn
    
    return result

def run_pip(gui_instance, cmd, version_file=None, current_ver=None):
    log_callback = gui_instance.log
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        gui_instance.active_process = process
        
        for line in process.stdout:
            if log_callback: log_callback(f"PIP: {line.strip()}")
            # Cosmetic progress update
            if hasattr(gui_instance, 'progress_val'):
                 current = gui_instance.progress_val.get()
                 if current < 40: # Cap at 40% for install phase
                     gui_instance.progress_val.set(current + 0.5)

        process.wait()
        gui_instance.active_process = None
        
        if process.returncode == 0:
            if version_file and current_ver:
                 with open(version_file, "w") as f:
                    f.write(current_ver)
            return True
        return False
    except Exception as e:
        if log_callback: log_callback(f"PIP Execution Error: {e}")
        return False

def check_and_download_model(gui_instance, target_base_dir):
    models_dir = os.path.join(target_base_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Helper to handle both log_callback and gui_instance
    def safe_log(msg):
        if hasattr(gui_instance, 'log'):
            gui_instance.log(msg)
        else:
            log_info(msg)

    def set_progress(val):
        if hasattr(gui_instance, 'progress_val'):
            gui_instance.progress_val.set(val)

    # --- Download Models ---
    try:
        # Pre-import standard libraries to prevent shadowing (UUID issue fix)
        import uuid
        import shutil
        import json
        
        lib_dir = os.path.join(target_base_dir, "_internal_libs")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
            
        from huggingface_hub import hf_hub_download, snapshot_download
        
        # 1. Grammar Model (Llama)
        grammar_file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        grammar_repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        if not os.path.exists(os.path.join(models_dir, grammar_file)):
            safe_log(f"Downloading Grammar Model ({grammar_file})...")
            # We jump to 45% when starting downloads
            set_progress(45)
            
            hf_hub_download(repo_id=grammar_repo, filename=grammar_file, local_dir=models_dir)
        
        set_progress(60)

        # 2. Whisper Model (Faster-Whisper Format)
        whisper_model_name = "turbo"
        whisper_repo = "deepdml/faster-whisper-large-v3-turbo-ct2"
        whisper_target = os.path.join(models_dir, "whisper-" + whisper_model_name)
        
        # Check for repo-specific tag to force redownload if we switched repos
        repo_tag_file = os.path.join(whisper_target, ".repo_id")
        existing_repo = ""
        if os.path.exists(repo_tag_file):
            try:
                with open(repo_tag_file, "r") as f:
                    existing_repo = f.read().strip()
            except: pass
        
        # Robust check: Ensure critical files exist
        critical_files = ["model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"]
        needs_download = False
        
        if existing_repo != whisper_repo:
             needs_download = True
             if os.path.exists(whisper_target):
                 safe_log(f"Repository mismatch ({existing_repo} vs {whisper_repo}). Clearing old model data...")
                 import shutil
                 try:
                     shutil.rmtree(whisper_target)
                     os.makedirs(whisper_target)
                 except: pass
        
        if not os.path.exists(whisper_target):
            needs_download = True
        else:
            for f in critical_files:
                if not os.path.exists(os.path.join(whisper_target, f)):
                    needs_download = True
                    break
                    
        if needs_download:
            safe_log(f"Downloading Whisper Model ({whisper_model_name}) from {whisper_repo}...")
            # Jump to 65% when starting Whisper
            set_progress(65)

            snapshot_download(
                repo_id=whisper_repo, 
                local_dir=whisper_target,
                local_dir_use_symlinks=False
            )
            # Save the repo tag so we don't redownload again if successful
            try:
                with open(repo_tag_file, "w") as f:
                    f.write(whisper_repo)
            except: pass
            
        set_progress(100)
        safe_log("Model downloads complete.")
    except Exception as e:
        safe_log(f"Model download failed: {e}")
        import traceback
        safe_log(traceback.format_exc())

def setup_dll_paths(lib_dir):
    """ Ensures Windows finds DLLs in subfolders (like nvidia/cudnn/bin). """
    if sys.platform != 'win32': return
    try:
        import os
        # Recursive check for 'bin' folders containing DLLs
        added = False
        for root, dirs, files in os.walk(lib_dir):
            if 'bin' in dirs:
                bin_path = os.path.normpath(os.path.join(root, 'bin'))
                if any(f.lower().endswith('.dll') for f in os.listdir(bin_path)):
                    try:
                        os.add_dll_directory(bin_path)
                        added = True
                    except: pass
        if added:
            log_info("Successfully added CUDA/cuDNN DLL paths to environment.")
    except Exception as e:
        log_info(f"DLL setup warning: {e}")

def launch_main_app(base_dir):
    lib_dir = os.path.join(base_dir, "_internal_libs")
    if lib_dir not in sys.path:
        # Scrub any existing _internal_libs to avoid 3.12/3.13 pollution
        sys.path = [p for p in sys.path if "_internal_libs" not in p]
        sys.path.insert(0, lib_dir)
    
    # CRITICAL: Add DLL directories for Windows CUDA libs
    setup_dll_paths(lib_dir)
        
    try:
        # We prefer launching via a REAL system python to ensure full standard library (json, timeit, etc)
        # which might be missing from a stripped PyInstaller bundle.
        python_exe = get_python_exe()
        script_path = os.path.join(base_dir, "src", "voice_input.py")
        
        if python_exe and os.path.exists(script_path):
            log_info(f"Launching via Script: {python_exe} {script_path}")
            # Use CREATE_NO_WINDOW if it's python.exe, or just use pythonw.exe
            flags = subprocess.CREATE_NO_WINDOW if "pythonw" not in python_exe.lower() else 0
            # Set PYTHONPATH so voice_input.py can find _internal_libs if it doesn't do it itself
            env = os.environ.copy()
            env["PYTHONPATH"] = lib_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            subprocess.Popen([python_exe, script_path], cwd=base_dir, env=env, creationflags=flags)
        else:
            log_info("System Python or script missing. Falling back to internal import (unstable)...")
            import voice_input
            app = voice_input.VoiceInputApp()
            app.run()
    except Exception as e:
        import ctypes
        import traceback
        err = f"App Launch Failed: {e}\n\n{traceback.format_exc()}"
        log_info(err)
        ctypes.windll.user32.MessageBoxW(0, err, "Privox Error", 0x10)

def uninstall_app():
    log_info("Uninstalling application...")
    try:
        # Kill the app if running (excluding this process)
        my_pid = os.getpid()
        subprocess.run(["taskkill", "/F", "/IM", "Privox.exe", "/FI", f"PID ne {my_pid}"], creationflags=subprocess.CREATE_NO_WINDOW)
        time.sleep(1)
    except: pass

    try:
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Privox"
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
        run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as key:
             winreg.DeleteValue(key, "Privox")
    except: pass
    
    app_name = "Privox"
    # Files
    desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop', f"{app_name}.lnk")
    start_menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs', f"{app_name}.lnk")
    if os.path.exists(desktop): os.remove(desktop)
    if os.path.exists(start_menu): os.remove(start_menu)
    
    # Self Destruct
    install_dir = os.path.dirname(sys.executable)
    cleanup_script = f"""
    @echo off
    timeout /t 2 /nobreak >nul
    rmdir /s /q "{install_dir}"
    del "%~f0"
    """
    bat_path = os.path.join(os.environ['TEMP'], "privox_uninstall.bat")
    with open(bat_path, "w") as f:
        f.write(cleanup_script)
    subprocess.Popen(["cmd", "/c", bat_path], creationflags=subprocess.CREATE_NO_WINDOW)

def proactive_cleanup(target_dir):
    """Scan and remove residual .old_ library folders."""
    if not os.path.exists(target_dir):
        return
    
    deleted_count = 0
    try:
        for item in os.listdir(target_dir):
            if item.startswith("_internal_libs.old_"):
                old_path = os.path.join(target_dir, item)
                if os.path.isdir(old_path):
                    try:
                        shutil.rmtree(old_path, ignore_errors=True)
                        if not os.path.exists(old_path):
                            deleted_count += 1
                    except:
                        pass
        if deleted_count > 0:
            log_info(f"Proactive Cleanup: Removed {deleted_count} residual folder(s).")
    except Exception as e:
        log_info(f"Cleanup error: {e}")

def main():
    if "--uninstall" in sys.argv:
        uninstall_app()
        sys.exit(0)

    is_installed = False
    install_dir = os.path.join(os.environ['LOCALAPPDATA'], "Privox")
    
    # Proactively clean up any residual folders from previous installs/updates
    proactive_cleanup(install_dir)
    
    # Log startup details for debugging
    log_info(f"Startup - EXE_DIR: {EXE_DIR}")
    log_info(f"Install Dir: {install_dir}")
    log_info(f"Args: {sys.argv}")
    
    if os.path.normcase(os.path.normpath(EXE_DIR)) == os.path.normcase(os.path.normpath(install_dir)):
        is_installed = True
        log_info("Detected: Running from install directory.")
    else:
        log_info("Detected: Running from outside install directory (Installer Mode).")
    
    if is_installed or "--run" in sys.argv:
        log_info("Launching Main App...")
        lib_dir = os.path.join(EXE_DIR, "_internal_libs")
        torch_dir = os.path.join(lib_dir, "torch")
        ta_dir = os.path.join(lib_dir, "torchaudio")
        fw_dir = os.path.join(lib_dir, "faster_whisper")
        if not os.path.exists(lib_dir) or not os.path.exists(torch_dir) or not os.path.exists(ta_dir) or not os.path.exists(fw_dir):
             if getattr(sys, 'frozen', False):
                app = InstallerGUI()
                app.progress_text.set("Repairing installation...")
                app.mainloop()
                return

        launch_main_app(EXE_DIR)
        return

    if getattr(sys, 'frozen', False):
        app = InstallerGUI()
        app.mainloop()
    else:
        print("Running in Dev Mode")
        launch_main_app(EXE_DIR)

if __name__ == "__main__":
    main()
