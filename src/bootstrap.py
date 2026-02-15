import os
import shutil
import sys
import subprocess
import threading
import time
import logging
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import ctypes
import zipfile
import urllib.request

# Disable Symlinks for Windows (Fixes WinError 1314)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Ensure we are in the same directory as the executable
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(sys.executable)
    os.chdir(EXE_DIR)
else:
    EXE_DIR = os.path.dirname(os.path.abspath(__file__))
    if EXE_DIR.endswith('src'):
        EXE_DIR = os.path.dirname(EXE_DIR)
    os.chdir(EXE_DIR)

# --- Logging Setup ---
log_level = logging.INFO
log_format = '%(asctime)s - %(levelname)s - %(message)s'

handlers = []
# Only add StreamHandler if we have a real stdout/stderr (i.e. not running with --noconsole)
# This prevents 'NoneType has no attribute write' and potential recursion
if sys.stdout:
     handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    format=log_format,
    level=log_level,
    force=True,
    handlers=handlers
)

# LoggerWriter removed to prevent recursion in noconsole mode

def log_info(msg):
    print(msg)

log_info("--- Privox Setup Started (Pixi Mode) ---")
log_info(f"Platform: {sys.platform}")

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
        default_path = self.get_existing_install_path()
        if not default_path:
            default_path = os.path.join(os.environ.get('LOCALAPPDATA', os.environ.get('USERPROFILE', 'C:\\')), "Privox")
        
        self.install_dir = tk.StringVar(value=default_path)
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

    def get_existing_install_path(self):
        try:
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Privox"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                install_dir, _ = winreg.QueryValueEx(key, "InstallLocation")
                if os.path.exists(install_dir):
                    return os.path.normpath(install_dir)
        except: pass
        return None

    def create_pages(self):
        # --- Welcome Page ---
        p1 = ttk.Frame(self.main_content)
        
        ttk.Label(p1, text="Welcome to Privox Setup", font=("Segoe UI", 16, "bold")).pack(pady=(10, 20))
        ttk.Label(p1, text="Privox uses Pixi for robust, isolated environment management.\nThis wizard will guide you through the setup process.", justify=tk.CENTER).pack(pady=10)
        
        # Path Selection
        path_frame = ttk.LabelFrame(p1, text="Installation Destination", padding=15)
        path_frame.pack(fill=tk.X, pady=10)
        
        path_input_frame = ttk.Frame(path_frame)
        path_input_frame.pack(fill=tk.X)
        
        self.entry_path = ttk.Entry(path_input_frame, textvariable=self.install_dir)
        self.entry_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(path_input_frame, text="Browse...", command=self.browse_path).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(path_frame, text="Privox will be installed in the folder above.", font=("Segoe UI", 8), foreground="gray").pack(anchor=tk.W, pady=(5, 0))

        self.pages["welcome"] = p1
        
        # --- Progress Page ---
        p2 = ttk.Frame(self.main_content)
        ttk.Label(p2, text="Installing Privox...", font=("Segoe UI", 14, "bold")).pack(pady=(10, 20))
        
        self.log_box = tk.Text(p2, height=15, state=tk.DISABLED, font=("Consolas", 8))
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

    def browse_path(self):
        new_dir = filedialog.askdirectory(initialdir=self.install_dir.get(), title="Select Installation Folder")
        if new_dir:
            self.install_dir.set(os.path.normpath(new_dir))

    def on_cancel(self):
        if self.current_page_name == "progress":
            if not messagebox.askyesno("Cancel Setup", "Do you want to stop the installation?"):
                return
            
            self.is_cancelling = True
            if self.active_process:
                try:
                    self.log("Stopping active processes...")
                    # Force kill process tree (handle pixi -> python grandchildren)
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.active_process.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        capture_output=True
                    )
                except: pass
        
        self.destroy()
        os._exit(0) # Force exit main process

    def log(self, msg):
        log_info(msg)
        self.after(0, self._log_ui, msg)

    def _log_ui(self, msg):
        self.log_box.config(state=tk.NORMAL)
        # Auto-scroll if at bottom
        # output is often multiline
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def start_installation(self):
        # 1. Disk Space Check
        target_dir = os.path.normpath(self.install_dir.get())
        drive = os.path.splitdrive(target_dir)[0] or "C:"
        
        try:
            total, used, free = shutil.disk_usage(drive)
            free_gb = free / (1024**3)
            log_info(f"Disk Check: {drive} has {free_gb:.2f} GB free.")
            
            if free_gb < 15.0:
                msg = (f"The selected drive ({drive}) has only {free_gb:.1f} GB of free space.\n\n"
                       f"Privox require approximately 13-15 GB for AI models and dependencies.\n"
                       f"You may run out of disk space during installation.\n\n"
                       f"Do you want to continue anyway?")
                if not messagebox.askyesno("Low Disk Space", msg):
                    return
        except Exception as e:
            log_info(f"Disk check failed: {e}")
            # Non-critical, just log and continue
            
        self.show_page("progress")
        threading.Thread(target=self.run_install_process, daemon=True).start()

    def run_install_process(self):
        try:
            log_info("Starting installation thread...")
            target_dir = os.path.normpath(self.install_dir.get())
            self.target_dir = target_dir
            self.target_exe = os.path.join(target_dir, "Privox.exe")
            
            # 1. Install Files (EXE)
            self.log(f"Installing application files to {target_dir}...")
            new_path = install_app_files(target_dir, self.log)
            if not new_path:
                self.log("Installation failed (File Copy Error).")
                self.after(0, self.show_failure_state)
                return
            
            # Switch context to installed location if needed
            self.pixi_exe = ensure_pixi(target_dir, self.log)
            if not self.pixi_exe:
                self.log("Failed to setup Pixi.")
                self.after(0, self.show_failure_state)
                return

            # 2. Run Pixi Install
            self.log("Setting up environment (this may take a while)...")
            
            # Force verbose to see what's happening
            success = run_pixi_command(self, [self.pixi_exe, "install", "-v"], cwd=target_dir)
            if not success:
                self.log("Environment setup failed!")
                self.after(0, self.show_failure_state)
                return
            
            # 3. Download Models
            # We run the script inside the pixi environment
            self.log("Checking AI Models...")
            model_script = os.path.join(target_dir, "src", "download_models.py")
            if os.path.exists(model_script):
                success = run_pixi_command(self, [self.pixi_exe, "run", "python", "src/download_models.py"], cwd=target_dir)
                if not success:
                    self.log("Model download failed (non-critical, can retry later).")
            else:
                 self.log("Model setup script missing. Skipping.")

            # Register Uninstaller
            self.log("Registering uninstaller...")
            register_uninstaller(target_dir, self.target_exe)

            self.log("Setup Finished.")
            
            # 4. Finalize
            self.after(500, lambda: self.show_page("success"))
            
        except Exception as e:
            msg = f"A fatal error occurred during installation:\n\n{e}"
            log_info(f"FATAL INSTALL ERROR: {e}")
            self.after(0, self.show_failure_state)

    def show_failure_state(self):
        self.btn_next.config(text="Exit", state=tk.NORMAL, command=self.destroy)
        self.btn_cancel.config(state=tk.DISABLED)
        messagebox.showerror("Installation Failed", "Privox setup could not complete.\nCheck the logs for details.")

    def launch_and_exit(self):
        try:
            # We always launch the installed EXE now
            subprocess.Popen([self.target_exe, "--run"])
            self.destroy()
            sys.exit(0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch: {e}")

# --- Helper Functions ---

def ensure_pixi(base_dir, log_callback):
    """ Checks for local pixi executable, downloads if missing. """
    internal_dir = os.path.join(base_dir, "_internal")
    pixi_dir = os.path.join(internal_dir, "pixi")
    pixi_exe = os.path.join(pixi_dir, "pixi.exe")
    
    if os.path.exists(pixi_exe):
        log_callback("Pixi detected.")
        return pixi_exe
        
    log_callback("Downloading Pixi (Standalone packages)...")
    if not os.path.exists(pixi_dir):
        os.makedirs(pixi_dir)
        
    url = "https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.zip"
    zip_path = os.path.join(pixi_dir, "pixi.zip")
    
    try:
        def reporthook(blocknum, blocksize, totalsize):
            pass # simplified progress
            
        urllib.request.urlretrieve(url, zip_path, reporthook)
        log_callback("Extracting Pixi...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(pixi_dir)
            
        os.remove(zip_path)
            
        if os.path.exists(pixi_exe):
             log_callback("Pixi installed successfully.")
             return pixi_exe
        else:
             log_callback("Error: pixi.exe not found in extracted archive.")
             return None
    except Exception as e:
        log_callback(f"Pixi download error: {e}")
        return None

def run_pixi_command(gui_instance, cmd, cwd):
    """ Runs a pixi command and streams output to GUI log. """
    log_callback = gui_instance.log
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        process = subprocess.Popen(
            cmd, 
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            encoding='utf-8',
            errors='replace',
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        gui_instance.active_process = process
        
        for line in process.stdout:
            # Pixi output can be verbose, maybe filter?
            # For now, show everything as requested
            msg = line.strip()
            if msg:
                if log_callback: log_callback(msg)

        process.wait()
        gui_instance.active_process = None
        
        return process.returncode == 0
    except Exception as e:
        if log_callback: log_callback(f"Command Error: {e}")
        return False

def install_app_files(target_dir, log_callback=None):
    """ Copies the EXE and resources to the target directory. Does NOT launch the app. """
    try:
        app_name = "Privox"
        exe_name = "Privox.exe"
        target_exe = os.path.join(target_dir, exe_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        current_exe = sys.executable
        if log_callback: log_callback(f"Terminating old instances...")
        try:
            my_pid = os.getpid()
            subprocess.run(["taskkill", "/F", "/IM", "Privox.exe", "/FI", f"PID ne {my_pid}"], 
                           creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            time.sleep(1.0) 
        except: pass
        
        if log_callback: log_callback(f"Copying to {target_dir}...")
        
        import shutil
        if os.path.normpath(current_exe) != os.path.normpath(target_exe):
            try:
                shutil.copy2(current_exe, target_exe)
            except Exception as e:
                # If we can't overwrite the exe, it might be running.
                if log_callback: log_callback(f"Copy Warning (continuing): {e}")
        
        # Copy Assets & Config
        config_path = os.path.join(EXE_DIR, "config.json")
        if not os.path.exists(config_path) and getattr(sys, 'frozen', False):
             config_path = os.path.join(sys._MEIPASS, "config.json")
             
        if os.path.exists(config_path):
             shutil.copy2(config_path, os.path.join(target_dir, "config.json"))

        # Copy pixi.toml (CRITICAL)
        pixi_toml = os.path.join(EXE_DIR, "pixi.toml")
        if not os.path.exists(pixi_toml) and getattr(sys, 'frozen', False):
             pixi_toml = os.path.join(sys._MEIPASS, "pixi.toml")
        
        if os.path.exists(pixi_toml):
            shutil.copy2(pixi_toml, os.path.join(target_dir, "pixi.toml"))
            
        # Copy pixi.lock if exists
        pixi_lock = os.path.join(EXE_DIR, "pixi.lock")
        if not os.path.exists(pixi_lock) and getattr(sys, 'frozen', False):
             pixi_lock = os.path.join(sys._MEIPASS, "pixi.lock")
             
        if os.path.exists(pixi_lock):
            shutil.copy2(pixi_lock, os.path.join(target_dir, "pixi.lock"))

        for folder in ["models", "assets"]: 
            src = os.path.join(EXE_DIR, folder)
            if not os.path.exists(src) and getattr(sys, 'frozen', False):
                 src = os.path.join(sys._MEIPASS, folder)
            
            dst = os.path.join(target_dir, folder)
            if os.path.exists(src):
                if log_callback: log_callback(f"Merging {folder}...")
                if folder == "models":
                    if not os.path.exists(dst): os.makedirs(dst)
                    for item in os.listdir(src):
                        s = os.path.join(src, item)
                        d = os.path.join(dst, item)
                        if os.path.isdir(s):
                            if not os.path.exists(d): shutil.copytree(s, d)
                        else:
                            if not os.path.exists(d): shutil.copy2(s, d)
                else:
                    if os.path.exists(dst): 
                        try: shutil.rmtree(dst)
                        except: pass
                    shutil.copytree(src, dst)
        
        # Copy Source Code (needed for pixi run python ...)
        src_dir = os.path.join(target_dir, "src")
        if not os.path.exists(src_dir): os.makedirs(src_dir)
        
        for script in ["voice_input.py", "download_models.py"]:
            script_src = os.path.join(EXE_DIR, "src", script)
            if not os.path.exists(script_src) and getattr(sys, 'frozen', False):
                 script_src = os.path.join(sys._MEIPASS, "src", script)
            if os.path.exists(script_src):
                shutil.copy2(script_src, os.path.join(src_dir, script))

        # Shortcuts
        if log_callback: log_callback("Creating Shortcuts...")
        create_shortcut(target_exe, target_dir)
        
        return target_exe
    except Exception as e:
        if log_callback: log_callback(f"Install error: {e}")
        return None

def create_shortcut(target_exe, target_dir):
    try:
        app_name = "Privox"
        icon_path = os.path.join(target_dir, "assets", "icon.ico")
        start_menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
        if not os.path.exists(start_menu): os.makedirs(start_menu)
        
        shortcut_path = os.path.join(start_menu, f"{app_name}.lnk")
        
        vbs_script = f"""
        Set oWS = WScript.CreateObject("WScript.Shell")
        sLinkFile = "{shortcut_path}"
        Set oLink = oWS.CreateShortcut(sLinkFile)
        oLink.TargetPath = "{target_exe}"
        oLink.Description = "Privox AI Voice Input"
        oLink.WorkingDirectory = "{target_dir}"
        oLink.IconLocation = "{icon_path}"
        oLink.Save
        """
        vbs_file = os.path.join(os.environ['TEMP'], f"mkshortcut_{os.getpid()}.vbs")
        with open(vbs_file, "w") as f:
            f.write(vbs_script)
        subprocess.call(["cscript", "//nologo", vbs_file], creationflags=subprocess.CREATE_NO_WINDOW)
        os.remove(vbs_file)
    except: pass

import winreg

def is_app_running():
    if sys.platform != 'win32': return None
    mutex_name = "Global\\Privox_SingleInstance_Mutex"
    handle = ctypes.windll.kernel32.CreateMutexW(None, False, mutex_name)
    last_error = ctypes.windll.kernel32.GetLastError()
    if last_error == 183: # ERROR_ALREADY_EXISTS
        return None
    return handle

def register_uninstaller(install_dir, exe_path):
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Privox"
        icon_path = os.path.join(install_dir, "assets", "icon.ico")
        
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Privox")
            winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, icon_path)
            winreg.SetValueEx(key, "DisplayVersion", 0, winreg.REG_SZ, "1.0.0")
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, f"\"{exe_path}\" --uninstall")
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Privox")
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, install_dir)
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
            
        log_info("Uninstaller registered in Registry.")
    except Exception as e:
        log_info(f"Failed to register uninstaller: {e}")

def uninstall_app():
    if not messagebox.askyesno("Uninstall Privox", "Are you sure you want to completely remove Privox?"):
        sys.exit(0)
        
    install_dir = os.path.dirname(sys.executable)
    
    # 0. Kill Running Instances
    try:
        # Kill Python (The App Logic)
        subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/T"], 
                       creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
        
        # Kill Other Privox Instances (The Launcher), excluding self
        my_pid = os.getpid()
        subprocess.run(["taskkill", "/F", "/FI", f"PID ne {my_pid}", "/IM", "Privox.exe"], 
                       creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
        
        # Give it a second to release file locks
        time.sleep(1)
    except: pass

    # 1. Remove Shortcut
    try:
        start_menu = os.path.join(os.environ['APPDATA'], 'Microsoft', 'Windows', 'Start Menu', 'Programs')
        lnk = os.path.join(start_menu, "Privox.lnk")
        if os.path.exists(lnk):
            os.remove(lnk)
    except: pass
    
    # 2. Remove Registry Key
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Privox"
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, key_path)
    except: pass
    
    # 3. Self-Destruct Script
    try:
        # Create a batch file in TEMP to delete the installation folder after we exit
        temp_dir = os.environ['TEMP']
        bat_file = os.path.join(temp_dir, f"privox_nuke_{os.getpid()}.bat")
        
        # Determine the parent directory if we are in _internal (which we shouldn't be for single file, but safe check)
        # Typically sys.executable is in %LOCALAPPDATA%\Privox\Privox.exe
        target_dir = install_dir
        
        with open(bat_file, "w") as f:
            f.write("@echo off\n")
            f.write("timeout /t 3 /nobreak >nul\n") # Wait for us to exit
            f.write(f"rmdir /s /q \"{target_dir}\"\n")
            f.write("del \"%~f0\"\n") # Delete script itself
            
        subprocess.Popen([bat_file], shell=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        messagebox.showinfo("Uninstall", "Privox has been removed. Cleanup will finish in a few seconds.")
    except Exception as e:
        messagebox.showerror("Error", f"Uninstall cleanup failed: {e}")
        
    sys.exit(0)

def main():
    # Handle Uninstall Flag
    if "--uninstall" in sys.argv:
        uninstall_app()

    # Mutex Check
    mutex_handle = is_app_running()
    if not mutex_handle:
        if "--run" in sys.argv:
            sys.exit(0) # Silent exit if run mode
        else:
            # If we are double-clicked but running, maybe bring to front?
            # For now just exit/alert
            messagebox.showinfo("Privox", "Another instance of Privox is already running.")
            sys.exit(0)

    # Determine if we are in an installed state
    # Robust check: Are the local project files (Pixi, TOML) next to us?
    exe_dir = os.path.dirname(os.path.normpath(sys.executable))
    local_pixi = os.path.join(exe_dir, "_internal", "pixi", "pixi.exe")
    local_toml = os.path.join(exe_dir, "pixi.toml")
    
    is_installed = os.path.exists(local_pixi) and os.path.exists(local_toml)

    if is_installed or "--run" in sys.argv:
        # Launch Main App via Pixi
        if os.path.exists(local_pixi):
            log_info(f"Launching App via Pixi: {local_pixi} run start")
            subprocess.Popen([local_pixi, "run", "start"], cwd=exe_dir, creationflags=subprocess.CREATE_NO_WINDOW)
            sys.exit(0)
        else:
            # Installation damaged or manual run flag used without install
            # Default to GUI setup
            pass
            
    # Installer Mode
    app = InstallerGUI()
    app.mainloop()
            

if __name__ == "__main__":
    main()
