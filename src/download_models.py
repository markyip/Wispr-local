import os
import sys
import shutil

def log(msg):
    print(f"[ModelSetup] {msg}", flush=True)

def main():
    # 0. Environment Isolation
    os.environ["PYTHONNOUSERSITE"] = "1"
    import site
    site.ENABLE_USER_SITE = False
    
    target_base_dir = os.getcwd()
    
    # Load settings from config.json if it exists
    whisper_model_name = "large-v3-turbo-cantonese" 
    whisper_repo = "JackyHoCL/whisper-large-v3-turbo-cantonese-yue-english-ct2"
    grammar_file = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    grammar_repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    
    config_path = os.path.join(target_base_dir, "config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
                whisper_model_name = config.get("whisper_model", whisper_model_name)
                whisper_repo = config.get("whisper_repo", whisper_repo)
                grammar_file = config.get("grammar_file", grammar_file)
                grammar_repo = config.get("grammar_repo", grammar_repo)
                asr_backend = config.get("asr_backend", "whisper")
            log(f"Loaded tailored settings from config.json: {whisper_model_name}")
        except Exception as e:
            log(f"Config load error (using defaults): {e}")
            asr_backend = "whisper"

    models_dir = os.path.join(target_base_dir, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    log(f"Checking AI Models (Backend: {asr_backend})...")

    # 0. SenseVoiceSmall (Alternative)
    if asr_backend == "sensevoice":
        sense_dir = os.path.join(models_dir, "SenseVoiceSmall")
        if not os.path.exists(sense_dir):
            log("Downloading SenseVoiceSmall from ModelScope...")
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                snapshot_download('iic/SenseVoiceSmall', local_dir=sense_dir)
            except ImportError:
                log("modelscope not installed. Using huggingface fallback...")
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id='iic/SenseVoiceSmall', local_dir=sense_dir)
        else:
            log("SenseVoiceSmall model present.")

    # 0. Install Llama-cpp-python with CUDA support
    llama_stable = False
    try:
        import llama_cpp
        # Attempt to initialize a dummy Llama instance with n_gpu_layers=1 to check for CUDA backend
        # Note: We need a small model file for this, or just check the system info string
        sys_info = llama_cpp.llama_print_system_info()
        if b"CUDA" in sys_info or "CUDA" in str(sys_info):
            log("llama-cpp-python is installed with CUDA support.")
            llama_stable = True
        else:
            log("llama-cpp-python is installed but CUDA support is MISSING.")
            llama_stable = False
    except ImportError:
        log("llama-cpp-python missing.")
        llama_stable = False

    if not llama_stable:
        log("Repairing llama-cpp-python (Forcing CUDA 12.4 binary wheel)...")
        import subprocess
        try:
            # Environment isolation
            env = os.environ.copy()
            env["PYTHONNOUSERSITE"] = "1"
            
            # First, try to uninstall any broken/CPU version
            subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"], env=env)
            
            # Use the "cu124" wheel index for CUDA 12.4
            # We pin 0.3.4 as it's known to have stable wheels, force binary, and skip deps
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python==0.3.4", 
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124",
                "--force-reinstall",
                "--no-cache-dir",
                "--only-binary=:all:",
                "--no-deps"
            ], env=env)
            log("llama-cpp-python binary wheel installed successfully.")
        except subprocess.CalledProcessError as e:
            log(f"CRITICAL: Failed to install llama-cpp-python (No binary wheel found?): {e}")
            pass
            
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        log("Error: huggingface_hub not installed in environment.")
        sys.exit(1)

    # 1. Grammar Model (Llama)
    if not os.path.exists(os.path.join(models_dir, grammar_file)):
        log(f"Downloading Grammar Model ({grammar_file})...")
        hf_hub_download(repo_id=grammar_repo, filename=grammar_file, local_dir=models_dir)
    else:
        log(f"Grammar Model {grammar_file} present.")

    # 2. Whisper Model (Faster-Whisper Format)
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
                log(f"Repository mismatch ({existing_repo} vs {whisper_repo}). Clearing old model data...")
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
        log(f"Downloading Whisper Model ({whisper_model_name}) from {whisper_repo}...")
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
        
    log("Model downloads complete.")

if __name__ == "__main__":
    main()
