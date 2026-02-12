import os
import shutil
import sys
import glob

def log(msg):
    print(f"[CUDA Fixer] {msg}")

def fix_cuda_integration():
    """
    Copies CUDA MSBuild extensions to Visual Studio 2022 BuildCustomizations.
    This fixes 'No CUDA toolset found' errors when compiling llama-cpp-python.
    """
    try:
        # 1. Locate CUDA Toolkit MSBuild Extensions
        # Default: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions
        cuda_path = os.environ.get("CUDA_PATH", "")
        if not cuda_path:
            log("CUDA_PATH not found in environment.")
            return

        source_dir = os.path.join(cuda_path, "extras", "visual_studio_integration", "MSBuildExtensions")
        if not os.path.exists(source_dir):
            log(f"CUDA MSBuild Extensions not found at: {source_dir}")
            return

        # 2. Locate Visual Studio 2022 MSBuild Directory
        # Target: C:\Program Files\Microsoft Visual Studio\2022\*\MSBuild\Microsoft\VC\v170\BuildCustomizations
        vs_root = r"C:\Program Files\Microsoft Visual Studio\2022"
        if not os.path.exists(vs_root):
            vs_root = r"C:\Program Files (x86)\Microsoft Visual Studio\2022"
        
        if not os.path.exists(vs_root):
            log("Visual Studio 2022 directory not found.")
            return

        # Find all editions (Community, Professional, Enterprise, BuildTools)
        found_targets = []
        for edition in os.listdir(vs_root):
            edition_path = os.path.join(vs_root, edition)
            if not os.path.isdir(edition_path): continue
            
            # Construct target path
            target_dir = os.path.join(edition_path, "MSBuild", "Microsoft", "VC", "v170", "BuildCustomizations")
            
            if os.path.exists(target_dir):
                found_targets.append(target_dir)

        if not found_targets:
            log("No valid Visual Studio BuildCustomizations folders found.")
            return

        # 3. Copy Files
        files_to_copy = glob.glob(os.path.join(source_dir, "*.*"))
        
        for target_dir in found_targets:
            log(f"Patching VS instance at: {target_dir}")
            for src_file in files_to_copy:
                filename = os.path.basename(src_file)
                dst_file = os.path.join(target_dir, filename)
                
                try:
                    shutil.copy2(src_file, dst_file)
                    # log(f"  Copied {filename}")
                except PermissionError:
                    log(f"  PERMISSION DENIED: Cannot write to {target_dir}")
                    log("  Please run the application as Administrator to fix compilation.")
                    return
                except Exception as e:
                    log(f"  Error copying {filename}: {e}")

        log("CUDA Integration files copied successfully.")

    except Exception as e:
        log(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    fix_cuda_integration()
