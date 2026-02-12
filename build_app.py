import PyInstaller.__main__
import os
import shutil

# Clean previous builds
if os.path.exists('build'):
    shutil.rmtree('build')
if os.path.exists('dist'):
    shutil.rmtree('dist')

print("Starting PyInstaller Build...")

PyInstaller.__main__.run([
    'src/voice_input.py',
    '--name=WisprLocal',
    '--onefile',
    '--clean',
    '--noconsole',
    '--icon=assets/icon.ico',
    # Hidden imports for dynamic dependencies
    '--hidden-import=pystray',
    '--hidden-import=pynput',
    '--hidden-import=faster_whisper',
    '--hidden-import=llama_cpp',
    '--hidden-import=torch',
    '--hidden-import=markupsafe',
    '--hidden-import=torchaudio',
    # Metadata and all data collection for complex libraries
    '--collect-all=llama_cpp',
    '--collect-all=faster_whisper',
    '--collect-all=pystray',
    '--copy-metadata=markupsafe',
    '--copy-metadata=torch',
    '--copy-metadata=tqdm',
    '--copy-metadata=regex',
    '--copy-metadata=requests',
    '--copy-metadata=packaging',
    # Exclude problematic large libraries that aren't used to keep the exe smaller
    '--exclude-module=transformers', 
    '--exclude-module=matplotlib',
    '--exclude-module=notebook',
    '--exclude-module=jedi',
    # Optimization: ignore irrelevant data
    '--collect-submodules=pynput',
])

print("Build Complete. Executable is in 'dist/WisprLocal.exe'")
