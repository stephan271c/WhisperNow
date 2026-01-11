# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Exclude massive libraries that we don't need or want to bundle (relying on system)
excluded_modules = [
    'nvidia',             # The biggest culprit (~2GB). We rely on system CUDA or CPU.
    'matplotlib',         # Not used in UI
    'tkinter',            # Not used (we use PySide6)
    'unittest',           # Dev only
    'pdb',                # Dev only
    'doctest',            # Dev only
    'IPython',            # Dev only
    'PIL',                # Not used directly (pyside handles images)
]

a = Analysis(
    ['src/transcribe/app.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Ensure critical pynput backends are found
        'pynput.keyboard._xorg',
        'pynput.mouse._xorg',
        'pynput.keyboard._win32',
        'pynput.mouse._win32',
        'pynput.keyboard._darwin',
        'pynput.mouse._darwin',
        # Torch/Audio
        'torchaudio.lib.libtorchaudio',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='whispernow',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed app (no terminal)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='whispernow',
)
