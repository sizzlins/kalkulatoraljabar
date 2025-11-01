# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Kalkulator.

Build with: pyinstaller kalkulator.spec
"""

import sys
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all SymPy and mpmath data
sympy_datas, sympy_binaries, sympy_hiddenimports = collect_all('sympy')
mpmath_datas, mpmath_binaries, mpmath_hiddenimports = collect_all('mpmath')

a = Analysis(
    ['kalkulator_minimal.py'],
    pathex=[],
    binaries=sympy_binaries + mpmath_binaries,
    datas=sympy_datas + mpmath_datas,
    hiddenimports=[
        'kalkulator_pkg',
        'kalkulator_pkg.config',
        'kalkulator_pkg.parser',
        'kalkulator_pkg.worker',
        'kalkulator_pkg.solver',
        'kalkulator_pkg.cli',
        'kalkulator_pkg.types',
        'kalkulator_pkg.logging_config',
        'kalkulator_pkg.api',
        'sympy.parsing.sympy_parser',
        'sympy.parsing',
        'sympy.solvers',
        'sympy.matrices',
        'mpmath',
    ] + sympy_hiddenimports + mpmath_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='kalkulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
