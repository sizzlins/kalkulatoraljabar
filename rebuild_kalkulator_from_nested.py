#!/usr/bin/env python3
"""
Rebuild a single Python source file from nested_extraction contents.

Requirements:
  - Run in your venv where decompyle3 (preferred) and uncompyle6 are installed.
  - Place this script in the same folder that contains the extracted nested_extraction folder.
Outputs:
  - kalkulator_reconstructed.py (single file)
  - decompile_logs/ (stdout/stderr for failing objects)
"""
import sys, os, subprocess, importlib.util, marshal, struct, time, json
from pathlib import Path

ROOT = Path.cwd()
NESTED_DIR = ROOT / "nested_extraction_unzipped"
if not NESTED_DIR.exists():
    # fallback to 'nested_extraction' if previous naming
    if (ROOT / "nested_extraction").exists():
        NESTED_DIR = ROOT / "nested_extraction"
    else:
        print("Could not find nested_extraction_unzipped or nested_extraction in", ROOT)
        sys.exit(1)

OUT_FILE = ROOT / "kalkulator_reconstructed.py"
LOG_DIR = ROOT / "decompile_logs"
LOG_DIR.mkdir(exist_ok=True)

# helper: try to run a command, capture output
def run_cmd(cmd):
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

# 1) collect all code-object folders and marshal files
entries = []
for folder in sorted(NESTED_DIR.iterdir()):
    if not folder.is_dir():
        continue
    # find marshal or pyc variants
    marshal_files = list(folder.glob("*.marshal"))
    pyc_variants = list(folder.glob("*.pyc"))
    dis_file = next(iter(folder.glob("*.dis.txt")), None)
    # prefer actual decompiled .py if it contains code (unlikely here), but include it for reference
    decomp_file = next(iter(folder.glob("decompiled/*.py")), None)
    entries.append({
        "folder": folder,
        "marshal": marshal_files[0] if marshal_files else None,
        "pyc_variants": pyc_variants,
        "dis": dis_file,
        "decomp": decomp_file,
    })

print(f"Found {len(entries)} recovered code object folders.")

# 2) helper to write .pyc wrappers from .marshal if needed
MAGIC = importlib.util.MAGIC_NUMBER
def write_pyc_variants_from_marshal(marshal_path: Path):
    """Return list of variant file paths (written) for this marshal."""
    data = marshal_path.read_bytes()
    ts = int(time.time()) & 0xFFFFFFFF
    out_base = marshal_path.with_suffix("")  # remove .marshal
    variants = []
    # legacy header
    hdrA = MAGIC + struct.pack("<I", ts)
    pA = out_base.with_suffix(".legacy.pyc")
    pA.write_bytes(hdrA + data); variants.append(pA)
    # pep552 ts+size
    hdrB = MAGIC + struct.pack("<I", 0) + struct.pack("<I", ts) + struct.pack("<I", len(data))
    pB = out_base.with_suffix(".pep552_ts_sz.pyc")
    pB.write_bytes(hdrB + data); variants.append(pB)
    # pep552 hash0
    hdrC = MAGIC + struct.pack("<I", 1) + (b"\x00"*8)
    pC = out_base.with_suffix(".pep552_hash0.pyc")
    pC.write_bytes(hdrC + data); variants.append(pC)
    # magic + zeros
    hdrD = MAGIC + (b"\x00"*4)
    pD = out_base.with_suffix(".magic0.pyc")
    pD.write_bytes(hdrD + data); variants.append(pD)
    return variants

# 3) decompile each code object: prefer decompyle3, fallback to uncompyle6
decompiled_by_name = {}  # name -> source text
order_list = []  # will store (folder, module_source or function source, entry_flag)
failures = []

for e in entries:
    folder = e["folder"]
    base_name = folder.name
    print("Processing", base_name)
    # ensure we have at least one .pyc to feed to decompiler
    pyc_candidates = list(e["pyc_variants"])  # copies
    if e["marshal"]:
        pyc_candidates.extend(write_pyc_variants_from_marshal(e["marshal"]))
    # also consider existing pyc variants already named in folder
    pyc_candidates = [Path(p) for p in pyc_candidates if p and Path(p).exists()]
    if not pyc_candidates:
        print("  No marshal or pyc variants for", base_name, " — skipping (check .dis.txt)")
        failures.append((base_name, "no_marshal_or_pyc"))
        continue

    decompiled_success = False
    # try each pyc candidate until one decompiles to non-trivial source
    for pc in pyc_candidates:
        # try decompyle3
        cmd = f"decompyle3 -o \"{folder / 'decompiled_auto'}\" \"{pc}\""
        rc, out, err = run_cmd(cmd)
        (LOG_DIR / f"{base_name}_decompyle3.stdout.txt").write_text(out + "\n")
        (LOG_DIR / f"{base_name}_decompyle3.stderr.txt").write_text(err + "\n")
        if rc == 0:
            # check produced files
            produced = list((folder / 'decompiled_auto').glob("**/*.py"))
            if produced:
                # prefer the first produced file content
                src = produced[0].read_text(encoding='utf-8', errors='replace')
                # consider successful only if produced content has more than header (heuristic)
                if len(src.strip()) > 50 and ("def " in src or "class " in src or "__name__" in src or "if __name__" in src):
                    decompiled_by_name[base_name] = src
                    decompiled_success = True
                    break
                else:
                    # still take it as fallback but continue trying other pyc candidates
                    decompiled_by_name.setdefault(base_name, src)
                    # continue loop to try other candidates
        # fallback to uncompyle6
        outp = folder / "decompiled_uncompyle6.py"
        cmd2 = f"uncompyle6 -o \"{outp}\" \"{pc}\""
        rc2, out2, err2 = run_cmd(cmd2)
        (LOG_DIR / f"{base_name}_uncompyle6.stdout.txt").write_text(out2 + "\n")
        (LOG_DIR / f"{base_name}_uncompyle6.stderr.txt").write_text(err2 + "\n")
        if rc2 == 0:
            # uncompyle6 sometimes outputs to outp file or to stdout
            if outp.exists() and outp.stat().st_size > 0:
                src = outp.read_text(encoding='utf-8', errors='replace')
                if len(src.strip()) > 50:
                    decompiled_by_name[base_name] = src
                    decompiled_success = True
                    break
            elif out2.strip():
                decompiled_by_name[base_name] = out2
                decompiled_success = True
                break
    if not decompiled_success:
        print("  Failed automated decompilation for", base_name, "- will use disassembly fallback")
        failures.append((base_name, "decompilation_failed"))

# If any failed, we will collect their disassembly texts and embed verbatim for manual reconstruction
# 4) Build output ordering: use the module object folder (000__module_) to determine ordering.
module_folder = None
for e in entries:
    if e["folder"].name.startswith("000_"):
        module_folder = e["folder"]
        break

ordered_blocks = []
# If module folder has decompiled text, try to parse its order by scanning for import lines and def/class order.
if module_folder and module_folder.name in decompiled_by_name:
    modsrc = decompiled_by_name[module_folder.name]
    ordered_blocks.append(("module", modsrc))
else:
    # fallback: include module disassembly or decompiled header
    if module_folder:
        disf = module_folder.glob("*.dis.txt")
        disf = next(disf, None)
        if disf:
            ordered_blocks.append(("module_dis", disf.read_text(encoding='utf-8', errors='replace')))
        else:
            # nothing found
            pass

# Append all other decompiled sources in alphanumeric folder order
for e in entries:
    name = e["folder"].name
    if name == module_folder.name if module_folder else False:
        continue
    src = decompiled_by_name.get(name)
    if src:
        ordered_blocks.append((name, src))
    else:
        # fallback: include disassembly text if available
        disf = next(iter(e["folder"].glob("*.dis.txt")), None)
        if disf:
            ordered_blocks.append((name + "_dis", disf.read_text(encoding='utf-8', errors='replace')))
        else:
            ordered_blocks.append((name + "_empty", "# EMPTY - no decompilation or disassembly available for " + name))

# 5) write single output file combining blocks in order (no code modifications)
with OUT_FILE.open("w", encoding="utf-8", errors="replace") as f:
    f.write("# Reconstructed source: kalkulator_reconstructed.py\n")
    f.write("# Generated by rebuild_kalkulator_from_nested.py\n\n")
    for name, block in ordered_blocks:
        f.write(f"# --- block: {name} ---\n")
        # if the block is disassembly text (not python), write as comment lines so the file remains valid Python
        if block.lstrip().startswith("  ") or "LOAD_CONST" in block[:120]:
            # it's disassembly; embed as triple-quoted string comment to preserve data but keep file runnable
            f.write('"""\n')
            f.write(block)
            f.write('\n"""\n\n')
        else:
            f.write(block)
            f.write("\n\n")
print("Wrote reconstructed file to", OUT_FILE)
print("Decompilation failures (if any) written to decompile_logs/ and included as disassembly blocks in the result:", failures)
if failures:
    print("One or more code objects failed automated decompilation. Paste the entries (from decompile_logs) here and I'll manually reconstruct them.")
else:
    print("All code objects decompiled successfully; output file contains the full reconstructed source blocks.")
