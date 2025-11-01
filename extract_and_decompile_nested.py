# extract_and_decompile_nested.py
# Usage: python extract_and_decompile_nested.py <file.marshal>
# This version sanitizes code object names for Windows filenames and logs attempts.
import sys, os, marshal, importlib.util, struct, time, subprocess, re, io
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python extract_and_decompile_nested.py <file.marshal>")
    sys.exit(1)

infile = Path(sys.argv[1])
if not infile.exists():
    print("File not found:", infile)
    sys.exit(1)

BASE_OUT = Path("nested_extraction")
BASE_OUT.mkdir(exist_ok=True)

MAGIC = importlib.util.MAGIC_NUMBER  # interpreter magic

data = infile.read_bytes()
try:
    top_co = marshal.loads(data)
except Exception as e:
    print("Error loading marshal:", e)
    sys.exit(1)

seen_signatures = set()

def sanitize_name(name: str) -> str:
    # remove characters invalid in Windows filenames and trim length
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    s = s.strip().rstrip('. ')
    if not s:
        s = "code"
    # avoid reserved names like CON, PRN, etc. (very unlikely)
    if s.upper() in {"CON","PRN","AUX","NUL","COM1","COM2","LPT1","LPT2"}:
        s = "_" + s
    return s[:64]

def is_codeobj(obj):
    return isinstance(obj, type((lambda:0).__code__))

def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8", errors="replace")

def try_decompile(vf: Path, decompiled_dir: Path, logs_dir: Path):
    # attempt decompyle3 then uncompyle6, capture outputs to logs
    decompiled_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    # decompyle3
    cmd = f"decompyle3 -o \"{decompiled_dir}\" \"{vf}\""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    write_text(logs_dir / (vf.name + ".decompyle3.stdout.txt"), p.stdout)
    write_text(logs_dir / (vf.name + ".decompyle3.stderr.txt"), p.stderr)
    if p.returncode == 0:
        produced = list(decompiled_dir.glob("**/*.py"))
        if produced:
            return True, "decompyle3"
    # fallback uncompyle6
    cmd2 = f"uncompyle6 -o \"{decompiled_dir}\" \"{vf}\""
    p2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
    write_text(logs_dir / (vf.name + ".uncompyle6.stdout.txt"), p2.stdout)
    write_text(logs_dir / (vf.name + ".uncompyle6.stderr.txt"), p2.stderr)
    if p2.returncode == 0:
        # uncompyle6 sometimes writes to stdout; check for .py in folder or stdout content
        produced = list(decompiled_dir.glob("**/*.py"))
        if produced:
            return True, "uncompyle6"
        if p2.stdout.strip():
            # write stdout to a .py as fallback
            (decompiled_dir / (vf.stem + "_stdout.py")).write_text(p2.stdout, encoding="utf-8", errors="replace")
            return True, "uncompyle6_stdout"
    return False, None

collected = []

def recurse_collect(co, parent_index):
    # signature to avoid duplicates: (name, firstlineno, consts length)
    sig = (getattr(co, "co_name", ""), co.co_firstlineno, len(co.co_consts))
    if sig in seen_signatures:
        return
    seen_signatures.add(sig)

    idx = len(collected)
    name = getattr(co, "co_name", "code")
    safe_name = sanitize_name(name)
    folder_name = f"{idx:03d}_{safe_name}"
    outdir = BASE_OUT / folder_name
    outdir.mkdir(parents=True, exist_ok=True)

    # write marshal
    mdata = marshal.dumps(co)
    mpath = outdir / (folder_name + ".marshal")
    mpath.write_bytes(mdata)

    # write disassembly for inspection
    try:
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        import dis
        dis.dis(co)
        sys.stdout = old
        dis_txt = buf.getvalue()
    except Exception as e:
        dis_txt = f"disassembly failed: {e}"
    write_text(outdir / (folder_name + ".dis.txt"), dis_txt)

    # Create .pyc header variants
    ts = int(time.time()) & 0xFFFFFFFF
    # Legacy header (magic + timestamp)
    hdrA = MAGIC + struct.pack("<I", ts)
    (outdir / (folder_name + "_legacy.pyc")).write_bytes(hdrA + mdata)
    # PEP552 flags=0 (magic + flags + ts + size)
    flags0 = 0
    hdrB = MAGIC + struct.pack("<I", flags0) + struct.pack("<I", ts) + struct.pack("<I", len(mdata))
    (outdir / (folder_name + "_pep552_ts_sz.pyc")).write_bytes(hdrB + mdata)
    # PEP552 flags=1 (magic + flags + 8-byte zero-hash)
    flags1 = 1
    hdrC = MAGIC + struct.pack("<I", flags1) + (b"\x00" * 8)
    (outdir / (folder_name + "_pep552_hash0.pyc")).write_bytes(hdrC + mdata)
    # magic + 4 zero bytes
    hdrD = MAGIC + (b"\x00" * 4)
    (outdir / (folder_name + "_magic_zero_ts.pyc")).write_bytes(hdrD + mdata)

    # try decompile variants (log results)
    logs_dir = outdir / "logs"
    decomp_dir = outdir / "decompiled"
    variant_files = [
        outdir / (folder_name + "_pep552_hash0.pyc"),
        outdir / (folder_name + "_pep552_ts_sz.pyc"),
        outdir / (folder_name + "_legacy.pyc"),
        outdir / (folder_name + "_magic_zero_ts.pyc"),
    ]
    decompiled_success = False
    for vf in variant_files:
        if not vf.exists():
            continue
        ok, method = try_decompile(vf, decomp_dir, logs_dir)
        if ok:
            decompiled_success = True
            write_text(outdir / "decompile_result.txt", f"SUCCESS using {method} on {vf.name}\n")
            break
    if not decompiled_success:
        write_text(outdir / "decompile_result.txt", "FAILED (see logs)\n")

    collected.append({"index": idx, "name": name, "safe_name": safe_name, "folder": str(outdir), "decompiled": decompiled_success})
    # recurse into nested code objects
    for c in co.co_consts:
        if is_codeobj(c):
            recurse_collect(c, idx)

# start
recurse_collect(top_co, -1)

# summary
summary_text = [
    f"Input: {infile}",
    f"Found objects: {len(collected)}",
    "Folders written under: nested_extraction/",
]
for c in collected:
    summary_text.append(f" - {c['index']:03d}: name={c['name']} -> folder={c['folder']} decompiled={c['decompiled']}")
write_text(BASE_OUT / "summary.txt", "\n".join(summary_text))
print("\n".join(summary_text))
print("Done. Inspect nested_extraction/* for .py files, .dis.txt and logs.")
