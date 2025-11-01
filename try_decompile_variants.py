# try_decompile_variants.py
# Runs decompyle3 then uncompyle6 on each .pyc in pyc_variants,
# writes results and logs to ./decompile_attempts/
import subprocess, shlex, sys
from pathlib import Path

base = Path.cwd()
variants = base / "pyc_variants"
outdir = base / "decompiled"
logdir = base / "decompile_attempts"
outdir.mkdir(exist_ok=True)
logdir.mkdir(exist_ok=True)

pyc_files = sorted(variants.glob("*.pyc"))
if not pyc_files:
    print("No .pyc files found in", variants)
    sys.exit(1)

def run_cmd(cmd):
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

for pyc in pyc_files:
    print("===> Trying", pyc.name)
    base_name = pyc.stem
    success = False

    # 1) try decompyle3
    out_subdir = outdir / (base_name + "_decomp_decompyle3")
    out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = f"decompyle3 -o \"{out_subdir}\" \"{pyc}\""
    rc, out, err = run_cmd(cmd)
    (logdir / (base_name + "_decompyle3.stdout.txt")).write_text(out)
    (logdir / (base_name + "_decompyle3.stderr.txt")).write_text(err)
    if rc == 0:
        # check if anything produced in out_subdir
        produced = list(out_subdir.glob("**/*.py"))
        if produced:
            print("  decompyle3 succeeded for", pyc.name, "->", produced[0])
            success = True
        else:
            print("  decompyle3 returned 0 but produced no .py files (see logs).")

    if not success:
        # 2) try uncompyle6
        out_file = outdir / (base_name + "_uncompyle6.py")
        cmd2 = f"uncompyle6 -o \"{out_file}\" \"{pyc}\""
        rc2, out2, err2 = run_cmd(cmd2)
        (logdir / (base_name + "_uncompyle6.stdout.txt")).write_text(out2)
        (logdir / (base_name + "_uncompyle6.stderr.txt")).write_text(err2)
        if rc2 == 0:
            # uncompyle6 sometimes writes to STDOUT; check out_file
            if out_file.exists() and out_file.stat().st_size > 0:
                print("  uncompyle6 succeeded for", pyc.name, "->", out_file)
                success = True
            else:
                # try read stdout to see if it included a decompilation
                if out2.strip():
                    out_file.write_text(out2, encoding="utf-8", errors="replace")
                    print("  uncompyle6 produced output via stdout ->", out_file)
                    success = True
                else:
                    print("  uncompyle6 returned 0 but produced no output (see logs).")

    if not success:
        print("  Both decompilers failed for", pyc.name, "(see logs in decompile_attempts).")
    print()

print("Done. Check 'decompiled' and 'decompile_attempts' folders for results and logs.")
