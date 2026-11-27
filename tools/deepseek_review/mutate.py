"""Apply ONE deliberate breakage to a file of this repo, run pytest with a timeout, ALWAYS restore.

    venv\\Scripts\\python -X utf8 tools/deepseek_review/mutate.py <file> <old> <new> <pytest args...>

Run it from the repo root with the project's venv python (the one pytest uses). <old> must occur
exactly once; on a CRLF file the pattern is converted to CRLF. Prints the pytest exit code (1 = the
breakage was caught) and the failing tests. A new test that does not fail on at least one breakage
of the code it is about does not test it. Check `git diff --stat` afterwards: it must be empty for
the file.
"""

import os
import shutil
import subprocess
import sys

TIMEOUT_SECONDS = 120

path, old, new = sys.argv[1:4]
pytest_args = sys.argv[4:]
backup = path + ".mutbak"
if os.path.exists(backup):
    sys.exit(
        f"{backup} exists: an earlier run did not finish; restore the file from it by hand first"
    )
shutil.copyfile(path, backup)
try:
    with open(path, encoding="utf-8", newline="") as fh:
        src = fh.read()
    if "\r\n" in src:
        old = old.replace("\r\n", "\n").replace("\n", "\r\n")
        new = new.replace("\r\n", "\n").replace("\n", "\r\n")
    if src.count(old) != 1:
        print(f"MUTATION NOT APPLIED: {src.count(old)} matches of {old!r}")
        sys.exit(2)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(src.replace(old, new))
    try:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "-x",
                "--no-header",
                *pytest_args,
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
        print("pytest exit code:", r.returncode)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith(("FAILED", "ERROR")) or " passed" in line or " failed" in line:
                print(line[:200])
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {TIMEOUT_SECONDS}s")
finally:
    shutil.copyfile(backup, path)
    os.remove(backup)
    print("restored", path)
