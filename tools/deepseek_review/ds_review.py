"""Review the trading bot's code with DeepSeek in themed groups, laid out for its prompt cache.

Adapted on 2026-11-27 from the Go backend's tool (go-monolith-backups/tools/ds_review.py, 2026-11-26).

The cache of DeepSeek keeps the start of a request that is the same byte for byte as an earlier one
(prompt_cache_hit_tokens, about 30x cheaper than a miss), and only once that request is answered. So
every request has the same blocks in one global order: the rules, the list of the files, then the
group's files with the files that more groups share first, and the question last. Two requests are the
same from the start up to their first different file. The first request goes alone; the next ones
follow it, the one that shares the most with what was sent first.

    python -X utf8 tools/deepseek_review/ds_review.py tools/deepseek_review/groups.json --out DIR
        [--root REPO] [--env-file PATH] [--model deepseek-v4-pro] [--thinking on|off]
        [--max-tokens 65536] [--parallel 4] [--only g1,g2] [--dry-run]
    python -X utf8 tools/deepseek_review/ds_review.py tools/deepseek_review/groups.json --check-coverage
    python -X utf8 tools/deepseek_review/ds_review.py tools/deepseek_review/groups.json --balance [--env-file PATH]

groups.json: {"rules": "<repo-relative file>", "system": "<the reviewer's instructions>",
"coverage": [patterns of git-tracked files that must be in a group], "coverage_skip": [exceptions],
"groups": {"name": [files] or {"review": [files], "context": [files]}}}. Files in "context" are shown
so the reviewer can read a caller or a callee, and the question says not to report on them.

The key: DEEPSEEK_API_KEY from the environment, else from the file given with --env-file (the Go
backend's .env holds it; the bot's own .env does not, and `validate-config` would warn about a name in
it that no setting reads). It is never printed. --dry-run sends nothing: it prints the sizes, the
share from the cache this layout expects and the cost of the input with and without it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

URL = "https://api.deepseek.com/chat/completions"
BALANCE_URL = "https://api.deepseek.com/user/balance"
# Per 1M tokens, peak (the prices of the Go backend's internal/llm/spend.go): hit, miss, output.
# Off-peak is cheaper: the real cost is the difference of the balance before and after (--balance).
PRICES = {"deepseek-v4-pro": (0.044, 1.32, 3.96), "deepseek-v4-flash": (0.006, 0.30, 1.20)}
CHARS_PER_TOKEN = 3.0  # measured on the Flet review: 175 482 characters -> 58 057 tokens
REPO = Path(__file__).resolve().parents[2]
# Never sent, whatever a groups file says: secrets, runtime state, logs, the lock file.
FORBIDDEN = re.compile(
    r"(^|/)(\.env(\..+)?|[^/]+\.(db|sqlite3?|db-wal|db-shm|log|pem|key|p12|pfx)|poetry\.lock)$"
    r"|^(var|venv|\.venv|secrets)/"
)
ALLOWED_ENV = ".env.example"


def read_key(env_file: str | None) -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    if env_file:
        for line in Path(env_file).read_text(encoding="utf-8").splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == "DEEPSEEK_API_KEY":
                return value.strip().strip('"').strip("'")
    sys.exit("DEEPSEEK_API_KEY not found (set it in the environment or pass --env-file)")


def check_sendable(rel: str) -> None:
    if rel != ALLOWED_ENV and FORBIDDEN.search(rel.replace("\\", "/")):
        sys.exit(f"refused: {rel} must never be sent (secrets or runtime state)")


def listing(root: Path, rel: str) -> str:
    text = (root / rel).read_text(encoding="utf-8")
    numbered = "\n".join(f"{i:5d}| {line}" for i, line in enumerate(text.splitlines(), 1))
    return f"===== FILE: {rel} =====\n{numbered}\n===== END {rel} ====="


def tokens(s: str) -> int:
    return int(len(s) / CHARS_PER_TOKEN)


def split_group(spec: list[str] | dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """(files to review, files shown for context only) of one group."""
    if isinstance(spec, list):
        return list(spec), []
    review = list(spec.get("review", []))
    return review, [f for f in spec.get("context", []) if f not in review]


def layout(cfg: dict[str, Any], root: Path) -> dict[str, list[str]]:
    """The blocks of each group's request, in the one global order (see the top of the file)."""
    groups = {name: split_group(spec) for name, spec in cfg["groups"].items()}
    count: dict[str, int] = {}
    for review, context in groups.values():
        for f in set(review) | set(context):
            check_sendable(f)
            if not (root / f).is_file():
                sys.exit(f"missing: {f} (named in a group, not in {root})")
            count[f] = count.get(f, 0) + 1
    order = sorted(count, key=lambda f: (-count[f], f))
    rules = (root / cfg["rules"]).read_text(encoding="utf-8")
    # Only the name of the rules file is sent, not where it is on this machine.
    head = [
        f"Project rules ({Path(cfg['rules']).name}):\n{rules}",
        "All files of this review: " + ", ".join(sorted(count)),
    ]
    blocks = {}
    for name, (review, context) in groups.items():
        shown = set(review) | set(context)
        mine = [f for f in order if f in shown]
        to_review = [f for f in mine if f in set(review)]
        question = (
            f"Review group '{name}': {', '.join(to_review)}. Report findings only in these files."
        )
        if context:
            ctx = [f for f in mine if f in set(context)]
            question += f" Shown for context only, do not report on them: {', '.join(ctx)}."
        blocks[name] = head + [listing(root, f) for f in mine] + [question]
    return blocks


def common_prefix(a: list[str], b: list[str]) -> int:
    """Tokens of the blocks two requests share from their start."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += tokens(x)
    return n


def send_order(names: list[str], blocks: dict[str, list[str]]) -> list[str]:
    """The first: the group that shares the most with the others; then each time the group that
    shares the most with one already sent."""
    if len(names) < 2:
        return list(names)
    sent = [
        max(names, key=lambda g: sum(common_prefix(blocks[g], blocks[h]) for h in names if h != g))
    ]
    while len(sent) < len(names):
        rest = [g for g in names if g not in sent]
        sent.append(max(rest, key=lambda g: max(common_prefix(blocks[g], blocks[h]) for h in sent)))
    return sent


def cost(model: str, hit: int, miss: int, out: int) -> float:
    h, m, o = PRICES.get(model, PRICES["deepseek-v4-pro"])
    return (hit * h + miss * m + out * o) / 1e6


def git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8"
    )
    return done.stdout.strip()


def check_coverage(cfg: dict[str, Any], root: Path) -> int:
    """Every git-tracked file the coverage patterns name is reviewed in some group (an empty file
    needs none); every file a group names exists. Returns the number of problems."""
    reviewed: set[str] = set()
    named: set[str] = set()
    for spec in cfg["groups"].values():
        review, context = split_group(spec)
        reviewed |= set(review)
        named |= set(review) | set(context)
    tracked = git(root, "ls-files").splitlines()
    wanted = [
        f
        for f in tracked
        if any(fnmatch.fnmatch(f, p) for p in cfg.get("coverage", []))
        and not any(fnmatch.fnmatch(f, p) for p in cfg.get("coverage_skip", []))
        and (root / f).is_file()
        and (root / f).stat().st_size > 0
    ]
    problems = 0
    for f in sorted(set(wanted) - reviewed):
        print(f"not in any group: {f}")
        problems += 1
    for f in sorted(named):
        if not (root / f).is_file():
            print(f"named in a group but missing: {f}")
            problems += 1
        elif f not in tracked:
            print(f"named in a group but not tracked by git: {f}")
            problems += 1
    print(
        f"{len(wanted)} files to cover, {len(reviewed)} reviewed in the groups, {problems} problems"
    )
    return problems


def balance(key: str) -> None:
    """The totals only (GET /user/balance), so the real cost of a run is the difference."""
    req = urllib.request.Request(BALANCE_URL, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for info in data.get("balance_infos", []):
        print(f"balance: {info.get('total_balance')} {info.get('currency')}")
    print(f"available: {data.get('is_available')}")


def ask(
    key: str, model: str, system: str, blocks: list[str], thinking: bool, max_tokens: int
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(blocks)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    started = time.monotonic()
    data: dict[str, Any] = {}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", "replace")[:300]
            if ex.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(20 * (attempt + 1))
                continue
            return {"error": f"HTTP {ex.code}: {detail}"}
        except (urllib.error.URLError, TimeoutError) as ex:
            if attempt < 2:
                time.sleep(20)
                continue
            return {"error": repr(ex)}
    if not data.get("choices"):
        # One answer of the Go review of 2026-11-26 came without "choices": keep it, do not fail.
        return {"error": "no choices in the answer", "raw": data}
    choice = data["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    try:
        findings = json.loads(content).get("findings")
    except (json.JSONDecodeError, AttributeError):
        findings = None
    return {
        "seconds": round(time.monotonic() - started),
        "usage": data.get("usage", {}),
        "finish": choice.get("finish_reason"),
        "findings": findings,
        "content_if_unparsed": content if findings is None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("groups")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--out")
    ap.add_argument("--env-file")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--thinking", default="on", choices=["on", "off"])
    # 32 768 went all on thinking for a ~58k group of the Go review, with no answer.
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-coverage", action="store_true")
    ap.add_argument("--balance", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.groups).read_text(encoding="utf-8"))
    root = Path(a.root).resolve()
    if a.check_coverage:
        sys.exit(1 if check_coverage(cfg, root) else 0)
    if a.balance:
        balance(read_key(a.env_file))
        return
    if not a.out and not a.dry_run:
        sys.exit("--out DIR is required for a real run")
    blocks = layout(cfg, root)
    names = [g for g in cfg["groups"] if not a.only or g in a.only.split(",")]
    unknown = sorted(set(a.only.split(",")) - set(cfg["groups"])) if a.only else []
    if unknown:
        sys.exit(f"unknown groups: {', '.join(unknown)}")
    system = cfg["system"]
    order = send_order(names, blocks)
    sys_t = tokens(system)
    hit_total = miss_total = 0
    for i, g in enumerate(order):
        size = sys_t + sum(tokens(b) for b in blocks[g])
        cached = sys_t + max(common_prefix(blocks[g], blocks[h]) for h in order[:i]) if i else 0
        hit_total, miss_total = hit_total + cached, miss_total + size - cached
        print(f"  {i + 1}. {g}: {size} tokens, ~{cached} expected from the cache")
    share = 100 * hit_total // max(1, hit_total + miss_total)
    print(
        f"input with this layout: ~${cost(a.model, hit_total, miss_total, 0):.3f} (~{share} % from the cache); "
        f"with nothing from the cache: ~${cost(a.model, 0, hit_total + miss_total, 0):.3f}; "
        "the output (mostly thinking) comes on top: about 10-30k tokens a group"
    )
    if a.dry_run:
        return
    key = read_key(a.env_file)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # The line numbers of the findings belong to this version of the code.
    run_info = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git(root, "rev-parse", "HEAD"),
        "uncommitted_changes": bool(git(root, "status", "--porcelain", "--untracked-files=no")),
        "model": a.model,
        "thinking": a.thinking,
        "max_tokens": a.max_tokens,
        "order": order,
    }
    (out / "_run.json").write_text(json.dumps(run_info, indent=1), "utf-8")
    total = [0, 0, 0]
    lock = threading.Lock()

    def run(g: str) -> None:
        res = ask(key, a.model, system, blocks[g], a.thinking == "on", a.max_tokens)
        res["group"] = g
        (out / f"{g}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        if "error" in res:
            print(f"{g}: ERROR {res['error']}", flush=True)
            return
        u = res["usage"]
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", 0)
        o = u.get("completion_tokens", 0)
        with lock:
            total[0] += hit
            total[1] += miss
            total[2] += o
        n = len(res["findings"]) if isinstance(res["findings"], list) else "UNPARSED"
        print(
            f"{g}: {n} findings, {res['seconds']}s, cache {hit} + miss {miss}, out {o}, {res['finish']}, "
            f"~${cost(a.model, hit, miss, o):.3f}",
            flush=True,
        )

    # The first alone: when it is answered, its start is in the cache for the others.
    run(order[0])
    with concurrent.futures.ThreadPoolExecutor(max(1, a.parallel)) as pool:
        list(pool.map(run, order[1:]))
    hit, miss, o = total
    print(
        f"total: cache {hit} + miss {miss} ({100 * hit // max(1, hit + miss)} % of the input from the cache), "
        f"out {o}, ~${cost(a.model, hit, miss, o):.3f} at peak prices"
    )


if __name__ == "__main__":
    main()
