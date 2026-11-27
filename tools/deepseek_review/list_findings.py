"""List the findings of a DeepSeek review run compactly, and write the table to fill with verdicts.

    python -X utf8 tools/deepseek_review/list_findings.py DIR [--md DIR/findings.md] [--min-severity low]

DIR is the --out of ds_review.py. Every finding is a claim until it is read in the real file (the
caller and the callee too, which DeepSeek did not see) or reproduced by a probe that fails on the
current code. The table has an empty verdict column for that: confirmed by a probe, true by reading,
possible, rejected (with the reason in one sentence).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RANK = {"high": 0, "medium": 1, "low": 2}


def load(out: Path) -> tuple[list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(out.glob("*.json")):
        if path.name.startswith("_"):
            continue
        res = json.loads(path.read_text(encoding="utf-8"))
        group = res.get("group", path.stem)
        if "error" in res:
            problems.append(f"{group}: ERROR {res['error']}")
            continue
        if not isinstance(res.get("findings"), list):
            problems.append(
                f"{group}: the answer could not be read ({res.get('finish')}); see content_if_unparsed"
            )
            continue
        if res.get("finish") != "stop":
            problems.append(f"{group}: finish {res.get('finish')} (the answer may be cut)")
        for f in res["findings"]:
            if isinstance(f, dict):
                findings.append({"group": group, **f})
    findings.sort(key=lambda f: (RANK.get(str(f.get("severity")), 3), f["group"], str(f.get("id"))))
    return findings, problems


def one_line(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--md")
    ap.add_argument("--min-severity", default="low", choices=list(RANK))
    a = ap.parse_args()
    out = Path(a.out)
    findings, problems = load(out)
    findings = [f for f in findings if RANK.get(str(f.get("severity")), 3) <= RANK[a.min_severity]]
    run = out / "_run.json"
    if run.is_file():
        info = json.loads(run.read_text(encoding="utf-8"))
        dirty = " + uncommitted changes" if info.get("uncommitted_changes") else ""
        print(
            f"code: {info.get('commit', '?')[:7]}{dirty}; model {info.get('model')}, thinking {info.get('thinking')}"
        )
    for p in problems:
        print(p)
    for f in findings:
        print(
            f"\n[{f.get('id')}] {f.get('kind')}/{f.get('severity')} (confidence {f.get('confidence')}) "
            f"{f.get('file')}:{f.get('lines')}\n  {one_line(f.get('title'), 200)}\n"
            f"  quote: {one_line(f.get('quote'), 200)}\n  why: {one_line(f.get('explanation'), 400)}"
        )
    print(f"\n{len(findings)} findings")
    if a.md:
        rows = [
            "| id | DeepSeek | file:lines | claim | verdict | reason |",
            "|---|---|---|---|---|---|",
        ]
        for f in findings:
            claim = one_line(f.get("title"), 160).replace("|", "\\|")
            rows.append(
                f"| {f.get('id')} | {f.get('kind')}/{f.get('severity')} | {f.get('file')}:{f.get('lines')} "
                f"| {claim} |  |  |"
            )
        Path(a.md).write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"table written: {a.md}")


if __name__ == "__main__":
    main()
