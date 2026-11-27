# Code review with DeepSeek

A second pair of eyes on the whole bot, in themed groups of files, sent straight to DeepSeek's API. The
method is the Go backend's (its review of 2026-11-27: 92 findings, of which 26 were true and 3 were
proved by a test; 7 of its 11 "high" were false). Here it is adapted to this repository: the groups
are this code, the rules are this bot's facts, the key is not read from the bot's own `.env`, and the
probes run under the bot's `tests/conftest.py`.

Nothing in this folder is part of the bot: it is not in the Docker image (`.dockerignore`), and it never
places an order or reads a database.

## Files

- `ds_review.py`: the sender. One global order of blocks so DeepSeek's prompt cache can work: the rules,
  the list of files, the group's files (the ones more groups share first), the question last; the first
  request goes alone, the others follow four at a time.
- `groups.json`: the reviewer's instructions (`system`), the rules file, the coverage patterns and the
  13 groups. A group is `{"review": [...], "context": [...]}`: context files are shown so the reviewer
  can read a caller or a callee (the main source of false findings in the Go review) and are not
  reported on.
- `rules.md`: the facts and rules of this bot, sent first in every request. Keep it true: when a
  deliberate behaviour changes (`docs/decisions.md`), change it here too.
- `list_findings.py`: a compact list of every finding, and a table to fill in with verdicts.
- `mutate.py`: one deliberate breakage of a file, pytest, always restored.

## Before a run

1. The groups still cover the code (new modules must be added to a group):

   ```
   venv\Scripts\python -X utf8 tools\deepseek_review\ds_review.py tools\deepseek_review\groups.json --check-coverage
   ```

2. Commit first: the line numbers of the findings belong to the commit recorded in `_run.json`.
3. The size and the price of the input (sends nothing):

   ```
   venv\Scripts\python -X utf8 tools\deepseek_review\ds_review.py tools\deepseek_review\groups.json --dry-run
   ```

   On 2026-11-27: 13 groups, 26k to 47k tokens each, about 0.51 dollars of input at peak prices (about
   20 % expected from the cache). The output comes on top and is most of the cost: the thinking, 10k to
   30k tokens a group, about 0.5 to 1.5 dollars at peak prices. The Go review paid 1.00 dollar where the
   script said 2.48 (off-peak). **The owner agrees to the amount before a paid run.**
4. The balance before (and after: the difference is the real cost):

   ```
   venv\Scripts\python -X utf8 tools\deepseek_review\ds_review.py tools\deepseek_review\groups.json --balance --env-file C:\GO\.env
   ```

## The run

The key is `DEEPSEEK_API_KEY`, from the environment or from the file `--env-file` names (the Go
backend's `.env` holds it). Do not put it in this bot's `.env`: no setting reads it, and
`validate-config` would warn about it. The key is never printed. These calls go straight to DeepSeek,
not through the backend, so they do not count in the backend's `PAID_LLM_DAILY_USD`; they spend the same
balance. The code goes to DeepSeek (China); the repository is public anyway, and `ds_review.py` refuses
to send `.env`, databases, logs, keys, `var/`, `venv/` and the lock file whatever a groups file says.

```
venv\Scripts\python -X utf8 tools\deepseek_review\ds_review.py tools\deepseek_review\groups.json --env-file C:\GO\.env --out var\review\2026-11-27
```

About 10 to 15 minutes; run it in the background. `--only loop,orders` sends some groups only. The
answers go to `<out>\<group>.json` (`findings`, `usage`, `finish`; an error or an unreadable answer is
saved, not fatal) and `<out>\_run.json` (commit, model, order). Keep the results under `var\` (ignored by
git) or outside the repository, NEVER in git: the repository is public and the findings name weak spots.

## Checking the findings

Every finding is a claim until it is checked.

1. `venv\Scripts\python -X utf8 tools\deepseek_review\list_findings.py var\review\2026-11-27 --md var\review\2026-11-27\findings.md`
2. Read every finding in the real file, and also the caller and the callee DeepSeek did not see, the
   ORM model and the migration for anything about the database, and `docs/decisions.md` for anything
   that may be deliberate (then fix `rules.md` too, so the next run does not report it again).
3. An important one gets a TEMPORARY probe outside the repository that must FAIL on the current code,
   run with the bot's conftest (it keeps the real `.env` from switching the Oracle on and the tests from
   writing into the real log). From the repo root:

   ```
   venv\Scripts\python -m pytest <scratch>\test_probe.py -c pyproject.toml -p tests.conftest -p no:cacheprovider --rootdir . -o addopts="" -q
   ```

4. Classify each one in the table: confirmed by a probe / true by reading / possible / rejected, with
   the reason in one sentence.
5. Also review the code yourself, without DeepSeek: in the Go review the most serious bug was found
   that way.

The findings and a plan of fixes go to the owner first. Each approved fix gets a test in `tests/` that
fails on the old code, deliberate breakages through `mutate.py`, then ruff, `ruff format --check`, mypy,
the full pytest, a commit and a push to `main` (the CI steps on a fresh clone when dependencies, the
Dockerfile or migrations change), and a restart of the containers only when the owner says so.

```
venv\Scripts\python -X utf8 tools\deepseek_review\mutate.py src\trading_system\risk\engine.py "<old>" "<new>" tests\unit\test_risk.py
```

## Pitfalls

- The cache: parallel requests never hit it, and neither do files in a different order per request.
  Hence the one global order and the first request alone. Changing `rules.md` or `system` changes the
  start of every request, which is fine; just do not shuffle the groups between runs for no reason.
- With thinking on, 32 768 `max_tokens` was spent entirely on thinking for two ~58k groups of the Go
  review, with no answer (`finish: length`). Keep 65 536 and the groups under ~48k tokens.
- One answer came without `choices`: it is saved raw, the run goes on.
- DeepSeek's usual false findings: what it cannot see (the caller already checks it, the ORM model has
  the constraint, the conftest sets it up); "there is no test"; behaviour of a library it guesses
  (alpaca-py, pandas 3, SQLAlchemy 2.1); "races" in code that runs in one thread with no await; asking to
  log or wrap an error that would carry a secret. Its "high" ratings were mostly wrong in the Go review:
  trust only what you reproduce.
- Windows: write helper scripts with the Write tool, not a Git Bash heredoc (it collapses `\\`), and use
  `encoding="utf-8"` (the console is cp1251).
