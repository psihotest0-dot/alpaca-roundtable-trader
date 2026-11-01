# Security policy

This project places orders on a brokerage account. Today it is **paper trading
only** (`IS_PAPER=true` is enforced), but it is built to carry real money later,
so treat credentials and the kill switch accordingly.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting (the repository's **Security** tab,
**Report a vulnerability**). Please do not open a public issue for anything that
could be exploited, and never paste API keys or logs containing them.

## Keeping secrets out of git

- Real credentials live only in `.env`, which is git-ignored. Only
  `.env.example` (placeholders) is tracked.
- Install the local guard rails once per machine: `pip install pre-commit`, then
  `pre-commit install`. They block env files, databases, logs, keys and Alpaca
  or AnythingLLM-style key patterns. CI runs the same checks on every push and
  pull request, so `--no-verify` does not bypass them.
- Runtime state (database, logs, vector store) lives in `var/`, which is ignored.

## If a secret is ever committed

1. **Revoke it first.** Regenerate the Alpaca key pair in the Alpaca dashboard
   (and rotate any other exposed key). Assume it is compromised, even if the
   repository was private or the commit was deleted: this repository is public
   and may have been cloned or cached.
2. Put the new values in `.env` only.
3. Only then remove it from the files and, if you decide it is worth it, from
   history (`git filter-repo`), coordinating with everyone who has a clone.
4. Check the Alpaca account for unexpected orders.
