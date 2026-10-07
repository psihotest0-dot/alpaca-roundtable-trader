# Paper Readiness Safety Audit

## Objective
This document outlines the safety and architectural checks completed before permitting local `--paper` mode.

## Checklist

| Area | Check | Status | Evidence / Notes |
|---|---|---|---|
| Live Orders | Are live market orders strictly blocked? | PASS | `AlpacaBroker` init checks `settings.is_paper` and passes `paper=True` to the TradingClient. Additionally, the `ExecutionEngine` refuses to send orders if `is_paper` is not True. |
| Duplicates | Are duplicate entries prevented? | PASS | `ExecutionEngine._generate_client_order_id` ensures deterministic IDs based on `signal_id`, enabling idempotency when using Alpaca's `client_order_id` field. |
| Stale Signals | Are stale signals rejected? | PASS | `RiskEngine.evaluate_signal` checks if `signal.expires_at < now` and returns False. |
| Risk Limits | Are pending entries counted in risk calculations? | PASS | `RiskEngine.evaluate_signal` identifies pending buy orders and adds their notional exposure to the `current_gross_exposure` calculation. |
| Submits/Timeouts | Is timeout during submission handled? | PASS | `ExecutionEngine.execute_intent` handles submission exceptions by catching them, delaying briefly, and calling `reconcile_order` rather than blindly retrying. |
| Partial Fills | Are partial fills protected? | PASS | Alpaca native OTOCO/bracket handling is used. The `cancel_bracket_order` method explicitly notes that canceling a parent limit order leaves the generated child legs for the partially filled quantity active. |
| Missing Protect | Are unprotected positions caught? | PASS | `ExecutionEngine.emergency_check_unprotected_positions` explicitly tallies position quantity against protective sell stops and submits emergency closing market orders for unprotected quantities. |
| Oracle Scope | Can Oracle bypass exits or risk sizing? | PASS | `src/trading_system/oracle/` contains no execution or sizing logic. The signal evaluation pipeline forces risk to authorize entry after the oracle acts as a veto filter. |
| Secrets | Are secrets exposed? | PASS | `pydantic.SecretStr` is used in configuration, and the CI uses a regex scanner to catch hardcoded secrets. |
| Defaults | Are defaults safe? | PASS | `is_paper` defaults to `True`. `oracle_enabled` defaults to `False`. Live commands default to `--dry-run`. |

## Conclusion
The application skeleton provides robust theoretical defenses against errant paper executions. Live mode is strictly prohibited. The system is conceptually ready for dry-run and local paper trading evaluation, but it is not certified as production-ready.
