"""The files in ops/ (alert rules and a Grafana dashboard for the Go backend's Prometheus) and the
ports in compose.yaml, held against the code.

The rules and the dashboard are copied to the backend's repository by hand, so they can drift from the
metrics in two silent ways: a metric is renamed (the panel goes blank and the alert never fires), or a
rule names something that nobody sets. This test reads every `expr` and checks the metric names and
labels in it against what the process really publishes. The rules' behaviour is tested by promtool
(ops/prometheus/alpaca_alerts_test.yml, command in its header); here it is only checked that each rule
has a case that fires and a case that does not.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from prometheus_client import REGISTRY

from trading_system.observability import metrics  # noqa: F401 - importing it registers the metrics

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "ops" / "prometheus" / "alpaca_alerts.yml"
RULE_TESTS = ROOT / "ops" / "prometheus" / "alpaca_alerts_test.yml"
DASHBOARD = ROOT / "ops" / "grafana" / "alpaca-trader.json"
COMPOSE = ROOT / "compose.yaml"

NAME = re.compile(r"\balpaca_[a-z0-9_]+")
SELECTOR = re.compile(r"(alpaca_[a-z0-9_]+)\s*\{([^}]*)\}")
MATCHER_LABEL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=~|!~|!=|=)")
GROUPING = re.compile(r"\b(?:by|without)\s*\(([^)]*)\)")
LEGEND_LABEL = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
# Labels that Prometheus puts on every scraped series: instance and job by itself, service by the
# `labels:` of the two targets in the backend's prometheus.yml (trader / observer).
SCRAPE_LABELS = {"instance", "job", "service"}


def _published() -> dict[str, tuple[str, ...]]:
    """Every series name the process exposes (counters with `_total`), with the labels it carries."""
    out: dict[str, tuple[str, ...]] = {}
    for name, collector in REGISTRY._names_to_collectors.items():
        if name.startswith("alpaca_"):
            out[name] = tuple(getattr(collector, "_labelnames", ()))
    return out


def _expressions() -> list[tuple[str, str, str]]:
    """(where, expr, legend) for every rule and every dashboard target."""
    found: list[tuple[str, str, str]] = []
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    for group in rules["groups"]:
        for rule in group["rules"]:
            found.append((f"rule {rule['alert']}", rule["expr"], ""))
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            found.append(
                (f"panel '{panel['title']}'", target["expr"], target.get("legendFormat", ""))
            )
    return found


def test_the_published_names_are_what_the_artifacts_assume() -> None:
    published = _published()
    # Counters are exposed with _total; the not-wired registry is not exposed at all.
    assert "alpaca_heartbeat_timestamp_seconds" in published
    assert "alpaca_entries_errors_total" in published
    assert "alpaca_oracle_answers_total" in published
    assert "alpaca_runtime_health" not in published  # defined, set by nothing: not published


@pytest.mark.parametrize("where, expr, legend", _expressions())
def test_every_metric_and_label_in_an_expression_is_published(
    where: str, expr: str, legend: str
) -> None:
    published = _published()
    used = set(NAME.findall(expr))
    unknown = used - set(published)
    assert not unknown, f"{where} uses {sorted(unknown)}, which the bot does not publish"

    allowed = SCRAPE_LABELS.union(*(set(published[name]) for name in used))
    labels: set[str] = set()
    for _metric, selector in SELECTOR.findall(expr):
        labels.update(MATCHER_LABEL.findall(selector))
    for grouping in GROUPING.findall(expr):
        labels.update(part.strip() for part in grouping.split(",") if part.strip())
    labels.update(LEGEND_LABEL.findall(legend))
    stray = labels - allowed
    assert not stray, (
        f"{where} uses the label(s) {sorted(stray)}; {sorted(used)} carry {sorted(allowed)}"
    )


# Series that only the trading loop ever sets. The observer exports them too, at 0 for good (an
# unlabelled gauge is exported from the first scrape, before anything sets it), so an expression that
# reads one without the filter service="trader" alerts on, or draws, a 0 that means "never set":
# a timestamp of 0 looks infinitely stale, an equity of 0 looks like a wiped account.
TRADER_ONLY = (
    "alpaca_account_equity",
    "alpaca_active_positions",
    "alpaca_gross_exposure",
    "alpaca_unprotected_positions",
    "alpaca_protection_check_timestamp_seconds",
    "alpaca_reconcile_timestamp_seconds",
)


@pytest.mark.parametrize("where, expr, legend", _expressions())
def test_a_series_only_the_trader_sets_is_read_with_the_trader_filter(
    where: str, expr: str, legend: str
) -> None:
    for name in TRADER_ONLY:
        for use in re.finditer(r"\b" + name + r"\b(\{[^}]*\})?", expr):
            assert 'service="trader"' in (use.group(1) or ""), (
                f'{where} reads {name} without service="trader": the observer exports it as 0'
            )


def test_every_dashboard_expression_names_a_metric_of_the_bot_or_up() -> None:
    """A panel that reads nothing of ours (a typo that happens to be a valid query) is a blank panel."""
    for where, expr, _legend in _expressions():
        assert NAME.search(expr) or expr.startswith("up{"), f"{where} reads no series of the bot"


def test_the_dashboard_is_a_well_formed_provisioned_dashboard() -> None:
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "alpaca-trader" and dashboard["title"]
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids)), "panel ids must be unique"
    for panel in dashboard["panels"]:
        grid = panel["gridPos"]
        assert 0 <= grid["x"] and grid["x"] + grid["w"] <= 24, panel["title"]
        if panel["type"] != "row":
            assert panel["targets"], f"panel '{panel['title']}' has no query"
            assert panel.get("description"), (
                f"panel '{panel['title']}' says nothing of what it shows"
            )
    # Overlap check: two panels must not claim the same cells.
    cells: set[tuple[int, int]] = set()
    for panel in dashboard["panels"]:
        grid = panel["gridPos"]
        for x in range(grid["x"], grid["x"] + grid["w"]):
            for y in range(grid["y"], grid["y"] + grid["h"]):
                assert (x, y) not in cells, f"panel '{panel['title']}' overlaps another"
                cells.add((x, y))


def test_every_rule_has_a_promtool_case_that_fires_and_one_that_does_not() -> None:
    """The backend does not accept a rule without a test. The behaviour itself is promtool's to check."""
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    names = {rule["alert"] for group in rules["groups"] for rule in group["rules"]}
    fires: set[str] = set()
    quiet: set[str] = set()
    for case in yaml.safe_load(RULE_TESTS.read_text(encoding="utf-8"))["tests"]:
        for check in case["alert_rule_test"]:
            (fires if check["exp_alerts"] else quiet).add(check["alertname"])
    assert fires == names, f"rules without a firing case: {sorted(names - fires)}"
    assert quiet == names, f"rules without a case that stays quiet: {sorted(names - quiet)}"


def test_every_rule_has_a_severity_and_text_for_a_human() -> None:
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    for group in rules["groups"]:
        for rule in group["rules"]:
            assert rule["labels"]["severity"] in {"info", "warning", "critical"}, rule["alert"]
            assert rule["annotations"]["summary"] and rule["annotations"]["description"], rule[
                "alert"
            ]


# --- compose.yaml ---------------------------------------------------------------------------


def _compose() -> dict[str, Any]:
    services: dict[str, Any] = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    return services


def test_only_the_two_loops_publish_a_port_and_only_on_the_hosts_loopback() -> None:
    """/metrics has no password: it must be reachable from this PC only. 127.0.0.1 in the mapping is
    what keeps Docker from publishing it on every interface."""
    published = {
        name: service["ports"] for name, service in _compose().items() if service.get("ports")
    }
    assert published == {
        "trader": ["127.0.0.1:9101:9100"],
        "observe": ["127.0.0.1:9102:9100"],
    }


def test_the_loops_serve_metrics_on_all_interfaces_inside_their_container() -> None:
    """The host cannot reach a container's 127.0.0.1, so inside it the server must listen on 0.0.0.0;
    the mapping above is what limits it to the host's loopback."""
    for name in ("trader", "observe"):
        env = _compose()[name]["environment"]
        assert str(env["METRICS_ENABLED"]).lower() == "true", name
        assert env["METRICS_HOST"] == "0.0.0.0", name
        assert str(env["METRICS_PORT"]) == "9100", name
    for name in ("data", "news", "forward", "backup"):
        env = _compose()[name].get("environment", {})
        assert "METRICS_ENABLED" not in env, f"{name} has no loop to measure"
