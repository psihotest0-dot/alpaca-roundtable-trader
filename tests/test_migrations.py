"""The migrations must build exactly what the models describe, and must not lose data."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, text

from trading_system.domain.enums import OrderSide, OrderStatus, OrderType
from trading_system.domain.models import BrokerOrder
from trading_system.storage.database import get_engine, get_session_factory
from trading_system.storage.models import Base
from trading_system.storage.repositories import OrderRepository

INITIAL = "7cbd8a703bf7"
ORDER_IDENTITY = "c3f1a9e47d52"
TAKE_PROFIT = "e8b2d5a61f07"

_ORDER_COLUMNS = (
    "(broker_id, client_order_id, symbol, side, type, status, quantity, filled_quantity, "
    "created_at, updated_at)"
)


@pytest.fixture
def db(tmp_path: Path) -> tuple[Any, Config]:
    url = f"sqlite:///{(tmp_path / 'migrations.db').as_posix()}"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", url)
    return create_engine(url), cfg


def _insert_old_order(conn: Any, broker_id: str, client_order_id: str, status: str) -> None:
    conn.execute(
        text(
            f"INSERT INTO broker_orders {_ORDER_COLUMNS} VALUES "
            "(:b, :c, 'AAPL', 'BUY', 'LIMIT', :s, 10, 0, '2026-11-01 10:00:00', '2026-11-01 10:00:00')"
        ),
        {"b": broker_id, "c": client_order_id, "s": status},
    )


def test_migrations_build_exactly_what_the_models_describe(db: tuple[Any, Config]) -> None:
    """Catches schema drift. Before the order-identity revision this reported one
    difference: orderstatus used other labels than the application stores."""
    engine, cfg = db
    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_upgrade_keeps_orders_and_fills_and_clears_placeholder_broker_ids(
    db: tuple[Any, Config],
) -> None:
    engine, cfg = db
    command.upgrade(cfg, INITIAL)
    with engine.begin() as conn:
        # Old convention: an in-flight placeholder carried its client id as broker_id.
        _insert_old_order(conn, "exec_pending", "exec_pending", "SUBMITTING")
        _insert_old_order(conn, "alpaca-uuid-1", "exec_done", "FILLED")
        conn.execute(
            text(
                "INSERT INTO fills (broker_order_id, broker_execution_id, price, quantity, timestamp) "
                "VALUES ('alpaca-uuid-1', 'exec-1', 150, 10, '2026-11-01 10:01:00')"
            )
        )

    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        rows = dict(
            conn.execute(text("SELECT client_order_id, broker_id FROM broker_orders")).all()
        )
        assert rows == {"exec_pending": None, "exec_done": "alpaca-uuid-1"}
        assert conn.execute(text("SELECT broker_order_id FROM fills")).scalars().all() == [
            "alpaca-uuid-1"
        ]


def test_a_placeholder_can_be_confirmed_in_place_after_upgrade(db: tuple[Any, Config]) -> None:
    """The reason for the migration: no UNIQUE clash between placeholder and confirmation."""
    engine, cfg = db
    command.upgrade(cfg, "head")

    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO broker_orders {_ORDER_COLUMNS} VALUES "
                "(NULL, 'exec_abc', 'AAPL', 'BUY', 'LIMIT', 'SUBMITTING', 10, 0, "
                "'2026-11-01 10:00:00', '2026-11-01 10:00:00')"
            )
        )
        conn.execute(
            text(
                "UPDATE broker_orders SET broker_id = 'alpaca-uuid-1', status = 'SUBMITTED' "
                "WHERE client_order_id = 'exec_abc'"
            )
        )
        # The longest status label must fit too (VARCHAR(16) in the old schema).
        conn.execute(
            text(
                f"INSERT INTO broker_orders {_ORDER_COLUMNS} VALUES "
                "(NULL, 'exec_x', 'AAPL', 'BUY', 'LIMIT', 'UNKNOWN_REQUIRES_RECONCILIATION', "
                "10, 0, '2026-11-01 10:00:00', '2026-11-01 10:00:00')"
            )
        )

    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM broker_orders")).scalar() == 2


def test_downgrade_restores_the_old_shape_and_fills_in_missing_broker_ids(
    db: tuple[Any, Config],
) -> None:
    engine, cfg = db
    command.upgrade(cfg, "head")
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO broker_orders {_ORDER_COLUMNS} VALUES "
                "(NULL, 'exec_pending', 'AAPL', 'BUY', 'LIMIT', 'SUBMITTING', 10, 0, "
                "'2026-11-01 10:00:00', '2026-11-01 10:00:00')"
            )
        )

    command.downgrade(cfg, INITIAL)

    with engine.connect() as conn:
        row = conn.execute(text("SELECT broker_id, client_order_id FROM broker_orders")).one()
        assert tuple(row) == ("exec_pending", "exec_pending")
        # ... and the schema is the old one again: broker_id is NOT NULL.
        info = {r[1]: r[3] for r in conn.execute(text("PRAGMA table_info(broker_orders)"))}
        assert info["broker_id"] == 1


def _insert_intent(conn: Any, signal_id: str) -> None:
    conn.execute(
        text(
            "INSERT INTO strategy_signals (signal_id, version, symbol, timeframe, "
            "closed_bar_timestamp, direction, confidence, created_at) VALUES "
            "(:s, '1', 'AAPL', '15m', '2026-11-01 10:00:00', 'LONG', 1.0, '2026-11-01 10:00:00')"
        ),
        {"s": signal_id},
    )
    conn.execute(
        text(
            "INSERT INTO order_intents (signal_id, symbol, side, type, quantity, limit_price, "
            "stop_price, intent_timestamp) VALUES "
            "(:s, 'AAPL', 'BUY', 'LIMIT', 10, 150, 144, '2026-11-01 10:00:00')"
        ),
        {"s": signal_id},
    )


def test_take_profit_column_is_added_to_existing_intents_as_null_and_dropped_again(
    db: tuple[Any, Config],
) -> None:
    """Intents saved before the C3 contract have no target; they must survive both ways."""
    engine, cfg = db
    command.upgrade(cfg, ORDER_IDENTITY)
    with engine.begin() as conn:
        _insert_intent(conn, "sig-before")

    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT symbol, limit_price, stop_price, take_profit_price FROM order_intents")
        ).one()
        assert tuple(row) == ("AAPL", 150, 144, None)

    command.downgrade(cfg, ORDER_IDENTITY)
    with engine.connect() as conn:
        columns = [r[1] for r in conn.execute(text("PRAGMA table_info(order_intents)"))]
        assert "take_profit_price" not in columns
        assert conn.execute(text("SELECT limit_price FROM order_intents")).scalar() == 150


def test_migration_chain_goes_all_the_way_down(db: tuple[Any, Config]) -> None:
    """What CI's migration smoke test does: upgrade head, then downgrade base."""
    _, cfg = db
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")


def test_repositories_work_on_a_migrated_database(db: tuple[Any, Config]) -> None:
    """The application opens SQLite with foreign keys ON. A fill must still find its
    order through the (now nullable, unique) broker_id after the table rebuild."""
    engine, cfg = db
    command.upgrade(cfg, "head")
    repo = OrderRepository(get_session_factory(get_engine(str(engine.url))))
    now = datetime.now(timezone.utc)
    order = {
        "client_order_id": "exec_abc",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "type": OrderType.LIMIT,
        "quantity": Decimal("10"),
        "filled_quantity": Decimal("0"),
        "created_at": now,
        "updated_at": now,
    }

    repo.save_broker_order(BrokerOrder(status=OrderStatus.SUBMITTING, **order))  # no broker id yet
    repo.save_broker_order(BrokerOrder(broker_id="alpaca-1", status=OrderStatus.SUBMITTED, **order))

    assert repo.save_fill("alpaca-1", "exec-1", 150.0, 10.0, now) is True
    assert repo.save_fill("no-such-order", "exec-2", 150.0, 10.0, now) is False  # FK enforced
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM broker_orders")).scalar() == 1


def _insert_old_oracle_rows(conn: Any) -> None:
    conn.execute(
        text(
            "INSERT INTO strategy_signals (signal_id, version, symbol, timeframe, "
            "closed_bar_timestamp, direction, confidence, created_at) VALUES "
            "('sig', '1', 'AAPL', '15m', '2026-11-01 10:00:00', 'LONG', 1.0, '2026-11-01 10:00:00')"
        )
    )
    for approved, reason in ((1, "fine"), (0, "no")):
        conn.execute(
            text(
                "INSERT INTO oracle_decisions (signal_id, approved, reasoning, decision_timestamp) "
                "VALUES ('sig', :a, :r, '2026-11-01 10:01:00')"
            ),
            {"a": approved, "r": reason},
        )


def test_oracle_decisions_become_verdicts_and_back(db: tuple[Any, Config]) -> None:
    """approved becomes APPROVE, not approved becomes VETO; the downgrade goes the other way."""
    engine, cfg = db
    command.upgrade(cfg, TAKE_PROFIT)
    with engine.begin() as conn:
        _insert_old_oracle_rows(conn)

    command.upgrade(cfg, "head")
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT verdict, reasoning, provider, failed, confidence FROM oracle_decisions ORDER BY id"
            )
        ).all()
        assert [tuple(r) for r in rows] == [
            ("APPROVE", "fine", "", 0, None),
            ("VETO", "no", "", 0, None),
        ]
        indexes = [r[1] for r in conn.execute(text("PRAGMA index_list(oracle_decisions)"))]
        assert "ix_oracle_decisions_signal_id" in indexes

    command.downgrade(cfg, TAKE_PROFIT)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT approved, reasoning FROM oracle_decisions ORDER BY id")
        ).all()
        assert [tuple(r) for r in rows] == [(1, "fine"), (0, "no")]
        columns = [r[1] for r in conn.execute(text("PRAGMA table_info(oracle_decisions)"))]
        assert "verdict" not in columns and "failed" not in columns
