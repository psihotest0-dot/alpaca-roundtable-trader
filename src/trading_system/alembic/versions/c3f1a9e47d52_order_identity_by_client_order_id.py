"""Order identity by client_order_id

Revision ID: c3f1a9e47d52
Revises: 7cbd8a703bf7
Create Date: 2026-11-01 19:30:00.000000

broker_orders was keyed by the broker's id, but we only learn that id after the
broker answers. The in-flight placeholder therefore used the client_order_id as a
fake broker_id, and the confirmed row (real id, same client_order_id) collided with
it on UNIQUE(client_order_id).

Now client_order_id, which we generate before anything is sent, is the primary key,
and broker_id is a nullable unique column that is filled in on confirmation.

The same revision aligns orderstatus with the domain enum. The initial migration
used the Alpaca labels (PENDING_NEW, ACCEPTED, ...), the application stores
CREATED, RISK_APPROVED, SUBMITTING, ... (SQLite ignores the length, other
databases would not).

SQLite cannot change a primary key in place, so this follows the documented
12-step rebuild: create the new table, copy the rows, drop the old one, rename.
Migrations run with foreign keys off, which that procedure requires; fills keeps
pointing at broker_orders(broker_id), which is still unique.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3f1a9e47d52"
down_revision: Union[str, Sequence[str], None] = "7cbd8a703bf7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SIDES = ("BUY", "SELL")
_TYPES = ("MARKET", "LIMIT", "STOP", "STOP_LIMIT", "TRAILING_STOP")
_STATUS_NEW = (
    "CREATED",
    "RISK_APPROVED",
    "SUBMITTING",
    "SUBMITTED",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELED",
    "REJECTED",
    "EXPIRED",
    "UNKNOWN_REQUIRES_RECONCILIATION",
)
_STATUS_OLD = (
    "PENDING_NEW",
    "ACCEPTED",
    "NEW",
    "PARTIALLY_FILLED",
    "FILLED",
    "CANCELED",
    "REJECTED",
    "EXPIRED",
    "REPLACED",
    "PENDING_CANCEL",
    "PENDING_REPLACE",
)


def _require_sqlite() -> None:
    if op.get_bind().dialect.name != "sqlite":
        raise NotImplementedError(
            "This migration rebuilds broker_orders and is written for SQLite, the only "
            "database the project supports (docs/decisions.md)."
        )


def _create_broker_orders(name: str, *, status_labels: Sequence[str], new_key: bool) -> None:
    op.create_table(
        name,
        sa.Column("client_order_id", sa.String(), nullable=False),
        # New schema: unknown until the broker answers. Old schema: always set.
        sa.Column("broker_id", sa.String(), nullable=new_key),
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("side", sa.Enum(*_SIDES, name="orderside"), nullable=False),
        sa.Column("type", sa.Enum(*_TYPES, name="ordertype"), nullable=False),
        sa.Column("status", sa.Enum(*status_labels, name="orderstatus"), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("filled_quantity", sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column("limit_price", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("stop_price", sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        *(
            (sa.PrimaryKeyConstraint("client_order_id"), sa.UniqueConstraint("broker_id"))
            if new_key
            else (sa.PrimaryKeyConstraint("broker_id"), sa.UniqueConstraint("client_order_id"))
        ),
    )


_COPIED = (
    "symbol, side, type, status, quantity, filled_quantity, "
    "limit_price, stop_price, created_at, updated_at"
)


def upgrade() -> None:
    """Upgrade schema."""
    _require_sqlite()
    _create_broker_orders("broker_orders_new", status_labels=_STATUS_NEW, new_key=True)
    # Placeholders were stored with broker_id = client_order_id; the broker id was unknown.
    op.execute(
        f"INSERT INTO broker_orders_new (client_order_id, broker_id, {_COPIED}) "
        f"SELECT client_order_id, NULLIF(broker_id, client_order_id), {_COPIED} "
        "FROM broker_orders"
    )
    op.drop_table("broker_orders")
    op.rename_table("broker_orders_new", "broker_orders")


def downgrade() -> None:
    """Downgrade schema."""
    _require_sqlite()
    _create_broker_orders("broker_orders_old", status_labels=_STATUS_OLD, new_key=False)
    # The old schema needs a broker_id; use the client id for unconfirmed orders, as it did.
    op.execute(
        f"INSERT INTO broker_orders_old (client_order_id, broker_id, {_COPIED}) "
        f"SELECT client_order_id, COALESCE(broker_id, client_order_id), {_COPIED} "
        "FROM broker_orders"
    )
    op.drop_table("broker_orders")
    op.rename_table("broker_orders_old", "broker_orders")
