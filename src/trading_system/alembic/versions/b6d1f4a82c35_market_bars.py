"""Closed candles from Alpaca are kept in the database

Revision ID: b6d1f4a82c35
Revises: f1a7c3d95b20
Create Date: 2026-11-02 23:30:00.000000

The backtest needs real candles and the live loop downloads the same 30 and 90 days again on
every cycle. `market_bars` keeps them, one row per symbol, timeframe and bar start. It is a
cache: dropping the table loses nothing that `fetch-history` cannot fetch again.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b6d1f4a82c35"
down_revision: Union[str, Sequence[str], None] = "f1a7c3d95b20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "market_bars",
        sa.Column("symbol", sa.String(), nullable=False),
        sa.Column("timeframe", sa.String(), nullable=False),
        sa.Column("bar_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("volume", sa.Float(), nullable=False),
        sa.Column("vwap", sa.Float(), nullable=True),
        sa.Column("trade_count", sa.Integer(), nullable=True),
        sa.Column("adjustment", sa.String(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("symbol", "timeframe", "bar_start"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("market_bars")
