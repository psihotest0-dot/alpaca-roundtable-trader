"""Take-profit price on order intents

Revision ID: e8b2d5a61f07
Revises: c3f1a9e47d52
Create Date: 2026-11-02 10:00:00.000000

The risk engine now decides all three prices of the bracket (limit, stop, take-profit)
and hands them to execution in the order intent (C3). Execution used to compute the
target itself, so the intent had no column for it. Existing rows keep NULL: they were
never executed with a target of their own.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8b2d5a61f07"
down_revision: Union[str, Sequence[str], None] = "c3f1a9e47d52"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("order_intents") as batch:
        batch.add_column(
            sa.Column("take_profit_price", sa.Numeric(precision=18, scale=8), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    # SQLite has no DROP COLUMN in older versions; batch mode rebuilds the table.
    with op.batch_alter_table("order_intents") as batch:
        batch.drop_column("take_profit_price")
