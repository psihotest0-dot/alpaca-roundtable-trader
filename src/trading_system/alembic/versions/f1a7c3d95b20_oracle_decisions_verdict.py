"""Oracle decisions carry a verdict

Revision ID: f1a7c3d95b20
Revises: e8b2d5a61f07
Create Date: 2026-11-02 18:00:00.000000

The Oracle's answer is APPROVE, VETO or HOLD (and an answer that never came is a HOLD marked
`failed`), not the yes/no of the old `approved` column. The table also gets the model's
confidence, the provider that answered, and an index on signal_id for "the final decision for
this signal". Rows written before: approved becomes APPROVE, not approved becomes VETO.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a7c3d95b20"
down_revision: Union[str, Sequence[str], None] = "e8b2d5a61f07"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_VERDICTS = ("APPROVE", "VETO", "HOLD")
_INDEX = "ix_oracle_decisions_signal_id"


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.add_column(sa.Column("verdict", sa.Enum(*_VERDICTS, name="oracleverdict")))
        batch.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch.add_column(sa.Column("provider", sa.String(), nullable=False, server_default=""))
        batch.add_column(
            sa.Column("failed", sa.Boolean(), nullable=False, server_default=sa.false())
        )
    op.execute(
        "UPDATE oracle_decisions SET verdict = CASE WHEN approved THEN 'APPROVE' ELSE 'VETO' END"
    )
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.alter_column(
            "verdict", existing_type=sa.Enum(*_VERDICTS, name="oracleverdict"), nullable=False
        )
        batch.drop_column("approved")
        batch.create_index(_INDEX, ["signal_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.drop_index(_INDEX)
        batch.add_column(sa.Column("approved", sa.Boolean(), nullable=True))
    op.execute("UPDATE oracle_decisions SET approved = (verdict = 'APPROVE')")
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.alter_column("approved", existing_type=sa.Boolean(), nullable=False)
        batch.drop_column("verdict")
        batch.drop_column("confidence")
        batch.drop_column("provider")
        batch.drop_column("failed")
