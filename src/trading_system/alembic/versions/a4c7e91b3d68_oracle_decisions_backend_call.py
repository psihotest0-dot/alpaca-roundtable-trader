"""oracle_decisions says in which mode and by which call an answer came

Revision ID: a4c7e91b3d68
Revises: b6d1f4a82c35
Create Date: 2026-11-06 12:00:00.000000

The Oracle now answers through the Go backend (task 2.2), first in shadow mode: the model is
asked and its answer is recorded, but the loop is told APPROVE whatever it said. Four columns
make such a row readable later:

* `mode`: "shadow" or "active" (empty for rows made before, and by providers without a mode),
* `error_kind`: why there was no answer (unavailable, timeout, unauthorized, bad_request,
  bad_reply, internal), empty when the model answered,
* `request_id` and `model`: to find the call in the backend's own journal.

Old rows keep working: `mode` is empty for them, the others are NULL.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a4c7e91b3d68"
down_revision: Union[str, Sequence[str], None] = "b6d1f4a82c35"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.add_column(sa.Column("mode", sa.String(), nullable=False, server_default=""))
        batch.add_column(sa.Column("error_kind", sa.String(), nullable=True))
        batch.add_column(sa.Column("request_id", sa.String(), nullable=True))
        batch.add_column(sa.Column("model", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.drop_column("model")
        batch.drop_column("request_id")
        batch.drop_column("error_kind")
        batch.drop_column("mode")
