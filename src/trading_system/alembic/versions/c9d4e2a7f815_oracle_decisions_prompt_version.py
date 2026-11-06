"""oracle_decisions keeps the backend's label of the prompt behind an answer

Revision ID: c9d4e2a7f815
Revises: a4c7e91b3d68
Create Date: 2026-11-06 22:30:00.000000

Since 2026-11-06 22:11 German time the Go backend returns `prompt_version` (12 hex characters: a hash of
the rendered system prompt, the answer schema, the temperature, the token limit, the attempts, the reason
limit and the layout of the facts) in the reply of /oracle/evaluate. A change of model or label starts a
new period of the evaluation (docs/oracle-provider-task.md, section 5), so the bot keeps the label with
each answer. NULL for the answers made before the label existed and for calls that got no reply.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c9d4e2a7f815"
down_revision: Union[str, Sequence[str], None] = "a4c7e91b3d68"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.add_column(sa.Column("prompt_version", sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("oracle_decisions") as batch:
        batch.drop_column("prompt_version")
