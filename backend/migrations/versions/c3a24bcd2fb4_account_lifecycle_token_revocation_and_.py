"""account lifecycle: token revocation and password reset

Revision ID: c3a24bcd2fb4
Revises: b7d2e91f4a05
Create Date: 2026-09-14 19:02:26.792616

token_version starts at 0 for every existing account, and tokens issued before
this migration carry no version claim at all — which is read as 0. So nobody is
signed out by the deploy; the counter only starts mattering the first time
somebody signs out, changes their password, or completes a reset.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c3a24bcd2fb4'
down_revision: Union[str, None] = 'b7d2e91f4a05'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLES = ("patients", "clinicians")


def upgrade() -> None:
    for table in TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            # server_default so the column can be NOT NULL on a table that
            # already has rows; the model's own default takes over for inserts
            # from here on.
            batch_op.add_column(sa.Column('token_version', sa.Integer(),
                                          nullable=False, server_default='0'))
            batch_op.add_column(sa.Column('reset_token_hash', sa.String(length=64), nullable=True))
            batch_op.add_column(sa.Column('reset_requested_at', sa.DateTime(timezone=True),
                                          nullable=True))


def downgrade() -> None:
    for table in reversed(TABLES):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_column('reset_requested_at')
            batch_op.drop_column('reset_token_hash')
            batch_op.drop_column('token_version')
