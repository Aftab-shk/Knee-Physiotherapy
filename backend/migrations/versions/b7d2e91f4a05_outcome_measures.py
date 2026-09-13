"""standardised outcome measures

Adds the table behind KOOS-JR: one row per completed questionnaire.

Both the seven raw answers and the derived score are stored. The answers are the
evidence — a corrected lookup table or a second instrument can be recomputed from
them — while the score is written once, so a chart does not silently redraw its
own history the day the scoring code changes.

Revision ID: b7d2e91f4a05
Revises: cc19af88dbbd
Create Date: 2026-09-03 11:42:18.905331
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b7d2e91f4a05'
down_revision: Union[str, None] = 'cc19af88dbbd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'outcome_scores',
        sa.Column('id', sa.String(length=32), nullable=False),
        sa.Column('patient_id', sa.String(length=32), nullable=False),
        sa.Column('instrument', sa.String(length=24), nullable=False),
        sa.Column('knee_side', sa.String(length=8), nullable=False),
        sa.Column('responses', sa.Text(), nullable=False),
        sa.Column('raw_sum', sa.Integer(), nullable=False),
        sa.Column('interval_score', sa.Float(), nullable=False),
        sa.Column('weeks_post_op', sa.Integer(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    with op.batch_alter_table('outcome_scores', schema=None) as batch_op:
        batch_op.create_index(
            'ix_outcome_scores_patient_recorded', ['patient_id', 'recorded_at'], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table('outcome_scores', schema=None) as batch_op:
        batch_op.drop_index('ix_outcome_scores_patient_recorded')

    op.drop_table('outcome_scores')
