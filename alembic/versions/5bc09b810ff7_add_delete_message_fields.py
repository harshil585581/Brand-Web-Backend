"""Add delete message fields

Revision ID: 5bc09b810ff7
Revises: a25e1d7f44b6
Create Date: 2026-03-17 12:55:54.293959

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5bc09b810ff7'
down_revision: Union[str, Sequence[str], None] = 'a25e1d7f44b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('messages', sa.Column('is_deleted_for_everyone', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('messages', sa.Column('deleted_by_sender', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('messages', sa.Column('deleted_by_receiver', sa.Boolean(), server_default='false', nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('messages', 'deleted_by_receiver')
    op.drop_column('messages', 'deleted_by_sender')
    op.drop_column('messages', 'is_deleted_for_everyone')
