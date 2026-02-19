import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from backend.database import engine

def migrate():
    with engine.connect() as connection:
        # Check if column exists
        check_sql = text("SELECT column_name FROM information_schema.columns WHERE table_name='messages' AND column_name='status';")
        result = connection.execute(check_sql)
        if result.fetchone():
            print("Column 'status' already exists in 'messages' table.")
            return

        print("Adding 'status' column to 'messages' table...")
        try:
            # Add column with default value 'sent'
            alter_sql = text("ALTER TABLE messages ADD COLUMN status VARCHAR DEFAULT 'sent';")
            connection.execute(alter_sql)
            
            # Update existing messages based on is_read (if is_read is true, set status to 'read')
            update_sql = text("UPDATE messages SET status = 'read' WHERE is_read = true;")
            connection.execute(update_sql)
            
            connection.commit()
            print("Migration successful.")
        except Exception as e:
            print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
