from sqlalchemy import text
from backend.database import engine

def migrate():
    with engine.connect() as connection:
        # Check if column exists
        check_sql = text("SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='generations_count';")
        result = connection.execute(check_sql)
        if result.fetchone():
            print("Column 'generations_count' already exists.")
            return

        print("Adding 'generations_count' column to 'users' table...")
        try:
            alter_sql = text("ALTER TABLE users ADD COLUMN generations_count INTEGER DEFAULT 0;")
            connection.execute(alter_sql)
            connection.commit()
            print("Migration successful.")
        except Exception as e:
            print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
