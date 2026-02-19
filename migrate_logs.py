from sqlalchemy import text
from backend.database import engine

def migrate():
    with engine.connect() as connection:
        print("Creating 'activity_logs' table...")
        try:
            create_sql = text("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                action VARCHAR,
                details VARCHAR,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """)
            connection.execute(create_sql)
            connection.commit()
            print("Migration successful.")
        except Exception as e:
            print(f"Migration failed: {e}")

if __name__ == "__main__":
    migrate()
