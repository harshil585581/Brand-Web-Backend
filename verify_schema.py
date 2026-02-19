import sys
import os
from sqlalchemy import create_engine, text, inspect
# Add the parent directory to sys.path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import DATABASE_URL

def fix_schema():
    engine = create_engine(DATABASE_URL)
    inspector = inspect(engine)
    
    if not inspector.has_table("users"):
        print("Table 'users' does not exist. It should be created by the app.")
        return

    columns = [col['name'] for col in inspector.get_columns("users")]
    print(f"Current columns: {columns}")

    with engine.connect() as conn:
        if "permissions" not in columns:
            print("Adding 'permissions' column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN permissions VARCHAR"))
            conn.commit()
        
        if "role" not in columns:
            print("Adding 'role' column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN role VARCHAR DEFAULT 'employee'"))
            conn.commit()
            
        if "profile_img" not in columns:
            print("Adding 'profile_img' column...")
            conn.execute(text("ALTER TABLE users ADD COLUMN profile_img VARCHAR"))
            conn.commit()

    # Re-inspect
    columns_after = [col['name'] for col in inspector.get_columns("users")]
    print(f"Columns after check: {columns_after}")

if __name__ == "__main__":
    fix_schema()
