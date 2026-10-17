"""CLI script to create an admin user directly in the database (Phase G)."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth import hash_password
from app.db.migrate import get_db, run_migrations
from app.db.users import create_user, get_user_by_email


def main():
    parser = argparse.ArgumentParser(description="Create an admin user directly.")
    parser.add_argument("--email", type=str, default="admin@example.com", help="Admin email")
    parser.add_argument("--password", type=str, default="admin123456", help="Admin password")
    args = parser.parse_args()

    # Ensure migrations are applied
    run_migrations()

    conn = get_db()
    try:
        existing = get_user_by_email(args.email, conn=conn)
        if existing:
            print(f"User with email '{args.email}' already exists (id: {existing['id']}, role: {existing['role']}).")
            return

        pwd_hash = hash_password(args.password)
        user = create_user(email=args.email, password_hash=pwd_hash, role="admin", conn=conn)
        print(f"Admin user successfully created! (id: {user['id']}, email: {user['email']}, role: {user['role']})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
