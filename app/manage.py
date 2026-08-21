"""Offline operations command: migrate, backup, restore, and local account provision."""
from __future__ import annotations
import argparse, os
from app.services.auth_service import LocalAuthService
from app.services.operations_service import backup_database, restore_database, storage_health

def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="KERASCAN local offline maintenance")
    actions=parser.add_subparsers(dest="action",required=True)
    actions.add_parser("migrate")
    backup=actions.add_parser("backup");backup.add_argument("--output",required=True)
    restore=actions.add_parser("restore");restore.add_argument("--input",required=True)
    actions.add_parser("health")
    user=actions.add_parser("create-user");user.add_argument("--operator-id",required=True);user.add_argument("--role",choices=["operator","reviewer","administrator"],required=True);user.add_argument("--password",required=True)
    args=parser.parse_args(argv);url=os.environ.get("KERASCAN_DB_URL",f"sqlite+pysqlite:///{os.path.expanduser('~/.kerascan/kerascan.db')}")
    from app.database import SessionLocal, init_db
    if args.action=="migrate": init_db();print("Local database migration complete.")
    elif args.action=="backup": print(backup_database(url,args.output))
    elif args.action=="restore": print(restore_database(url,args.input));init_db()
    elif args.action=="create-user":
        init_db()
        with SessionLocal() as session:
            LocalAuthService(session).create_account(args.operator_id,args.password,args.role);session.commit()
        print("Local account created.")
    else:
        db_path=url.removeprefix("sqlite+pysqlite:///")
        print(storage_health(db_path))
    return 0
if __name__=="__main__":raise SystemExit(main())
