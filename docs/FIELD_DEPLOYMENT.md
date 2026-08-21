# Offline field deployment

Install from locally approved packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links /path/to/wheelhouse -r app/requirements.txt
python -m app.manage migrate
python -m app.manage create-user --operator-id field_admin --role administrator --password 'local-only-long-passphrase'
streamlit run app/streamlit_app.py
```

The application has no telemetry, remote logging, cloud APIs, or unauthenticated operator access. Each operator needs a local role: `operator`, `reviewer`, or `administrator`; only reviewer/administrator accounts may attempt a decision override.

Set `KERASCAN_LOCAL_IMAGE_DIR` and `KERASCAN_LOCAL_OUTPUT_DIR` to protected local directories when the defaults under `~/.kerascan/` are unsuitable. Source images are retained only in that configured local image directory; no image previews are written to the repository or aggregate evaluation results.

Use operating-system full-disk encryption and access-controlled local directories for the database, source images, backups, and reports. KERASCAN does not silently transmit or synchronise these files.

Backup and restore use SQLite's local backup API:

```bash
python -m app.manage backup --output /protected/backups/kerascan.db
python -m app.manage restore --input /protected/backups/kerascan.db
python -m app.manage health
```

For an operational export without direct encounter identifiers or source-image paths, use `app.services.deidentified_export_service.export_deidentified(...)` with local repository records and a protected local output path.

Before field use: verify free storage, successful backup/restore on a non-production copy, database migration, local account provisioning, printer/PDF generation, recapture instructions, and a safe shutdown after all reports are written. Do not store source-image previews outside the configured local export location.
