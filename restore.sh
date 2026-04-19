#!/bin/bash
# Restore the database data from a compressed SQL dump.
# This runs automatically on first boot (docker-entrypoint-initdb.d).
# init.sql (01-init.sql) creates the schema first, then this script
# loads the data.

DUMP_FILE="/scholargraph_data.sql.gz"

if [ -f "$DUMP_FILE" ]; then
    echo "Restoring database data from dump (this may take a few minutes)..."
    gunzip -c "$DUMP_FILE" | psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -q
    CHUNK_COUNT=$(psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c 'SELECT COUNT(*) FROM chunks;' | tr -d ' ')
    echo "Database restore complete. $CHUNK_COUNT chunks loaded."
else
    echo "No dump file found, starting with empty database."
    echo "Run: python3 scripts/load_jsonl.py chunked_results.jsonl"
fi
