#!/usr/bin/env bash
set -euo pipefail

echo "Validating FAISS/index assets..."
python scripts/check_faiss.py || {
  echo "FAISS validation failed; continuing startup so API can still serve non-index routes."
}

exec "$@"
