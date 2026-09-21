#!/usr/bin/env bash
set -euo pipefail

echo "[care] waiting for Elasticsearch ..."
python - <<'PY'
import time
from app.es import wait_for_es, count_docs, ensure_index

es = wait_for_es(timeout=180)
# Wait until cluster accepts index ops
for i in range(30):
    try:
        ensure_index(es)
        print("[care] indexed docs:", count_docs(es))
        break
    except Exception as e:
        print(f"[care] ensure_index retry {i+1}: {e}")
        time.sleep(2)
else:
    raise RuntimeError("failed to ensure index")
PY

if [[ "${CARE_AUTO_INDEX:-1}" == "1" ]]; then
  count=$(python -c "from app.es import count_docs; print(count_docs())" || echo 0)
  if [[ "$count" == "0" ]]; then
    echo "[care] empty index — running index_memes ..."
    python -m scripts.index_memes
  else
    echo "[care] index already has $count docs — skip indexing"
  fi
fi

echo "[care] starting API ..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
