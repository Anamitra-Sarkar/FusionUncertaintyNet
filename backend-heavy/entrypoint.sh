#!/bin/sh
# Downloads the approved checkpoint revision from HF Hub into MODEL_PATH
# before starting the server, if MODEL_RELEASE_APPROVED + MODEL_ARTIFACT_REVISION
# are set. If not approved, starts anyway -- app/main.py's release_configuration()
# correctly fails closed (503) with no model loaded.
set -e

if [ "$MODEL_RELEASE_APPROVED" = "true" ] && [ -n "$MODEL_ARTIFACT_REVISION" ] && [ -n "$MODEL_PATH" ]; then
  echo "[entrypoint] Downloading approved checkpoint revision $MODEL_ARTIFACT_REVISION into $MODEL_PATH"
  python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='bhumika-tewari-282006/fusionuncertaintynet-best-v2-leakfree',
    revision=os.environ['MODEL_ARTIFACT_REVISION'],
    local_dir=os.environ['MODEL_PATH'],
    token=os.environ.get('HF_TOKEN'),
)
print('[entrypoint] Download complete.')
"
else
  echo "[entrypoint] No approved release configured; starting in fail-closed abstention mode."
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 7860
