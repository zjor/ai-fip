#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_dir=$(dirname "$script_dir")

set -a
. "$project_dir/.env.local"
set +a

: "${YOUTRACK_API_TOKEN:?YOUTRACK_API_TOKEN is missing from ai-fip .env.local}"

exec python3 -c '
import json
import os

print(json.dumps({
    "Authorization": "Bearer " + os.environ["YOUTRACK_API_TOKEN"],
}))
'
