#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
export DOCQA_OPEN_BROWSER=true
exec bash scripts/start.sh
