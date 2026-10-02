#!/usr/bin/env bash
# One-time Unity Catalog setup shared by every model in this repo (idempotent).
# The scripts also self-create these, so this is only needed if you prefer to pre-create them.
#
# Usage:  CATALOG=main PROFILE=DEFAULT ./setup.sh
set -euo pipefail

: "${CATALOG:?Set CATALOG to a Unity Catalog you can write to, e.g. CATALOG=main PROFILE=DEFAULT ./setup.sh}"
PROFILE="${PROFILE:-DEFAULT}"
SCHEMA="${SCHEMA:-air_samples}"

echo "Using catalog=$CATALOG schema=$SCHEMA profile=$PROFILE"
databricks current-user me --profile "$PROFILE" >/dev/null

databricks schemas create "$SCHEMA" "$CATALOG" --profile "$PROFILE" >/dev/null 2>&1 \
  && echo "Created schema $CATALOG.$SCHEMA" || echo "Schema $CATALOG.$SCHEMA already exists (ok)"
databricks volumes create "$CATALOG" "$SCHEMA" predictions MANAGED --profile "$PROFILE" >/dev/null 2>&1 \
  && echo "Created volume $CATALOG.$SCHEMA.predictions" || echo "Volume predictions already exists (ok)"

echo
echo "Point the CLI jobs at this catalog without editing files:"
echo "  (cd models/<model>/01_ai_runtime/cli && air run --file batch_inference.yaml --override env_variables.UC_CATALOG=$CATALOG --watch --profile $PROFILE)"
