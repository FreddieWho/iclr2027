#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-actionmode-research}"
mkdir -p "$DEST"/{configs,data,src/actionmode/{data,graphs,interventions,probes,models,tasks,eval,figures},scripts,tests,reports,artifacts,literature}

cat > "$DEST/STATUS.md" <<'EOT'
# STATUS

Current phase: P0_BOOTSTRAP
Study mode: exploratory_discovery
Last updated:

## Completed

## Active

## Blocked

## Next exploration
Establish the initial data and intervention pipeline.
EOT

cat > "$DEST/DECISIONS.md" <<'EOT'
# DECISIONS

Record important scientific and engineering decisions with date, evidence, alternatives considered, and later revisions.
EOT

cat > "$DEST/CLAIM_LEDGER.md" <<'EOT'
# CLAIM LEDGER

| ID | Claim | Status | Supporting evidence | Caution |
|---|---|---|---|---|
| C1 | Matched-energy reversal exists | open | exploration checkpoint 1 | avoid generalizing to all models |
| C2 | Organization exceeds frequency | open | exploration checkpoint 2 | do not equate with understands tactics |
| C3 | Mechanism explains the pattern | open | exploration checkpoint 3 | distinguish correlation from mechanism |
| C4 | AMR improves the useful frontier | open | exploration checkpoint 4 | do not equate with solves relational reasoning |
| C5 | Sports observations inform calligraphy | open | exploration checkpoint 5 | state mapping versions and scope |
EOT

cat > "$DEST/reports/exploration_log.yaml" <<'EOT'
version: 1
study_mode: exploratory_discovery
entries: []
EOT

printf 'Initialized research repository at %s\n' "$DEST"
