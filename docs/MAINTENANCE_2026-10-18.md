# Zenodo inventory transient-failure recovery

The prior inventory workflow failed with HTTP 504 while querying the pinned
Zenodo record. Record acquisition now retries transient connection failures,
timeouts and HTTP 408/429/500/502/503/504 up to four attempts. Backoff is bounded
at 30 seconds, including Retry-After hints. Permanent HTTP errors, malformed or
empty inventories and exhausted retries still fail. No cached inventory is
substituted for a successful live response and no checksum is changed.

A fresh request returned all 12 artifacts at record 17651628, revision 4; the
received inventory is retained in `validation/maintenance_2026_10_18/live_inventory.json`.
No large simulation shard was downloaded or trained in this check.

Local Python 3.12: 67 tests passed, three optional torch tests skipped; the 12
targeted reference/retry checks passed. Ruff 0.6.9 passes the changed files.
Inventory availability is not surrogate accuracy or clinical qualification.
