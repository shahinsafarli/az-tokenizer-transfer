# Repository evidence paths

Repository documents use these root-relative locations:

- `evidence/source_snapshot/` preserves evidence imported from the edited source snapshot.
- `evidence/carried_forward_pre_grid/` preserves material created before the final grid.
- `results/` contains the retained completed run records, logs, tables, and ledgers.
- `evidence/environment/` records the captured execution environment.
- `execution/run.log` names the separately retained release asset `run.log.txt`; it is not a tracked Git file.

The edited source snapshot and the earlier as-run evidence are separate provenance layers. A path alias does not imply that an experiment was rerun.
