# Working on this repository

Read the README and the relevant technical guide before editing. Keep physical
observations distinct from CAD checks and offline tests. Current V7 hardware
limits are in `artifacts/g1_mount_v7/FIT_STATUS.md`.

Do not run robot commands, install a watchdog, collect data, or fit another
calibration as part of repository or documentation maintenance. Such work needs
an explicit operator request and the applicable investigation record. Existing
calibration results must not be treated as instructions to repeat experiments.

Keep running notes, internal proposals and raw experiment directories private.
Use `.local/` for working records. Public Markdown should explain implemented
behavior, reproducible interfaces, results and limitations. Preserve upstream
notices and distinguish nominal CAD transforms from fitted registrations.

The checked-in bundle and base URDF are hash-bound. Do not silently change their
values or the selected calibration. Keep generated files outside source paths
unless a deliberate CAD revision is being made.
