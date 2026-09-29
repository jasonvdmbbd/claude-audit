---
description: Backfill the machine-wide audit DB from all existing Claude Code transcripts
---

Sync the corporate-transparency audit database — machine-wide, every workspace.

Run:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/audit_log.py" sync --all --classify
```

This discovers every workspace Claude Code has transcripts for under `~/.claude/projects/` (recovering each one's real path from its own transcripts, since the project directory name is a lossy slug), and for each one backfills any sessions/turns not yet in the central database at `~/.claude/audit/audit.db` (or wherever `$CLAUDE_AUDIT_DB` points — idempotent either way, already-logged turns are skipped), archives raw transcripts (named `<workspace-folder>-<session_id>` so one archive directory can hold every workspace without collisions), and tags/titles untagged turns with Haiku.

After it finishes, report to the user:
- how many workspaces were swept, and how many were skipped (and why — e.g. no recoverable path, or the workspace folder no longer exists),
- how many sessions and turns were newly added vs already present, rolled up across all workspaces (the command's own `audit sync totals:` block),
- how many turns were classified (and how many fell back to heuristic tagging),
- any errors from the command output or `hook-errors.log` next to the database (`~/.claude/audit/hook-errors.log` by default).

To browse the results, tell the user to open `${CLAUDE_PLUGIN_ROOT}/views/viewer.html` in a browser and pick the central `audit.db` — its workspace dropdown lets them narrow the view to one project at a time.
