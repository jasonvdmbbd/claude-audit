#!/usr/bin/env python3
"""Corporate-transparency audit log for Claude Code.

Wired in as three hooks (UserPromptSubmit / Stop / SessionEnd), this script
mirrors every completed prompt turn out of Claude Code's own (time-limited)
JSONL transcript into a durable machine-wide SQLite database, then tags and
titles each turn out-of-band with Haiku. The subagents a turn spawns -- Task
agents and workflow steps alike -- are mirrored from their own sidecar
transcripts and joined back to the parent turn.

ONE DATABASE PER MACHINE. Every workspace logs into ~/.claude/audit/audit.db
(override with $CLAUDE_AUDIT_DB, or an explicit --db on a subcommand). The
workspace a turn belongs to lives in sessions.workspace, cursors live inside
the database, and archived transcripts are named <workspace>-<session>, so one
file can serve every project on the machine. Writing a per-workspace database
is still possible -- it is just what --db pointing elsewhere now means.

Invocation modes
    <hook json on stdin>                  dispatched on `hook_event_name`
    python3 audit_log.py init     [--db P]  create/upgrade the schema
    python3 audit_log.py classify [--db P] --turn N   tag+title one turn
    python3 audit_log.py classify [--db P] --agent SESSION:AGENT
                                          tag+title one subagent
    python3 audit_log.py sync [--classify] [--db P] [--projects-dir D]
                                          backfill EVERY workspace found under
                                          the projects root (the default)
    python3 audit_log.py sync --workspace P [--classify] [--db P]
                                          narrow the backfill to one workspace
    python3 audit_log.py merge --from SOURCE.db [--db DEST]
                                          fold a backed-up audit database into
                                          a live one (corporate restore)

Design rules
    * Never fail the turn: every path swallows exceptions, logs to
      <audit dir>/hook-errors.log and exits 0.
    * Never write to stdout in hook mode -- the harness parses it.
      CLI subcommands are not hooks: they print, and may exit nonzero.
    * Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The classifier model rides whatever routing the local `claude` CLI has —
# first-party by default, but a proxy/Vertex environment can point it at its
# own model name (e.g. "vertex-ai/claude-haiku-4-5@20251001") via this env
# var, set in the shell or in settings.json's "env" block so hooks see it.
HAIKU_MODEL = os.environ.get("CLAUDE_AUDIT_CLASSIFY_MODEL", "claude-haiku-4-5-20251001")
CLASSIFY_TIMEOUT_S = 120
MAX_RESULT_CHARS = 10_000       # per tool result stored
MAX_INPUT_CHARS = 10_000        # per tool input stored
MAX_CLASSIFY_CHARS = 4_000      # per field handed to Haiku
MAX_AGENT_TEXT_CHARS = 10_000   # per subagent prompt / result stored
STRUCTURED_OUTPUT_TOOL = "StructuredOutput"
TITLE_MAX_CHARS = 80
SCHEMA_VERSION = "1"

DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# The local viewer service (scripts/service.py). The hooks keep it alive: a
# 0.3s loopback connect at the end of Stop/SessionEnd, and a detached respawn
# when nothing answers. Set CLAUDE_AUDIT_NO_SERVICE=1 to opt out entirely.
SERVICE_SCRIPT = "service.py"
SERVICE_HOST = "127.0.0.1"
SERVICE_DEFAULT_PORT = 4737
SERVICE_PORT_ENV_VAR = "CLAUDE_AUDIT_PORT"
SERVICE_DISABLE_ENV_VAR = "CLAUDE_AUDIT_NO_SERVICE"
SERVICE_PROBE_TIMEOUT_S = 0.3

# THE AUDIT IS MACHINE-WIDE. One database holds every workspace on the box;
# sessions.workspace says which project a session belongs to. The env var is
# for fleet deployments that want the file somewhere else (a mounted volume,
# a per-user share); an explicit --db beats it, and is the only way to get the
# old per-workspace behaviour back.
DEFAULT_AUDIT_DIR = Path.home() / ".claude" / "audit"
DEFAULT_DB_FILE = DEFAULT_AUDIT_DIR / "audit.db"
DB_ENV_VAR = "CLAUDE_AUDIT_DB"

# The human tag taxonomy. Single source of truth: VALID_TAGS is derived from
# it, the Haiku rubric is rendered from it, and the classifier validates the
# model's answer against it -- so adding a tag here is the whole change.
# (system:* kinds are NOT in here: they are assigned deterministically from the
# prompt text by synthetic_kind(), never chosen by a model.)
TAG_TAXONOMY = (
    ("plan", "exploring, researching, designing, deciding an approach"),
    ("build", "writing or changing product code/features"),
    ("fix", "debugging and repairing something broken"),
    ("validate", "testing, reviewing, verifying behavior"),
    ("docs", "writing documentation, READMEs, runbooks"),
    ("ops", "environment/tooling work: installs, hooks, config, CI, deploys"),
    ("chat", "conversational Q&A with no work product"),
)
VALID_TAGS = tuple(name for name, _ in TAG_TAXONOMY)

# Deterministic fallback, used whenever the model call is unavailable. First
# match wins, and the list is ordered by how strongly the words imply the tag:
# "fix the failing test" is a fix, not a validate. Checked only AFTER the
# permission_mode == 'plan' rule, which is evidence rather than vocabulary.
HEURISTIC_KEYWORDS = (
    ("fix", ("fix", "bug", "error", "broken", "crash", "debug")),
    ("validate", ("test", "verify", "validate", "review", "lint", "assert")),
    ("docs", ("readme", "document", "docs", "runbook")),
    ("ops", ("install", "deploy", "config", "hook", "setup", "plugin", "provision")),
)
# A prompt opening with one of these (or ending in '?') is being asked, not
# instructed -- and if the turn then called no tools, nothing was built.
QUESTION_OPENERS = ("why", "what", "how", "is", "are", "can", "does")

# --------------------------------------------------------------------------
# Tool-call tagging vocabulary
# --------------------------------------------------------------------------
#
# Every tool call -- a turn's and a subagent's alike -- carries the same seven
# tags a turn does (tool_calls.tag / agent_tool_calls.tag) plus a one-line
# human summary. NO MODEL IS INVOLVED: a tool call says in its own name and
# input what kind of work it is, so the mapping is a pure function of the two
# and a backfill over old rows therefore reaches exactly the verdict capture
# would have. The tag vocabulary is TAG_TAXONOMY's, unchanged, so a tool-call
# tag and a turn tag mean the same thing and can be counted together.
TOOL_TAG_ASK = ("AskUserQuestion",)
TOOL_TAG_EDIT = ("Edit", "Write", "MultiEdit", "NotebookEdit")
TOOL_TAG_READ = (
    "Read",
    "Grep",
    "Glob",
    "WebFetch",
    "WebSearch",
    "NotebookRead",
    "LS",
    "ListDir",
)
TOOL_TAG_SPAWN = ("Agent", "Workflow", "Task")

# An edit is `docs` or `build` purely by what it writes to.
DOC_SUFFIXES = (".md", ".markdown")

# Where a tool names the file it is working on, most specific key first.
PATH_INPUT_KEYS = ("file_path", "notebook_path", "filePath", "path")

# The spawn tools describe their work in prose, so they are read with the turn
# heuristic's vocabulary (HEURISTIC_KEYWORDS), pared down to the words that
# actually appear in a subagent description and extended with the `plan` group
# a turn gets from its permission mode instead. First match wins, same as
# there; matching is plain substring, so "testing" and "documentation" count.
#
# `read` is the one word that cannot be a substring test: it sits in the plan
# group, which is consulted BEFORE the docs group, and "write the readme"
# would come out as planning. Matched as the verb (read/reads/reading) it
# leaves "readme" to the group that owns it.
READ_VERB_RE = re.compile(r"(?<![A-Za-z0-9_])read(?:s|ing)?(?![A-Za-z0-9_])", re.IGNORECASE)
DESCRIPTION_TAG_KEYWORDS = (
    ("fix", ("fix", "bug", "error")),
    ("validate", ("test", "verify", "review")),
    ("plan", ("research", "explore", "investigate", READ_VERB_RE)),
    ("docs", ("document", "readme")),
    ("ops", ("install", "setup", "deploy")),
)

# Bash is read off the command line itself (plus its description). Ordered
# most-specific first: a `git` invocation inside a test command is still a
# test. Matched at a word START rather than anywhere in the text -- `ls` must
# not fire on "tools/", and `cat` must not fire on "concatenate" -- while a
# trailing suffix is still allowed, so "installing" matches `install` and
# "tests" matches `test`.
BASH_TAG_KEYWORDS = (
    ("validate", ("test", "pytest", "lint", "--check", "validate", "verify")),
    (
        "ops",
        (
            "install",
            "deploy",
            "launchctl",
            "systemctl",
            "config",
            "chmod",
            "mkdir",
            "plugin",
            "git",
            "curl",
        ),
    ),
    ("plan", ("grep", "find", "ls", "cat", "head", "tail", "wc", "lsof")),
)

# Past-tense verb for the file-touching tools' summaries.
TOOL_SUMMARY_VERBS = {
    "Read": "Read",
    "NotebookRead": "Read",
    "Edit": "Edited",
    "MultiEdit": "Edited",
    "NotebookEdit": "Edited",
    "Write": "Wrote",
}

SUMMARY_MAX_CHARS = 100          # one line in the viewer, never more
BASH_SUMMARY_CHARS = 60          # of the command, when it described itself not
QUESTION_SUMMARY_CHARS = 50      # of the first question put to the user

# Recovers `"key": "value"` pairs from an input payload too long to have been
# stored whole (see tool_input_dict). Anchored on the quote before the key so
# it cannot match inside a value, and the value is JSON-decoded rather than
# used raw, so escapes survive.
TRUNCATED_FIELD_RE = re.compile(
    r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*("(?:[^"\\]|\\.)*")'
)

# A user-role transcript line is not always something a human typed. Interrupt
# notices, slash-command expansions and harness-injected context all arrive
# wearing the same clothes. They are still logged -- corporate transparency
# wants completeness -- but they are flagged so reports and the classifier can
# tell a real instruction from machinery.
SYNTHETIC_PREFIXES = ("[Request interrupted",)
SYNTHETIC_MARKERS = (
    "<command-name>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<system-reminder>",
    "<task-notification>",
)

# MID-TURN INTERJECTIONS. A message the human sends WHILE the assistant is
# still working is not a new turn -- Claude Code injects it into the running
# turn and the assistant answers it as part of that same turn. Recorded as a
# turn of its own it becomes a PHANTOM: it steals the rest of the enclosing
# turn's tool calls, text and tokens, it has no response of its own, it stays
# untagged, and because the transcript's parent chain never runs through it,
# it floats in the flow view as a lineage orphan.
#
# WHAT THE TRANSCRIPTS ACTUALLY LOOK LIKE (measured over every transcript in
# ~/.claude/projects on this machine -- 449 files -- while writing this):
#
#   * A mid-turn human message is NOT a `user` line at all. It is written as
#     type "attachment" with attachment.type == "queued_command" and
#     commandMode "prompt", carrying the text in attachment.prompt, and the
#     queue-operation line beside it says reason "absorbed_mid_turn". 23 of
#     these exist here and NONE of them also appears as a real prompt line --
#     so the text was being dropped on the floor entirely, while the
#     UserPromptSubmit hook still opened a row for it. Those hook rows are the
#     phantoms actually found in the database.
#   * The harness writes that attachment AFTER the tool_result it rides along
#     with, so at that point NOTHING is awaiting a tool result. A pending-tool
#     test alone would therefore miss every real case; the attachment shape is
#     the reliable signal and is handled unconditionally. The pending-tool
#     test is kept as well, for the shape where an interjection does arrive as
#     a `user` line (older/newer clients); it currently fires on zero lines
#     here, so it changes nothing about existing data.
#   * TASK NOTIFICATIONS appear BOTH ways and the two forms mean different
#     things: as a real `user` line they always arrive BETWEEN turns (measured:
#     0 of 172 mid-flight) and stay their own system:task-result turn; as a
#     queued_command attachment with commandMode "task-notification" they were
#     injected into a running turn (34 here). Only the injected form is folded
#     in, and only as a one-line marker -- their payload is bulky and already
#     lives in the spawning tool call's result.
INTERJECTION_HEADER = "--- mid-turn addition ---"
TASK_DONE_HEADER = "--- background task completed: {0} ---"

# Background-task linkage. Only these tools ever answer with a task id, so no
# other tool result is even looked at (see parse_task_link): a Read result
# quoting "task_id: 7" out of a source file must not be mistaken for a spawn.
TASK_LINK_TOOLS = ("Bash", "Agent", "Workflow")

# The id a spawning call hands back, first match wins. The harness has worded
# this several ways over time and all of them are still in stored transcripts:
# "Task ID: x" / "task_id: x" (one pattern, the separator is the only
# difference), "Command running in background with ID: x", and an `agentId`
# hex for a background Agent spawn. An id is [A-Za-z0-9_-]+; an optional
# leading quote is consumed so a JSON-shaped result parses the same way.
TASK_ID_PATTERNS = (
    re.compile(r"\btask[ _-]?id\s*[:=]\s*\"?([A-Za-z0-9_-]+)", re.IGNORECASE),
    re.compile(
        r"running in background with id\s*[:=]\s*\"?([A-Za-z0-9_-]+)", re.IGNORECASE
    ),
    re.compile(r"\bagent[ _-]?id\s*[:=]\s*\"?([0-9a-fA-F]{6,})", re.IGNORECASE),
)

# A SECOND reference out of the same result text, one that bridges to the
# agents table: a workflow run id (agents.workflow_id) or the agentId hex
# (agents.agent_id). Kept apart from TASK_ID_PATTERNS because the two answer
# different questions -- "which notification closes this call" vs "which
# mirrored agent rows belong to it" -- and a Workflow result carries both.
RUN_REF_PATTERNS = (
    re.compile(r"\brun\s*id\s*[:=]\s*\"?(wf_[A-Za-z0-9-]+)", re.IGNORECASE),
    # Unlabelled, and the form actually observed: a Workflow spawn names its
    # run only inside the transcript directory it reports
    # (.../subagents/workflows/wf_ca5423cd-a45), which is exactly the
    # directory agent_files() derives agents.workflow_id from.
    re.compile(r"\b(wf_[0-9a-fA-F]{6,}-[A-Za-z0-9-]+)"),
    re.compile(r"\bagent[ _-]?id\s*[:=]\s*\"?([0-9a-fA-F]{6,})", re.IGNORECASE),
)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  workspace TEXT NOT NULL,
  git_branch TEXT,
  cc_version TEXT,
  started_at TEXT,
  ended_at TEXT,
  end_reason TEXT
);
CREATE TABLE IF NOT EXISTS turns (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  prompt_id TEXT,
  prompt_uuid TEXT UNIQUE,
  ts_start TEXT,
  ts_end TEXT,
  duration_s REAL,
  model TEXT,
  permission_mode TEXT,
  prompt TEXT,
  response TEXT,
  -- Deliberately unconstrained. The vocabulary (VALID_TAGS) is enforced in
  -- code, where changing it is an edit; as a CHECK it would be a table
  -- rebuild migration for every future taxonomy tweak.
  tag TEXT,
  title TEXT,
  tag_source TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_creation_tokens INTEGER DEFAULT 0,
  thinking_tokens INTEGER DEFAULT 0,
  status TEXT DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS tool_calls (
  tool_call_id INTEGER PRIMARY KEY AUTOINCREMENT,
  turn_id INTEGER NOT NULL REFERENCES turns(turn_id),
  seq INTEGER,
  tool_name TEXT,
  tool_use_id TEXT,
  input_json TEXT,
  result_text TEXT,
  result_truncated INTEGER DEFAULT 0,
  is_error INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agents (
  session_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  turn_id INTEGER,
  prompt_id TEXT,
  tool_use_id TEXT,
  workflow_id TEXT,
  agent_type TEXT,
  description TEXT,
  model TEXT,
  prompt TEXT,
  result TEXT,
  ts_start TEXT,
  ts_end TEXT,
  duration_s REAL,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_creation_tokens INTEGER DEFAULT 0,
  thinking_tokens INTEGER DEFAULT 0,
  tool_call_count INTEGER DEFAULT 0,
  src_bytes INTEGER DEFAULT 0,
  PRIMARY KEY (session_id, agent_id)
);
-- The subagent's own tool calls, one row per tool_use in its transcript.
-- agents.tool_call_count is the tally; this is the flow itself, so the viewer
-- can draw what a subagent actually did rather than just how much it did.
-- Keyed by (session_id, agent_id) -- the agents primary key -- so no id remap
-- is ever needed, and the rows are rebuilt wholesale whenever their agent is
-- (re)parsed, exactly like tool_calls under a turn.
CREATE TABLE IF NOT EXISTS agent_tool_calls (
  atc_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  agent_id TEXT NOT NULL,
  seq INTEGER,
  tool_name TEXT,
  tool_use_id TEXT,
  input_json TEXT,
  input_truncated INTEGER DEFAULT 0,
  result_text TEXT,
  result_truncated INTEGER DEFAULT 0,
  is_error INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cursors (
  session_id TEXT PRIMARY KEY,
  byte_offset INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_agents_turn ON agents(turn_id);
CREATE INDEX IF NOT EXISTS idx_atc_agent ON agent_tool_calls(session_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace);
"""

# Columns added after the first release. SQLite has no ADD COLUMN IF NOT
# EXISTS, so each one is attempted on every open and a "duplicate column"
# complaint is the success case for a database that already has it.
#
# TURN LINEAGE (parent_uuid / last_uuid). Claude Code's transcript is a linked
# list: every line carries its own `uuid` and the `parentUuid` of the line it
# answers. Recording the prompt line's parentUuid and the uuid of the turn's
# final line turns the turns table into a graph the viewer can draw:
#
#   * turn B is a CHILD of turn A when B.parent_uuid == A.last_uuid;
#   * two turns sharing the same parent_uuid are a FORK -- the same point in
#     history was continued twice (a rewind, an edited prompt, a /resume);
#   * a turn whose parent_uuid is NULL (or matches no last_uuid in the
#     session) is a ROOT of its own branch.
#
# Subagents join to their spawning turn through agents.turn_id / prompt_id,
# so the two together describe the whole tree: turn -> turn, turn -> agents.
#
# BACKGROUND-TASK LIFECYCLE (tool_calls.task_id / run_ref, turns.task_id). A
# backgrounded Bash command, a background Agent spawn and a Workflow run all
# answer their spawning tool call with an id, and the harness later injects a
# <task-notification> prompt carrying the same id in <task-id>. Recording both
# ends closes the loop:
#
#   * tool_calls.task_id = X is where the background task was STARTED;
#   * turns.task_id = X is the synthetic turn that reported it FINISHED;
#   * tool_calls.run_ref is a second reference off the same result text --
#     a workflow "Run ID: wf_..." (agents.workflow_id) or an agentId hex
#     (agents.agent_id) -- so the spawning call also reaches the agent rows
#     mirrored from the subagent transcripts.
MIGRATIONS = (
    "ALTER TABLE turns ADD COLUMN is_synthetic INTEGER DEFAULT 0",
    "ALTER TABLE tool_calls ADD COLUMN input_truncated INTEGER DEFAULT 0",
    "ALTER TABLE turns ADD COLUMN parent_uuid TEXT",
    "ALTER TABLE turns ADD COLUMN last_uuid TEXT",
    "ALTER TABLE tool_calls ADD COLUMN task_id TEXT",
    "ALTER TABLE tool_calls ADD COLUMN run_ref TEXT",
    "ALTER TABLE turns ADD COLUMN task_id TEXT",
    # Per-tool-call tag + one-line summary, assigned deterministically at
    # capture time and backfilled by sync (see tool_tag_and_summary). Both
    # tables get them: a subagent's flow is read the same way a turn's is.
    "ALTER TABLE tool_calls ADD COLUMN tag TEXT",
    "ALTER TABLE tool_calls ADD COLUMN summary TEXT",
    "ALTER TABLE agent_tool_calls ADD COLUMN tag TEXT",
    "ALTER TABLE agent_tool_calls ADD COLUMN summary TEXT",
    # A subagent is classified exactly like a turn -- same taxonomy, same
    # Haiku rubric, same heuristic fallback -- so it carries the same three
    # columns (see classify_agent).
    "ALTER TABLE agents ADD COLUMN tag TEXT",
    "ALTER TABLE agents ADD COLUMN title TEXT",
    "ALTER TABLE agents ADD COLUMN tag_source TEXT",
    # How many mid-turn messages were folded into this turn's prompt (see the
    # INTERJECTION_HEADER comment). 0 for the overwhelming majority.
    "ALTER TABLE turns ADD COLUMN interjections INTEGER DEFAULT 0",
)

# Indexes over ALTER-added columns. They cannot live in SCHEMA (which runs
# before the migrations, when the column may not exist yet) and they cannot
# survive the turns rebuild below, so they are (re)created afterwards.
POST_MIGRATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_turns_parent_uuid ON turns(parent_uuid)",
    "CREATE INDEX IF NOT EXISTS idx_turns_last_uuid ON turns(last_uuid)",
    "CREATE INDEX IF NOT EXISTS idx_turns_task_id ON turns(task_id)",
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_task_id ON tool_calls(task_id)",
)

# Older releases constrained turns.tag with a CHECK: first to the three human
# tags, then (v1.2) to those plus system:%. Both are now wrong -- the taxonomy
# is code-side, so the column is plain TEXT and a taxonomy edit never needs a
# migration again. SQLite cannot alter a constraint in place, so an old table
# is rebuilt from this DDL (which includes every ALTER-added column: keep it
# in step with MIGRATIONS) and the rows copied across.
TURNS_REBUILD_DDL = """
CREATE TABLE turns_new (
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(session_id),
  prompt_id TEXT,
  prompt_uuid TEXT UNIQUE,
  ts_start TEXT,
  ts_end TEXT,
  duration_s REAL,
  model TEXT,
  permission_mode TEXT,
  prompt TEXT,
  response TEXT,
  tag TEXT,
  title TEXT,
  tag_source TEXT,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  cache_read_tokens INTEGER DEFAULT 0,
  cache_creation_tokens INTEGER DEFAULT 0,
  thinking_tokens INTEGER DEFAULT 0,
  status TEXT DEFAULT 'pending',
  is_synthetic INTEGER DEFAULT 0,
  parent_uuid TEXT,
  last_uuid TEXT,
  task_id TEXT,
  interjections INTEGER DEFAULT 0
)
"""

TURN_COLUMN_LIST = (
    "turn_id, session_id, prompt_id, prompt_uuid, ts_start, ts_end,"
    " duration_s, model, permission_mode, prompt, response, tag, title,"
    " tag_source, input_tokens, output_tokens, cache_read_tokens,"
    " cache_creation_tokens, thinking_tokens, status, is_synthetic,"
    " parent_uuid, last_uuid, task_id, interjections"
)

TOKEN_COLUMNS = (
    # (db column, transcript usage key)
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("cache_read_tokens", "cache_read_input_tokens"),
    ("cache_creation_tokens", "cache_creation_input_tokens"),
)

def _tag_rubric() -> str:
    width = max(len(name) for name in VALID_TAGS)
    return "\n".join(
        "  {0}  - {1}".format(name.ljust(width), description)
        for name, description in TAG_TAXONOMY
    )


CLASSIFY_INSTRUCTIONS = """\
You are labelling one turn of a software-engineering session for an audit log.

Choose exactly one tag:
{rubric}

If a turn does several things, tag it by its dominant purpose. Prefer `fix`
over `build` when the work is repairing something that was broken, and `fix`
over `validate` when a test was run in order to repair rather than to confirm.
Use `chat` only when the turn produced no work product at all.

Also write a title: one short sentence, plain past tense, describing what this
turn actually did. Maximum {title_max} characters. No trailing period.

Reply with STRICT JSON and nothing else -- no prose, no markdown fences:
{{"tag": "{choices}", "title": "..."}}
""".format(
    rubric=_tag_rubric(),
    title_max=TITLE_MAX_CHARS,
    choices="|".join(VALID_TAGS),
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def utcnow() -> str:
    """UTC ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_ts(value):
    """Parse a transcript ISO-8601 timestamp; None when unparseable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration_seconds(start, end):
    a, b = parse_ts(start), parse_ts(end)
    if a is None or b is None:
        return None
    return round((b - a).total_seconds(), 3)


def default_db_path(override=None) -> Path:
    """The database this invocation writes to.

    Precedence: an explicit --db, then $CLAUDE_AUDIT_DB, then the machine-wide
    ~/.claude/audit/audit.db. Hooks never pass an override, so every workspace
    on the machine lands in one file; passing --db elsewhere is what now
    produces a separate (e.g. per-workspace) database.
    """
    if override:
        return Path(override).expanduser()
    env = os.environ.get(DB_ENV_VAR)
    if env and env.strip():
        return Path(env.strip()).expanduser()
    return DEFAULT_DB_FILE


def audit_dir(db_override=None) -> Path:
    """The directory that owns a database's sidecars.

    hook-errors.log and the transcripts/ archive tree live NEXT TO the database
    they describe, so moving the database moves its whole audit trail with it.
    """
    return default_db_path(db_override).parent


def log_error(directory, message) -> None:
    """Best-effort error trail. Must never raise."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "hook-errors.log").open("a", encoding="utf-8") as fh:
            fh.write("{0} {1}\n".format(utcnow(), message))
    except Exception:
        pass


def classifier_workdir() -> Path:
    """Disposable cwd for the headless classifier session.

    Deliberately a single STABLE directory rather than a fresh mkdtemp per
    call: Claude Code derives a transcript project directory from its cwd, so a
    new temp dir each time would leave one junk directory in ~/.claude/projects
    per classified turn. One fixed path means exactly one, ever -- while still
    keeping the classifier out of the audited workspace's own project dir.
    """
    path = Path(tempfile.gettempdir()) / "claude-audit-classifier"
    path.mkdir(parents=True, exist_ok=True)
    return path


def truncate(text, limit):
    text = text or ""
    return text if len(text) <= limit else text[:limit]


def is_synthetic_prompt(text) -> bool:
    """True when a user-role prompt was machine-generated, not typed.

    Interrupt notices, slash-command expansions and harness-injected context
    all reach the transcript as user content; tagging them as engineering
    intent would be a lie, so they are marked and left untagged.
    """
    blob = (text or "").strip()
    if not blob:
        return False
    if blob.startswith(SYNTHETIC_PREFIXES):
        return True
    return any(marker in blob for marker in SYNTHETIC_MARKERS)


def command_name(text):
    """The slash command inside a <command-name> tag, or None."""
    blob = text or ""
    start = blob.find("<command-name>")
    if start == -1:
        return None
    start += len("<command-name>")
    end = blob.find("</command-name>", start)
    name = (blob[start:end] if end != -1 else blob[start:]).strip()
    return name[:TITLE_MAX_CHARS] or None


def tag_body(text, tag):
    """The trimmed text between <tag> and </tag>, or None."""
    match = re.search(
        r"<{0}>(.*?)</{0}>".format(re.escape(tag)), text or "", re.DOTALL
    )
    body = match.group(1).strip() if match else ""
    return body or None


def first_match(patterns, text):
    """The first capture group any of `patterns` finds in `text`, or None."""
    blob = text if isinstance(text, str) else ""
    if not blob:
        return None
    for pattern in patterns:
        match = pattern.search(blob)
        if match:
            found = (match.group(1) or "").strip()
            if found:
                return found
    return None


def parse_task_link(tool_name, result_text):
    """(task_id, run_ref) for a call that spawned a background task.

    Returns (None, None) for every tool outside TASK_LINK_TOOLS -- the id only
    means anything when the call is the kind that spawns one, and narrowing by
    tool name is what keeps an unrelated result that happens to contain the
    words from inventing a link. The text parsed is the STORED result text
    (already capped at MAX_RESULT_CHARS), so capture and backfill always see
    exactly the same input and agree on the answer.
    """
    if tool_name not in TASK_LINK_TOOLS:
        return None, None
    return (
        first_match(TASK_ID_PATTERNS, result_text),
        first_match(RUN_REF_PATTERNS, result_text),
    )


def prompt_task_id(text):
    """The <task-id> of a background-task notification prompt, or None.

    Only <task-notification> prompts carry one, and that is the synthetic turn
    that closes a spawning call's loop (tool_calls.task_id == turns.task_id).
    """
    blob = text if isinstance(text, str) else ""
    if "<task-notification>" not in blob:
        return None
    body = tag_body(blob, "task-id")
    if not body:
        return None
    body = body.strip()
    return body if re.fullmatch(r"[A-Za-z0-9_-]+", body) else None


def synthetic_kind(text):
    """Deterministic (tag, title) for a machine-generated prompt. These are
    harness artifacts, so no model call is needed: the text itself says what
    kind of machinery produced it."""
    blob = text if isinstance(text, str) else ""
    if blob.startswith(SYNTHETIC_PREFIXES):
        return "system:interrupt", "Turn interrupted by user"
    if "<command-name>" in blob:
        name = command_name(blob)
        title = "Slash command: {0}".format(name) if name else "Slash command"
        return "system:command", truncate(title, TITLE_MAX_CHARS)
    if "<task-notification>" in blob:
        summary = tag_body(blob, "summary")
        title = summary or "Background task notification"
        return "system:task-result", truncate(title, TITLE_MAX_CHARS)
    return "system:context", "Harness-injected context"


def project_slug(workspace) -> str:
    """Claude Code's transcript directory name for a workspace path.

    Verified against ~/.claude/projects: /Users/mac/dev/claude-code-tools maps
    to -Users-mac-dev-claude-code-tools -- every character outside [A-Za-z0-9]
    (separators, dots, underscores alike) becomes a single dash.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(workspace).resolve()))


def path_under(path, parent) -> bool:
    """True when `path` is `parent` or lives beneath it. Never raises.

    Both sides are resolved first: on macOS tempfile.gettempdir() reports
    /var/folders/... while a resolved cwd reports /private/var/folders/...,
    and a purely textual comparison would miss every match.
    """
    try:
        path = Path(path).resolve()
        parent = Path(parent).resolve()
    except (OSError, ValueError):
        return False
    return path == parent or parent in path.parents


def projects_root(override=None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get("CLAUDE_AUDIT_PROJECTS_DIR")
    return Path(env).expanduser() if env else DEFAULT_PROJECTS_DIR


def first_sentence(text, limit=TITLE_MAX_CHARS):
    """First sentence of `text`, whitespace-collapsed and length-capped."""
    blob = " ".join((text or "").split())
    if not blob:
        return "(untitled turn)"
    for stop in (". ", "! ", "? "):
        idx = blob.find(stop)
        if idx != -1:
            blob = blob[:idx]
            break
    return blob[:limit]


# --------------------------------------------------------------------------
# Deterministic tool-call tagging
# --------------------------------------------------------------------------
#
# The whole pass is a pure function of (tool name, stored input payload), with
# no model call and no database access, for one reason: capture and backfill
# must agree. A row tagged live and the same row tagged months later by
# `sync` go through this identical code over the identical stored text -- the
# same discipline parse_task_link follows for background-task links -- so a
# re-run can never revise a verdict, and the backfill converges after one pass.


_KEYWORD_PATTERNS = {}


def keyword_pattern(keyword):
    """Cached 'this word starts here' matcher for one keyword."""
    pattern = _KEYWORD_PATTERNS.get(keyword)
    if pattern is None:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(keyword), re.IGNORECASE
        )
        _KEYWORD_PATTERNS[keyword] = pattern
    return pattern


def keyword_tag(text, table, at_word_start=False):
    """First tag in `table` whose vocabulary appears in `text`, else None.

    `at_word_start` is for command lines, where a bare substring test would
    have `ls` firing on "tools/" and `cat` on "concatenate"; prose
    descriptions are matched as plain substrings so morphology ("testing",
    "installing") still counts. A table entry may also be a compiled pattern,
    which is searched either way -- for the word whose plain substring would
    steal another group's match (see READ_VERB_RE).
    """
    blob = (text or "").lower()
    if not blob:
        return None
    for tag, keywords in table:
        for keyword in keywords:
            if hasattr(keyword, "search"):
                if keyword.search(blob):
                    return tag
            elif keyword_pattern(keyword).search(blob) if at_word_start else keyword in blob:
                return tag
    return None


def salvage_fields(text):
    """Top-level string fields recovered from a payload that was cut short.

    A single Write can carry a whole file, so input_json is capped at
    MAX_INPUT_CHARS and what is stored may not be parseable JSON at all. The
    short scalar fields this module reads (file_path, command, description,
    pattern, ...) are written before the bulky ones and survive the cut, so
    they are picked out textually rather than thrown away with the parse.
    """
    found = {}
    for key, raw in TRUNCATED_FIELD_RE.findall(text or ""):
        if key in found:
            continue
        try:
            value = json.loads(raw)
        except ValueError:
            continue
        if isinstance(value, str):
            found[key] = value
    return found


def tool_input_dict(input_json):
    """A tool call's stored input as a dict; {} when there is nothing usable."""
    if isinstance(input_json, dict):
        return input_json
    if not isinstance(input_json, str) or not input_json.strip():
        return {}
    try:
        parsed = json.loads(input_json)
    except ValueError:
        return salvage_fields(input_json)
    return parsed if isinstance(parsed, dict) else {}


def input_str(data, *keys):
    """The first of `keys` holding a non-blank string, trimmed; else None."""
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def spawn_description(data):
    """What an Agent / Workflow / Task call says it is going to do.

    A Workflow names itself in a meta block or a plain `name`; an Agent gets a
    `description`. The spawning prompt is the last resort -- it is the work
    itself rather than a label for it, but it is better than nothing.
    """
    meta = data.get("meta") if isinstance(data, dict) else None
    return (
        input_str(data, "description", "name", "task")
        or input_str(meta if isinstance(meta, dict) else {}, "description", "name", "title")
        or input_str(data, "prompt")
    )


def first_question(data):
    """The first question text of an AskUserQuestion call, or None."""
    questions = data.get("questions") if isinstance(data, dict) else None
    if isinstance(questions, list):
        for item in questions:
            if isinstance(item, str) and item.strip():
                return item.strip()
            found = input_str(item, "question", "header", "text")
            if found:
                return found
    return input_str(data, "question", "prompt")


def url_host(url):
    """The hostname of a URL, or None. Never raises."""
    if not url:
        return None
    try:
        parsed = urlsplit(url if "//" in url else "//" + url)
        return parsed.hostname or None
    except ValueError:
        return None


def tool_tag(tool_name, data):
    """The taxonomy tag for one tool call. First match wins."""
    name = tool_name or ""
    if name in TOOL_TAG_ASK:
        return "chat"
    if name in TOOL_TAG_EDIT:
        target = (input_str(data, *PATH_INPUT_KEYS) or "").lower()
        return "docs" if target.endswith(DOC_SUFFIXES) else "build"
    if name in TOOL_TAG_READ:
        return "plan"
    if name in TOOL_TAG_SPAWN:
        return keyword_tag(spawn_description(data), DESCRIPTION_TAG_KEYWORDS) or "build"
    if name == "Bash":
        text = "{0}\n{1}".format(
            input_str(data, "command") or "", input_str(data, "description") or ""
        )
        return keyword_tag(text, BASH_TAG_KEYWORDS, at_word_start=True) or "build"
    # Everything else -- MCP servers, harness tools, whatever ships next -- is
    # tooling acting on the environment rather than on the product.
    return "ops"


def tool_summary(tool_name, data):
    """One human line saying what this call did, from its input alone."""
    name = tool_name or ""
    if name == "Bash":
        # The description field is written for a human already; the command is
        # only a fallback for a call that did not describe itself.
        return (
            input_str(data, "description")
            or truncate(input_str(data, "command") or "", BASH_SUMMARY_CHARS)
            or "Called Bash"
        )
    if name in TOOL_SUMMARY_VERBS:
        target = input_str(data, *PATH_INPUT_KEYS)
        if target:
            return "{0} {1}".format(TOOL_SUMMARY_VERBS[name], Path(target).name or target)
    elif name in ("Grep", "Glob"):
        needle = input_str(data, "pattern", "query")
        if needle:
            return "Searched for '{0}'".format(needle)
    elif name == "WebFetch":
        host = url_host(input_str(data, "url"))
        if host:
            return "Fetched {0}".format(host)
    elif name == "WebSearch":
        query = input_str(data, "query")
        if query:
            return "Searched web for '{0}'".format(query)
    elif name == "Workflow":
        label = spawn_description(data)
        if label:
            return "Workflow: {0}".format(label)
    elif name in ("Agent", "Task"):
        label = spawn_description(data)
        if label:
            return "Agent: {0}".format(label)
    elif name in TOOL_TAG_ASK:
        question = first_question(data)
        if question:
            return "Asked the user: {0}".format(truncate(question, QUESTION_SUMMARY_CHARS))
    return "Called {0}".format(name or "tool")


def tool_tag_and_summary(tool_name, input_json):
    """(tag, summary) for one tool call, from its name and STORED input.

    Total: an input that is missing, truncated past recovery or shaped in some
    way this code has never seen still yields a tag and a sentence, because a
    row with no tag would be re-examined by every future sync.
    """
    try:
        data = tool_input_dict(input_json)
        tag = tool_tag(tool_name, data)
        summary = tool_summary(tool_name, data)
    except Exception:
        tag, summary = "ops", "Called {0}".format(tool_name or "tool")
    return tag, truncate(" ".join((summary or "").split()), SUMMARY_MAX_CHARS)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------


def apply_migrations(conn) -> None:
    """Bring an older database up to the current column set."""
    for statement in MIGRATIONS:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise
    relax_tag_check(conn)
    for statement in POST_MIGRATION_INDEXES:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass  # an index is an optimisation, never a correctness condition


def relax_tag_check(conn) -> None:
    """Rebuild the turns table while it still constrains tag with a CHECK.

    Fires for BOTH historical constraints -- the v1.0 one that allowed only
    plan/build/validate and the v1.2 one that also allowed system:% -- by
    testing for the `tag IN (` they share; the rebuilt table has no CHECK at
    all, so this converges after one pass and is a no-op thereafter. Runs
    after the ALTER migrations, so the old table is guaranteed to have every
    column the rebuild copies.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='turns'"
    ).fetchone()
    if row is None or "tag IN (" not in (row["sql"] or ""):
        return
    conn.execute(TURNS_REBUILD_DDL)
    conn.execute(
        "INSERT INTO turns_new ({0}) SELECT {0} FROM turns".format(TURN_COLUMN_LIST)
    )
    conn.execute("DROP TABLE turns")
    conn.execute("ALTER TABLE turns_new RENAME TO turns")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id)")
    conn.commit()


def open_db(path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(SCHEMA)  # idempotent on every open
    apply_migrations(conn)
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('created_at', ?)", (utcnow(),)
    )
    conn.commit()
    return conn


def upsert_session(conn, session_id, workspace, git_branch=None, cc_version=None):
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, workspace, started_at) VALUES (?, ?, ?)",
        (session_id, str(workspace), utcnow()),
    )
    # Branch/version only become known once transcript lines are parsed.
    conn.execute(
        "UPDATE sessions SET workspace = ?,"
        " git_branch = COALESCE(?, git_branch),"
        " cc_version = COALESCE(?, cc_version)"
        " WHERE session_id = ?",
        (str(workspace), git_branch, cc_version, session_id),
    )


def get_offset(conn, session_id) -> int:
    row = conn.execute(
        "SELECT byte_offset FROM cursors WHERE session_id = ?", (session_id,)
    ).fetchone()
    return int(row["byte_offset"]) if row else 0


def set_offset(conn, session_id, offset) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cursors (session_id, byte_offset, updated_at)"
        " VALUES (?, ?, ?)",
        (session_id, int(offset), utcnow()),
    )


def git_branch(cwd):
    """Fallback branch lookup for sessions whose transcript never named one."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            text=True,
        )
    except Exception:
        return None
    branch = proc.stdout.strip()
    return branch if proc.returncode == 0 and branch else None


# --------------------------------------------------------------------------
# Transcript reading
# --------------------------------------------------------------------------


def read_new_records(transcript_path, offset):
    """Read complete JSONL lines from `offset`.

    Returns (records, consumed_offset) where each record is
    {"obj": <parsed dict>, "offset": <byte offset of that line's first byte>}.
    A partially written trailing line is left for the next sweep. If the file
    is shorter than the cursor (rewritten or rotated) we restart from zero.
    """
    path = Path(transcript_path) if transcript_path else None
    if path is None or not path.is_file():
        return [], offset

    try:
        size = path.stat().st_size
    except OSError:
        return [], offset
    if size < offset:
        offset = 0

    with path.open("rb") as fh:
        fh.seek(offset)
        buf = fh.read()

    records = []
    pos = 0
    consumed = offset
    while True:
        newline = buf.find(b"\n", pos)
        if newline == -1:
            break
        line_start = offset + pos
        raw = buf[pos:newline]
        pos = newline + 1
        consumed = offset + pos
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped.decode("utf-8", "replace"))
        except ValueError:
            continue  # defensive: the format is internal and may change
        if isinstance(obj, dict):
            records.append({"obj": obj, "offset": line_start})
    return records, consumed


def iter_transcript_objects(path):
    """Stream one transcript's parsed lines. Cursor-free, one line in memory."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def iter_transcript_records(path):
    """Stream a whole transcript in read_new_records' record shape.

    Same {"obj", "offset"} records assemble_turns consumes, but from byte zero
    and without ever consulting or writing the cursors table -- the repair pass
    re-reads a session in full without disturbing the incremental sync.
    """
    offset = 0
    with Path(path).open("rb") as fh:
        for raw in fh:
            start = offset
            offset += len(raw)
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped.decode("utf-8", "replace"))
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield {"obj": obj, "offset": start}


def flatten_text(value) -> str:
    """Transcript text fields may be a string, or a list of content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    if value is None:
        return ""
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def is_real_prompt(obj) -> bool:
    """True for a genuine user prompt line.

    Lines of type 'user' are also used to carry tool_result blocks back into
    the conversation; those continue the current turn rather than starting one.
    """
    if obj.get("type") != "user" or obj.get("isMeta") or obj.get("isSidechain"):
        return False
    content = (obj.get("message") or {}).get("content")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        has_text = False
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                return False
            if block.get("type") == "text":
                has_text = True
        return has_text
    return False


def queued_interjection(obj):
    """(kind, text) when this line is a message injected into a RUNNING turn.

    Only the `queued_command` attachment shape reaches here; a mid-flight
    `user` line is judged by the caller, which is the only place that knows
    whether tool results are outstanding. See INTERJECTION_HEADER for why the
    attachment needs no such test: the harness writes it after the tool result
    it rides along with, so nothing is pending by then, yet it is by
    construction a message the human sent while the assistant was working.
    """
    if obj.get("type") != "attachment":
        return None
    attachment = obj.get("attachment")
    if not isinstance(attachment, dict) or attachment.get("type") != "queued_command":
        return None
    text = attachment.get("prompt")
    if attachment.get("commandMode") == "task-notification":
        return "task", tag_body(text, "summary")
    return "prompt", text


def note_flight(pending, obj) -> None:
    """Track which tool_use ids are still waiting for their tool_result.

    A non-empty list means the conversation is MID-FLIGHT: the assistant asked
    for something and has not been answered yet, so nothing arriving now can
    be the start of a new turn.

    Kept in issue order, and a result RETIRES EVERY CALL OLDER THAN ITSELF as
    well as its own. Tools are answered in the order they were asked, so an
    older id still outstanding when a younger one comes back was abandoned --
    an interrupted turn, a transcript truncated mid-write. Without this a
    single unanswered call would leave the conversation permanently
    "mid-flight" and swallow every later prompt into one enormous turn.
    """
    message = obj.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    if obj.get("type") == "assistant":
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("id") and block["id"] not in pending:
                pending.append(block["id"])
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        tool_use_id = block.get("tool_use_id")
        if tool_use_id in pending:
            del pending[: pending.index(tool_use_id) + 1]


def classify_line(obj, pending, has_turn):
    """What one transcript line means for turn assembly.

    Returns (verdict, payload):
      ("skip", None)                subagent traffic -- its own file, own chain
      ("start", None)               a real prompt that OPENS a new turn
      ("interject", (kind, text))   a mid-turn message folded into the open turn
      ("line", None)                anything else

    `pending` is the mid-flight tool_use list and is UPDATED here, so every
    caller -- the assembler and the lighter outline walk the repair pass uses
    -- draws turn boundaries in exactly the same places.

    THE BOUNDARY RULE: a real prompt line opens a turn only when the
    conversation is not mid-flight. Machinery lines are exempt: an interrupt
    notice, a slash-command expansion and harness-injected context each have
    their own handling and stay their own turn whatever is in flight. A
    <task-notification> is the one machinery line that IS folded in when it
    arrives mid-flight, and then only as a marker (see apply_interjection).
    `pending` is cleared whenever a turn opens, so one tool call that never
    saw a result -- an interrupted or truncated turn -- cannot swallow the
    rest of the session.
    """
    if obj.get("isSidechain"):
        return "skip", None

    injected = queued_interjection(obj)
    if injected is not None:
        return ("interject", injected) if has_turn else ("line", None)

    if is_real_prompt(obj):
        text = flatten_text((obj.get("message") or {}).get("content"))
        task_note = "<task-notification>" in (text or "")
        folds = task_note or not is_synthetic_prompt(text)
        if has_turn and pending and folds:
            if task_note:
                return "interject", ("task", tag_body(text, "summary"))
            return "interject", ("prompt", text)
        pending.clear()
        return "start", None

    note_flight(pending, obj)
    return "line", None


def interjection_addition(kind, text):
    """The block a mid-turn message contributes to its turn's prompt, or None.

    One function so the assembler and the repair pass's staleness test agree
    to the character on what a folded-in message looks like.
    """
    blob = (text or "").strip()
    if kind == "task":
        # Deliberately NOT the notification's own text: it is bulky and the
        # whole of it already sits in the spawning tool call's result.
        return TASK_DONE_HEADER.format(
            truncate(blob, TITLE_MAX_CHARS) or "background task"
        )
    if not blob:
        return None
    return "{0}\n{1}".format(INTERJECTION_HEADER, blob)


def apply_interjection(turn, kind, text) -> None:
    """Fold a mid-turn message into the turn that was already running.

    The enclosing turn KEEPS ITS IDENTITY -- prompt_uuid, parent_uuid and the
    synthetic verdict derived from its original prompt are untouched, and
    last_uuid still ends up naming the turn's true final line -- so the
    lineage graph is unchanged and nothing is re-parented. Only the prompt
    text grows, by a marked block, and the tally goes up by one.
    """
    addition = interjection_addition(kind, text)
    if addition is None:
        return
    if kind != "task":
        turn["interjection_texts"].append((text or "").strip())
    base = turn["prompt"] or ""
    turn["prompt"] = "{0}\n\n{1}".format(base, addition) if base.strip() else addition
    turn["interjections"] += 1


def new_turn(obj, byte_offset):
    timestamp = obj.get("timestamp")
    prompt = flatten_text((obj.get("message") or {}).get("content"))
    return {
        "byte_offset": byte_offset,
        "prompt_uuid": obj.get("uuid"),
        # Lineage: what this prompt was a reply to, and (updated by absorb as
        # the turn plays out) the uuid of its own last line. See the MIGRATIONS
        # comment for how the viewer walks these into a graph.
        "parent_uuid": obj.get("parentUuid"),
        "last_uuid": obj.get("uuid"),
        "prompt_id": obj.get("promptId"),
        "ts_start": timestamp,
        "ts_end": timestamp,
        "prompt": prompt,
        "is_synthetic": 1 if is_synthetic_prompt(prompt) else 0,
        # Set only on a <task-notification> turn: the background task this
        # turn is the completion notice for (see the MIGRATIONS comment).
        "task_id": prompt_task_id(prompt),
        "git_branch": obj.get("gitBranch"),
        "cc_version": obj.get("version"),
        "model": None,
        # Mid-turn messages folded in by apply_interjection: the tally that
        # reaches turns.interjections, and the raw texts, which persist_turns
        # uses to recognise (and clear) the hook rows those messages opened.
        "interjections": 0,
        "interjection_texts": [],
        "texts": [],
        "tools": [],  # ordered tool_use entries
        "tools_by_id": {},
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "thinking_tokens": 0,
        },
        "usage_seen": set(),
    }


def add_usage(turn, obj, message) -> None:
    """Accumulate token usage, counting each API response exactly once.

    Claude Code writes one transcript line per content block, and every line
    belonging to the same API response repeats that response's *identical*
    usage object. Summing blindly over assistant lines inflates the totals
    (observed ~2.5x on a real transcript), so responses are de-duplicated by
    message id, falling back to the line's requestId.
    """
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return
    key = message.get("id") or obj.get("requestId")
    if key is not None:
        if key in turn["usage_seen"]:
            return
        turn["usage_seen"].add(key)

    for column, source in TOKEN_COLUMNS:
        try:
            turn["tokens"][column] += int(usage.get(source) or 0)
        except (TypeError, ValueError):
            pass
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        try:
            turn["tokens"]["thinking_tokens"] += int(details.get("thinking_tokens") or 0)
        except (TypeError, ValueError):
            pass


def record_tool_use(turn, block) -> None:
    tool_use_id = block.get("id")
    if tool_use_id and tool_use_id in turn["tools_by_id"]:
        return  # already captured (re-read of the same line)
    try:
        input_json = json.dumps(block.get("input"), default=str)
    except (TypeError, ValueError):
        input_json = None
    # A single Write/Edit can carry a whole file; cap it like tool results so
    # one payload cannot dominate the database.
    input_truncated = 0
    if input_json is not None and len(input_json) > MAX_INPUT_CHARS:
        input_json = input_json[:MAX_INPUT_CHARS]
        input_truncated = 1
    # Tagged from the input exactly as STORED (truncation and all), so the
    # sync's backfill over this same text agrees with capture and never
    # rewrites the row. See tool_tag_and_summary.
    tag, summary = tool_tag_and_summary(block.get("name"), input_json)
    entry = {
        "tool_use_id": tool_use_id,
        "tool_name": block.get("name"),
        "input_json": input_json,
        "input_truncated": input_truncated,
        "tag": tag,
        "summary": summary,
        "result_text": None,
        "result_truncated": 0,
        "is_error": 0,
        # Filled by attach_tool_result, which is where the result text (and
        # with it any background-task id) arrives.
        "task_id": None,
        "run_ref": None,
    }
    turn["tools"].append(entry)
    if tool_use_id:
        turn["tools_by_id"][tool_use_id] = entry


def attach_tool_result(turn, block) -> None:
    entry = turn["tools_by_id"].get(block.get("tool_use_id"))
    if entry is None:
        return  # result for a tool_use we never saw (cursor started mid-turn)
    text = flatten_text(block.get("content"))
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS]
        entry["result_truncated"] = 1
    entry["result_text"] = text
    entry["is_error"] = 1 if block.get("is_error") else 0
    # Parsed from the text as STORED (already truncated above), so the sync's
    # backfill pass over the database reaches the same verdict.
    entry["task_id"], entry["run_ref"] = parse_task_link(entry["tool_name"], text)


def absorb(turn, obj) -> None:
    """Fold one non-prompt line into the turn currently being assembled."""
    timestamp = obj.get("timestamp")
    if isinstance(timestamp, str) and timestamp > (turn["ts_end"] or ""):
        turn["ts_end"] = timestamp
    if obj.get("gitBranch"):
        turn["git_branch"] = obj["gitBranch"]
    if obj.get("version"):
        turn["cc_version"] = obj["version"]

    message = obj.get("message")
    if not isinstance(message, dict):
        return
    content = message.get("content")

    if obj.get("type") == "assistant":
        if message.get("model"):
            turn["model"] = message["model"]
        add_usage(turn, obj, message)
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text" and block.get("text"):
                    turn["texts"].append(block["text"])
                elif kind == "tool_use":
                    record_tool_use(turn, block)
    elif isinstance(content, list):  # user line carrying tool results
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                attach_tool_result(turn, block)


def chain_tip(turn, obj) -> None:
    """Advance a turn's lineage tip to this line, when it has a uuid.

    Deliberately wider than absorb(): the transcript's parent chain runs
    through line types this audit stores nothing else from -- `system`
    notices, `attachment` blocks, meta `user` lines -- and measured against a
    real transcript the NEXT prompt's parentUuid points at one of THOSE far
    more often than at an assistant line. Tracking only user/assistant lines
    would leave last_uuid naming a line nothing ever chains onto, and every
    turn would look like an orphan root. Types that carry no uuid at all
    (custom-title, queue-operation, file-history-*, ...) are outside the chain
    and correctly leave the tip alone.
    """
    if obj.get("uuid"):
        turn["last_uuid"] = obj["uuid"]


def assemble_turns(records):
    """Group transcript records into turns, one per genuine turn boundary.

    Boundaries are classify_line's: a real prompt starts a turn only when the
    conversation is not mid-flight, and a message the human sent while the
    assistant was working is folded into the turn that was running instead of
    opening one of its own.
    """
    turns = []
    current = None
    pending = []
    for record in records:
        obj = record["obj"]
        verdict, payload = classify_line(obj, pending, current is not None)
        if verdict == "skip":
            continue
        if verdict == "start":
            current = new_turn(obj, record["offset"])
            turns.append(current)
            continue
        if current is None:
            continue  # lines before the first prompt close an earlier turn
        if verdict == "interject":
            apply_interjection(current, payload[0], payload[1])
            chain_tip(current, obj)
            continue
        chain_tip(current, obj)
        if obj.get("isMeta") or obj.get("type") not in ("user", "assistant"):
            continue  # lineage only; nothing here is turn content
        absorb(current, obj)
    return turns


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

TURN_FIELDS = (
    "prompt_uuid",
    "parent_uuid",
    "last_uuid",
    "ts_start",
    "ts_end",
    "duration_s",
    "model",
    "prompt",
    "response",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "thinking_tokens",
    "status",
    "interjections",
)


def find_turn_row(conn, session_id, turn, claimed):
    """Locate the row for this turn: by prompt_uuid, then by pending prompt_id,
    then the oldest unclaimed pending row for the session."""
    if turn["prompt_uuid"]:
        row = conn.execute(
            "SELECT turn_id, status, response, is_synthetic, tag FROM turns"
            " WHERE prompt_uuid = ?",
            (turn["prompt_uuid"],),
        ).fetchone()
        if row is not None:
            return row

    pending_sql = (
        "SELECT turn_id, status, response, is_synthetic, tag FROM turns WHERE session_id = ?"
        " AND status = 'pending' AND prompt_uuid IS NULL{0} ORDER BY turn_id ASC"
    )
    attempts = []
    if turn["prompt_id"]:
        attempts.append((pending_sql.format(" AND prompt_id = ?"),
                         (session_id, turn["prompt_id"])))
    # Last resort: the oldest pending row this sweep has not already used.
    attempts.append((pending_sql.format(""), (session_id,)))

    for sql, params in attempts:
        for row in conn.execute(sql, params).fetchall():
            if row["turn_id"] not in claimed:
                return row
    return None


def persist_turn(conn, session_id, turn, final_status, claimed):
    """Write one assembled turn.

    Returns (turn_id, newly_completed, created, reclassify, retagged).
    `created` distinguishes a freshly inserted row from one the live hooks had
    already opened; `reclassify` says the row's response text changed and it
    therefore needs (re)classifying; `retagged` is the narrower case worth
    REPORTING -- the row already carried a tag, so that tag has just been
    thrown away. An ordinary pending -> complete first fill sets `reclassify`
    but not `retagged`, and must not be counted as a re-opened turn.
    """
    values = dict(turn["tokens"])
    values["prompt_uuid"] = turn["prompt_uuid"]
    values["parent_uuid"] = turn["parent_uuid"]
    values["last_uuid"] = turn["last_uuid"]
    values["ts_start"] = turn["ts_start"]
    values["ts_end"] = turn["ts_end"]
    values["duration_s"] = duration_seconds(turn["ts_start"], turn["ts_end"])
    values["model"] = turn["model"]
    values["prompt"] = turn["prompt"]
    values["response"] = "\n\n".join(turn["texts"])
    values["interjections"] = turn["interjections"]

    row = find_turn_row(conn, session_id, turn, claimed)
    was_complete = bool(row is not None and row["status"] == "complete")
    # A turn already finalised by Stop stays 'complete' when SessionEnd re-sweeps.
    values["status"] = "complete" if was_complete else final_status

    created = row is None
    reclassify = False
    retagged = False
    if row is not None:
        turn_id = row["turn_id"]
        claimed.add(turn_id)
        # A turn whose response text has grown or changed since it was last
        # written was tagged from incomplete data -- typically the Stop hook
        # firing before the final text blocks landed, which left the newest
        # turn wearing a title describing half of it. Dropping tag/title/
        # tag_source here re-opens it: the Stop handler (and `sync --classify`,
        # whose query is `tag IS NULL`) picks it up and classifies it again
        # from the whole thing. Synthetic turns are exempt -- their kind is
        # derived from the prompt, which never changes.
        new_response = values["response"] or ""
        old_response = row["response"] or ""
        reclassify = bool(
            new_response.strip()
            and new_response != old_response
            and not (row["is_synthetic"] or turn["is_synthetic"])
        )
        # Only a row that ALREADY had a tag has actually been re-opened. The
        # common case here is a pending row (opened by UserPromptSubmit, never
        # classified) receiving its response for the first time: it needs
        # classifying, but reporting it as "re-tagged after growing" would
        # count every ordinary turn in the sweep.
        retagged = reclassify and row["tag"] is not None
        # `prompt` is assigned separately so the fallback below is unambiguous.
        updated = [name for name in TURN_FIELDS if name != "prompt"]
        assignments = ", ".join("{0} = ?".format(name) for name in updated)
        params = [values[name] for name in updated]
        # Transcript text wins, but a blank one keeps the submitted prompt --
        # and with it the synthetic verdict derived from that same text.
        transcript_prompt = values["prompt"] if values["prompt"].strip() else None
        synthetic = turn["is_synthetic"] if transcript_prompt is not None else None
        reset = ", tag = NULL, title = NULL, tag_source = NULL" if reclassify else ""
        # task_id follows the prompt text exactly as is_synthetic does: both
        # are DERIVED from it, so the transcript's verdict -- including "this
        # prompt names no task" -- replaces whatever was there, and a sweep
        # that saw no prompt text says nothing and leaves the column alone.
        # (It cannot ride along in COALESCE: find_turn_row's last-resort match
        # can hand this turn a pending row opened for a different prompt, and
        # that row's task id must not survive onto it.)
        task = ", task_id = ?" if transcript_prompt is not None else ""
        task_params = [turn["task_id"]] if transcript_prompt is not None else []
        conn.execute(
            "UPDATE turns SET {0}, prompt_id = COALESCE(?, prompt_id),"
            " prompt = COALESCE(?, prompt),"
            " is_synthetic = COALESCE(?, is_synthetic){1}{2} WHERE turn_id = ?".format(
                assignments, task, reset
            ),
            params
            + [turn["prompt_id"], transcript_prompt, synthetic]
            + task_params
            + [turn_id],
        )
    else:
        columns = [
            "session_id",
            "prompt_id",
            "permission_mode",
            "is_synthetic",
            "task_id",
        ] + list(TURN_FIELDS)
        params = [
            session_id,
            turn["prompt_id"],
            None,
            turn["is_synthetic"],
            turn["task_id"],
        ] + [values[name] for name in TURN_FIELDS]
        cursor = conn.execute(
            "INSERT OR IGNORE INTO turns ({0}) VALUES ({1})".format(
                ", ".join(columns), ", ".join("?" * len(columns))
            ),
            params,
        )
        turn_id = cursor.lastrowid
        if not cursor.rowcount and turn["prompt_uuid"]:
            existing = conn.execute(
                "SELECT turn_id FROM turns WHERE prompt_uuid = ?",
                (turn["prompt_uuid"],),
            ).fetchone()
            turn_id = existing["turn_id"] if existing else None
            created = False
        if turn_id is not None:
            claimed.add(turn_id)

    if turn_id is None:
        return None, False, False, False, False

    # Tool calls are rebuilt wholesale so re-processing stays idempotent.
    conn.execute("DELETE FROM tool_calls WHERE turn_id = ?", (turn_id,))
    conn.executemany(
        "INSERT INTO tool_calls (turn_id, seq, tool_name, tool_use_id, input_json,"
        " input_truncated, result_text, result_truncated, is_error, task_id,"
        " run_ref, tag, summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                turn_id,
                seq,
                entry["tool_name"],
                entry["tool_use_id"],
                entry["input_json"],
                entry["input_truncated"],
                entry["result_text"],
                entry["result_truncated"],
                entry["is_error"],
                entry.get("task_id"),
                entry.get("run_ref"),
                entry.get("tag"),
                entry.get("summary"),
            )
            for seq, entry in enumerate(turn["tools"])
        ],
    )
    return turn_id, not was_complete, created, reclassify, retagged


# --------------------------------------------------------------------------
# Subagents
# --------------------------------------------------------------------------
#
# A session transcript at <projects>/<slug>/<session_id>.jsonl has a sibling
# DIRECTORY <projects>/<slug>/<session_id>/subagents/ holding one JSONL per
# subagent, in the same line format as the main transcript but flagged
# isSidechain. Directly spawned agents sit at the top level next to a
# <name>.meta.json sidecar; workflow steps sit under workflows/<run_id>/.
#
# The join key back to the main transcript is promptId: every line of an
# agent's file carries the PARENT TURN's prompt id. The format is internal and
# undocumented, so every field here is read defensively -- any key may be
# missing, and a shape that surprises us costs at most one skipped agent.

AGENT_COLUMNS = (
    "session_id",
    "agent_id",
    "turn_id",
    "prompt_id",
    "tool_use_id",
    "workflow_id",
    "agent_type",
    "description",
    "model",
    "prompt",
    "result",
    "ts_start",
    "ts_end",
    "duration_s",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "thinking_tokens",
    "tool_call_count",
    "src_bytes",
    # Classified exactly as a turn is (classify_agent). store_agent carries an
    # existing classification forward across a re-parse that changed nothing;
    # the merge copies them like any other column.
    "tag",
    "title",
    "tag_source",
)

# The columns of one agent_tool_calls row, minus its autoincrement atc_id. The
# key columns lead, because an agent's rows are always written as a set for one
# (session_id, agent_id).
AGENT_TOOL_CALL_COLUMNS = (
    "session_id",
    "agent_id",
    "seq",
    "tool_name",
    "tool_use_id",
    "input_json",
    "input_truncated",
    "result_text",
    "result_truncated",
    "is_error",
    "tag",
    "summary",
)

# What a scan can report about one agent file. 'backfilled' is an otherwise
# UNCHANGED agent that was re-parsed anyway, because its tool calls predate the
# agent_tool_calls table and would never arrive on their own (see store_agent).
AGENT_COUNT_KEYS = ("new", "updated", "unchanged", "backfilled")


def agent_counts_zero():
    return {key: 0 for key in AGENT_COUNT_KEYS}


def subagents_dir(transcript_path, session_id):
    """The directory of subagent transcripts beside a session transcript."""
    path = Path(transcript_path) if transcript_path else None
    if path is None:
        return None
    name = str(session_id or path.stem or "")
    if not name:
        return None
    directory = path.parent / name / "subagents"
    return directory if directory.is_dir() else None


def agent_files(directory):
    """Every agent JSONL under the subagents tree, as (path, workflow_id).

    Walked rather than globbed at two fixed depths so an unexpected layout
    still yields its agents. `workflow_id` is the run directory under
    workflows/, or None for a directly spawned agent. The *.meta.json sidecars
    and a workflow run's journal.jsonl do not match the pattern.
    """
    found = []
    for path in sorted(directory.rglob("agent-*.jsonl")):
        if not path.is_file():
            continue
        try:
            parts = path.relative_to(directory).parts
        except ValueError:
            continue
        workflow = parts[1] if len(parts) >= 3 and parts[0] == "workflows" else None
        found.append((path, workflow))
    return found


def read_agent_meta(path):
    """The agent's meta.json sidecar; {} when absent or unreadable.

    Direct agents carry agentType/description/toolUseId/spawnDepth; workflow
    agents may carry a different set entirely, so nothing is required.
    """
    meta_path = path.parent / "{0}.meta.json".format(path.stem)
    try:
        with meta_path.open("r", encoding="utf-8", errors="replace") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def assistant_text(obj) -> str:
    """The concatenated text blocks of one assistant line."""
    content = (obj.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    return "\n".join(parts)


def agent_result_text(assistants) -> str:
    """What the subagent handed back to its parent.

    Normally the text of its last speaking turn. An agent given a structured
    output schema instead ends on a StructuredOutput tool call whose input IS
    the result, so that payload is serialised rather than losing it.
    """
    if not assistants:
        return ""
    content = (assistants[-1].get("message") or {}).get("content")
    blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
    if blocks:
        last = blocks[-1]
        if (
            last.get("type") == "tool_use"
            and last.get("name") == STRUCTURED_OUTPUT_TOOL
        ):
            try:
                return json.dumps(last.get("input"), default=str)
            except (TypeError, ValueError):
                return str(last.get("input"))
    for obj in reversed(assistants):
        text = assistant_text(obj)
        if text:
            return text
    return ""


def parse_agent_file(path):
    """Fold one agent JSONL into the fields the agents table stores.

    Also returns the agent's OWN tool calls under "tool_calls", in call order
    and with each result matched back to its tool_use -- the subagent's
    internal flow, which agents.tool_call_count could only count.
    """
    record = {
        "agent_id": None,
        "prompt_id": None,
        "model": None,
        "prompt": "",
        "result": "",
        "ts_start": None,
        "ts_end": None,
        "tool_call_count": 0,
    }
    # add_usage() only ever touches these two keys, and record_tool_use /
    # attach_tool_result only these two more, so the turn accumulator is reused
    # verbatim -- including its de-duplication of a response's usage object
    # across the several lines that repeat it, and the same input/result
    # truncation limits a main-thread tool call is stored under.
    acc = {
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "thinking_tokens": 0,
        },
        "usage_seen": set(),
        "tools": [],
        "tools_by_id": {},
    }
    assistants = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue

            timestamp = obj.get("timestamp")
            if isinstance(timestamp, str) and timestamp:
                if record["ts_start"] is None or timestamp < record["ts_start"]:
                    record["ts_start"] = timestamp
                if record["ts_end"] is None or timestamp > record["ts_end"]:
                    record["ts_end"] = timestamp
            if not record["prompt_id"] and obj.get("promptId"):
                record["prompt_id"] = obj["promptId"]
            if not record["agent_id"] and obj.get("agentId"):
                record["agent_id"] = obj["agentId"]

            message = obj.get("message")
            if not isinstance(message, dict):
                continue
            kind = obj.get("type")
            content = message.get("content")
            if kind == "user":
                # The first user line is the spawning instruction; later ones
                # carry tool results, which flatten to nothing.
                if not record["prompt"]:
                    record["prompt"] = flatten_text(content)
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "tool_result"
                        ):
                            attach_tool_result(acc, block)
            elif kind == "assistant":
                assistants.append(obj)
                if message.get("model"):
                    record["model"] = message["model"]
                add_usage(acc, obj, message)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            record["tool_call_count"] += 1
                            # Nothing is excluded: a StructuredOutput call is
                            # how a schema-bound agent returns its answer, and
                            # it is as much a step of the flow as any other.
                            record_tool_use(acc, block)

    record["result"] = agent_result_text(assistants)
    record["tokens"] = acc["tokens"]
    record["tool_calls"] = acc["tools"]
    return record


def lookup_turn_id(conn, session_id, prompt_id):
    """The parent turn for an agent's promptId, or None if not stored yet."""
    if not prompt_id:
        return None
    row = conn.execute(
        "SELECT turn_id FROM turns WHERE session_id = ? AND prompt_id = ?"
        " ORDER BY turn_id ASC LIMIT 1",
        (session_id, prompt_id),
    ).fetchone()
    return row["turn_id"] if row else None


def needs_tool_backfill(conn, session_id, agent_id, row) -> bool:
    """True when an unchanged agent must be re-parsed for its tool calls.

    agent_tool_calls arrived after the agents table, so every agent already
    mirrored is size-identical to its source and would be skipped forever --
    its flow would never be recorded. An agent that CALLED tools (per the
    tally it does have) but holds no agent_tool_calls rows is therefore
    re-parsed once; after that pass it has rows and is skipped again. The
    source file's existence is not re-tested here because the caller has
    already stat()ed it to get src_bytes.
    """
    if not (row["tool_call_count"] or 0) > 0:
        return False
    return conn.execute(
        "SELECT 1 FROM agent_tool_calls WHERE session_id = ? AND agent_id = ? LIMIT 1",
        (session_id, agent_id),
    ).fetchone() is None


def agent_id_for(path) -> str:
    """The agents.agent_id an agent transcript's filename carries."""
    return path.stem[len("agent-"):] or path.stem


def carry_classification(conn, session_id, agent_id, values) -> None:
    """Keep an agent's existing tag/title across a re-parse that changed it.

    The turn side drops a classification when the response text grows, and an
    agent is treated the same way: if the prompt and the result this parse
    produced are byte-identical to what the row already holds, the tag still
    describes the agent and survives INSERT OR REPLACE; if either changed --
    or the row was never classified -- the columns are left NULL and the
    classifier picks the agent up again (`tag IS NULL` is the sync's query,
    and scan_agents spawns one live).
    """
    row = conn.execute(
        "SELECT tag, title, tag_source, prompt, result FROM agents"
        " WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    ).fetchone()
    if row is None or row["tag"] is None:
        return
    if (row["prompt"] or "") != (values.get("prompt") or ""):
        return
    if (row["result"] or "") != (values.get("result") or ""):
        return
    values["tag"] = row["tag"]
    values["title"] = row["title"]
    values["tag_source"] = row["tag_source"]


def agent_needs_tag(conn, session_id, agent_id) -> bool:
    """True when this agent row exists and is still unclassified."""
    row = conn.execute(
        "SELECT 1 FROM agents WHERE session_id = ? AND agent_id = ? AND tag IS NULL",
        (session_id, agent_id),
    ).fetchone()
    return row is not None


def store_agent(conn, session_id, path, workflow_id) -> str:
    """Persist one agent file.

    Returns 'new', 'updated', 'unchanged' or 'backfilled' (unchanged source,
    re-parsed only to give an older row its tool calls).
    """
    src_bytes = path.stat().st_size
    agent_id = agent_id_for(path)

    row = conn.execute(
        "SELECT src_bytes, tool_call_count FROM agents"
        " WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    ).fetchone()
    # An agent transcript is append-only and finished agents never grow, so an
    # unchanged size means unchanged content: skip the parse entirely -- unless
    # the row predates agent_tool_calls and its flow is still missing.
    backfill = False
    if row is not None and row["src_bytes"] == src_bytes:
        backfill = needs_tool_backfill(conn, session_id, agent_id, row)
        if not backfill:
            return "unchanged"

    record = parse_agent_file(path)
    meta = read_agent_meta(path)

    values = dict(record["tokens"])
    values["session_id"] = session_id
    values["agent_id"] = agent_id
    values["prompt_id"] = record["prompt_id"]
    values["turn_id"] = lookup_turn_id(conn, session_id, record["prompt_id"])
    values["tool_use_id"] = meta.get("toolUseId")
    values["workflow_id"] = workflow_id
    values["agent_type"] = meta.get("agentType")
    values["description"] = meta.get("description") or meta.get("label")
    values["model"] = record["model"] or meta.get("model")
    values["prompt"] = truncate(record["prompt"], MAX_AGENT_TEXT_CHARS)
    values["result"] = truncate(record["result"], MAX_AGENT_TEXT_CHARS)
    values["ts_start"] = record["ts_start"]
    values["ts_end"] = record["ts_end"]
    values["duration_s"] = duration_seconds(record["ts_start"], record["ts_end"])
    values["tool_call_count"] = record["tool_call_count"]
    values["src_bytes"] = src_bytes
    # Before the write, because INSERT OR REPLACE below would otherwise blank
    # a classification this parse did not change anything to invalidate.
    carry_classification(conn, session_id, agent_id, values)

    conn.execute(
        "INSERT OR REPLACE INTO agents ({0}) VALUES ({1})".format(
            ", ".join(AGENT_COLUMNS), ", ".join("?" * len(AGENT_COLUMNS))
        ),
        [values.get(name) for name in AGENT_COLUMNS],
    )
    write_agent_tool_calls(conn, session_id, agent_id, record["tool_calls"])
    if backfill:
        return "backfilled"
    return "updated" if row is not None else "new"


def write_agent_tool_calls(conn, session_id, agent_id, entries) -> int:
    """Rebuild one agent's tool calls wholesale, so re-parsing is idempotent.

    Same contract as the turn-side tool_calls rewrite: the rows have no
    identity of their own, they belong to the agent, and the agent is the unit
    that gets re-read. Returns the number of rows written.
    """
    conn.execute(
        "DELETE FROM agent_tool_calls WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    )
    conn.executemany(
        "INSERT INTO agent_tool_calls (session_id, agent_id, seq, tool_name,"
        " tool_use_id, input_json, input_truncated, result_text,"
        " result_truncated, is_error, tag, summary)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                session_id,
                agent_id,
                seq,
                entry["tool_name"],
                entry["tool_use_id"],
                entry["input_json"],
                entry["input_truncated"],
                entry["result_text"],
                entry["result_truncated"],
                entry["is_error"],
                entry.get("tag"),
                entry.get("summary"),
            )
            for seq, entry in enumerate(entries or [])
        ],
    )
    return len(entries or [])


def scan_agents(conn, session_id, transcript_path, directory, pending=None):
    """Mirror a session's subagent transcripts into the agents table.

    `directory` is the audit directory owning the open database -- errors are
    logged there rather than in the audited workspace, which under the
    machine-wide model is not where this database lives.

    `pending`, when given, collects (session_id, agent_id) for every agent
    this scan wrote that is still unclassified, so the caller can spawn a
    classifier for it exactly as the Stop hook does for a finished turn.

    Returns a count per AGENT_COUNT_KEYS. A file that cannot be read or parsed
    is logged and skipped -- one malformed agent never costs the rest of the
    sweep.
    """
    counts = agent_counts_zero()
    if not session_id:
        return counts
    try:
        root = subagents_dir(transcript_path, session_id)
        files = agent_files(root) if root is not None else []
    except OSError:
        return counts

    for path, workflow_id in files:
        try:
            status = store_agent(conn, session_id, path, workflow_id)
            counts[status] += 1
            if pending is not None and status != "unchanged":
                agent_id = agent_id_for(path)
                if agent_needs_tag(conn, session_id, agent_id):
                    pending.append((session_id, agent_id))
        except Exception:
            log_error(
                directory,
                "agent scan failed for {0}: {1}".format(
                    path, traceback.format_exc().replace("\n", " | ")
                ),
            )
    return counts


def resolve_agent_turns(conn, session_id=None) -> int:
    """Attach still-orphaned agent rows to their parent turn.

    An agent can be written before the turn that spawned it is finalised (and
    a sync may meet the two in either order), so the join is retried here
    after every scan. One indexed UPDATE, no per-row work.
    """
    sql = (
        "UPDATE agents SET turn_id = (SELECT t.turn_id FROM turns t"
        " WHERE t.session_id = agents.session_id AND t.prompt_id = agents.prompt_id"
        " ORDER BY t.turn_id ASC LIMIT 1)"
        " WHERE turn_id IS NULL AND prompt_id IS NOT NULL"
        " AND EXISTS (SELECT 1 FROM turns t WHERE t.session_id = agents.session_id"
        " AND t.prompt_id = agents.prompt_id)"
    )
    params = []
    if session_id:
        sql += " AND session_id = ?"
        params.append(session_id)
    return conn.execute(sql, params).rowcount


def sweep_agents(payload, db_file):
    """Hook-side subagent capture. Never raises.

    Returns (counts, pending) where `pending` is the (session_id, agent_id)
    of every agent stored here that still has no tag.
    """
    session_id = payload.get("session_id")
    directory = Path(db_file).parent
    counts = agent_counts_zero()
    pending = []
    try:
        conn = open_db(db_file)
    except Exception:
        log_error(
            directory,
            "agent scan could not open {0}: {1}".format(
                db_file, traceback.format_exc().replace("\n", " | ")
            ),
        )
        return counts, pending
    try:
        with conn:
            counts = scan_agents(
                conn,
                session_id,
                payload.get("transcript_path"),
                directory,
                pending=pending,
            )
            resolve_agent_turns(conn, session_id)
    except Exception:
        log_error(
            directory,
            "agent scan failed: {0}".format(
                traceback.format_exc().replace("\n", " | ")
            ),
        )
    finally:
        conn.close()
    return counts, pending


# --------------------------------------------------------------------------
# Hook handlers
# --------------------------------------------------------------------------


def handle_prompt_submit(payload) -> None:
    """Fast path: record that a prompt was submitted. No transcript, no LLM.

    The payload's cwd is still what sessions.workspace records -- it is the
    project the human is working in -- but the row is written to the one
    machine-wide database.
    """
    session_id = payload.get("session_id")
    workspace = payload.get("cwd") or os.getcwd()
    conn = open_db(default_db_path())
    try:
        with conn:
            upsert_session(conn, session_id, workspace)
            conn.execute(
                "INSERT INTO turns (session_id, prompt_id, ts_start, permission_mode,"
                " prompt, is_synthetic, task_id, status)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')",
                (
                    session_id,
                    payload.get("prompt_id"),
                    utcnow(),
                    payload.get("permission_mode"),
                    payload.get("prompt"),
                    1 if is_synthetic_prompt(payload.get("prompt")) else 0,
                    prompt_task_id(payload.get("prompt")),
                ),
            )
    finally:
        conn.close()


def read_turns(conn, session_id, transcript_path):
    """Assemble the turns a session's cursor has not yet consumed."""
    offset = get_offset(conn, session_id)
    records, consumed = read_new_records(transcript_path, offset)
    return assemble_turns(records), consumed


def adopt_agents(conn, session_id, phantom, owner) -> None:
    """Re-point every agent row that hangs off `phantom` onto `owner`.

    Both joins are rewritten: agents.turn_id (set once the parent turn is
    known) and agents.prompt_id (what resolve_agent_turns re-joins on, and
    which would otherwise name a row that is about to disappear).
    """
    conn.execute(
        "UPDATE agents SET turn_id = ? WHERE turn_id = ?",
        (owner["turn_id"], phantom["turn_id"]),
    )
    if phantom["prompt_id"] and phantom["prompt_id"] != owner["prompt_id"]:
        conn.execute(
            "UPDATE agents SET turn_id = ?, prompt_id = COALESCE(?, prompt_id)"
            " WHERE session_id = ? AND prompt_id = ?",
            (owner["turn_id"], owner["prompt_id"], session_id, phantom["prompt_id"]),
        )


def drop_interjection_rows(conn, session_id, turns) -> int:
    """Remove the empty rows UserPromptSubmit opened for mid-turn messages.

    The hook cannot know that a prompt is about to be absorbed into the
    running turn -- it fires the moment the human presses enter -- so every
    interjection leaves behind a `pending` row with no transcript uuid, no
    response and no tool calls. Now that the text itself lives in the
    enclosing turn's prompt, that row is a duplicate; left alone it either
    floats untagged in the flow or, worse, gets adopted by find_turn_row's
    last-resort match and hands its prompt_id to an unrelated turn.

    Only a row that is still pending, still unkeyed, still empty and whose
    prompt is EXACTLY one of this batch's interjection texts is deleted, so a
    genuine prompt still in flight is never touched. Any agent that joined to
    it is handed to the enclosing turn first. Returns rows removed.

    Runs AFTER the batch is persisted, so the enclosing turn's row is there to
    inherit. Idempotent: a second pass finds nothing left to delete.
    """
    owners = {}
    for turn in turns:
        for text in turn["interjection_texts"]:
            owners.setdefault(text, turn)
    if not owners:
        return 0
    removed = 0
    for text, turn in owners.items():
        rows = conn.execute(
            "SELECT turn_id, prompt_id FROM turns WHERE session_id = ?"
            " AND status = 'pending' AND prompt_uuid IS NULL"
            " AND TRIM(COALESCE(prompt, '')) = ? AND COALESCE(response, '') = ''"
            " AND NOT EXISTS (SELECT 1 FROM tool_calls tc WHERE tc.turn_id = turns.turn_id)",
            (session_id, text),
        ).fetchall()
        if not rows:
            continue
        owner = conn.execute(
            "SELECT turn_id, prompt_id FROM turns WHERE prompt_uuid = ?",
            (turn["prompt_uuid"],),
        ).fetchone() if turn["prompt_uuid"] else None
        for row in rows:
            if owner is not None and owner["turn_id"] != row["turn_id"]:
                adopt_agents(conn, session_id, row, owner)
            conn.execute("DELETE FROM turns WHERE turn_id = ?", (row["turn_id"],))
            removed += 1
    return removed


def persist_turns(conn, session_id, turns, consumed, final_status):
    """Store assembled turns and advance the cursor.

    Returns ([turn_id needing classification, ...], counts). A turn needs
    classification when it became complete here OR when its response text
    changed under an already-classified row, which dropped its tag.
    """
    claimed = set()
    completed = []
    counts = {
        "added": 0,
        "updated": 0,
        "present": 0,
        "tool_calls": 0,
        "reclassified": 0,
    }
    for turn in turns:
        turn_id, newly, created, reclassify, retagged = persist_turn(
            conn, session_id, turn, final_status, claimed
        )
        if turn_id is None:
            continue
        counts["tool_calls"] += len(turn["tools"])
        if created:
            counts["added"] += 1
        elif newly:
            counts["updated"] += 1
        else:
            counts["present"] += 1
        # Reported count: only rows that lost an existing tag (see persist_turn).
        if retagged:
            counts["reclassified"] += 1
        if newly or reclassify:
            completed.append(turn_id)

    # The interjections are in their enclosing turns' prompts now, so the
    # hook rows they left behind are duplicates. Cleared here, after the
    # batch, so each one's enclosing row exists to inherit its agents.
    counts["repaired"] = drop_interjection_rows(conn, session_id, turns)

    # Rewind to the last turn's first byte rather than to EOF: tool
    # results can still be appended to it after Stop fires, and
    # re-reading it is harmless because persistence is idempotent.
    set_offset(conn, session_id, turns[-1]["byte_offset"] if turns else consumed)
    return completed, counts


def sweep(payload, final_status):
    """Extract every newly completed turn from the transcript.

    Returns (db_path, [turn_id, ...]) for turns that became complete here --
    plus any already-complete turn whose response text grew on this sweep and
    therefore had its tag dropped for re-classification.
    """
    session_id = payload.get("session_id")
    workspace = payload.get("cwd") or os.getcwd()
    db_file = default_db_path()

    conn = open_db(db_file)
    try:
        with conn:
            turns, consumed = read_turns(
                conn, session_id, payload.get("transcript_path")
            )
            branch = next((t["git_branch"] for t in turns if t["git_branch"]), None)
            version = next((t["cc_version"] for t in turns if t["cc_version"]), None)
            upsert_session(conn, session_id, workspace, branch, version)

            completed, _ = persist_turns(
                conn, session_id, turns, consumed, final_status
            )
        return db_file, completed
    finally:
        conn.close()


def service_port() -> int:
    """The port the viewer service listens on. Same rule as service.py."""
    raw = os.environ.get(SERVICE_PORT_ENV_VAR) or ""
    try:
        port = int(raw.strip())
    except (AttributeError, TypeError, ValueError):
        return SERVICE_DEFAULT_PORT
    return port if 1 <= port <= 65535 else SERVICE_DEFAULT_PORT


def service_is_up(port) -> bool:
    """One short loopback connect. A refused connection is the answer, not an
    error -- and the timeout is the whole budget this check may spend."""
    try:
        with socket.create_connection(
            (SERVICE_HOST, port), SERVICE_PROBE_TIMEOUT_S
        ):
            return True
    except OSError:
        return False


def ensure_service() -> None:
    """Watchdog: leave a viewer service running behind every finished turn.

    Runs at the very END of Stop and SessionEnd, after the audit itself is
    safely written, because it exists for the human's convenience and the
    audit does not depend on it. The contract is the hook contract: it never
    raises, never prints, and never blocks for longer than the connect
    timeout. A machine that does not want a listening socket sets
    CLAUDE_AUDIT_NO_SERVICE=1 and nothing here happens.

    Note that the probe only proves SOMETHING is listening on the port. That
    is deliberate: the alternative is an HTTP round trip on every turn, and
    the failure it would catch (another program squatting on 4737) is one the
    service itself reports the moment it cannot bind.
    """
    try:
        if os.environ.get(SERVICE_DISABLE_ENV_VAR) == "1":
            return
        port = service_port()
        if service_is_up(port):
            return
        script = Path(__file__).resolve().parent / SERVICE_SCRIPT
        if not script.is_file():
            return
        subprocess.Popen(
            [sys.executable, str(script), "serve"],
            cwd=str(script.parent),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # outlives this hook process
        )
    except Exception:
        return  # a hook must never fail the turn, least of all over a viewer


def handle_stop(payload) -> None:
    db_file, completed = sweep(payload, "complete")
    # `completed` holds newly finished turns AND turns whose text grew since
    # the last sweep (the previous turn is deliberately re-read every time,
    # because the cursor is rewound to its first byte). Both need a classifier.
    for turn_id in completed:
        spawn_classifier(db_file, turn_id, Path(db_file).parent)
    # After the turn extraction above, so an agent spawned by the turn that
    # just ended finds its parent row already written.
    _, agents_pending = sweep_agents(payload, db_file)
    # Subagents are classified like turns, and just as much out of band: the
    # agent rows are already durable, so a classifier that never lands costs
    # a tag and nothing else (SessionEnd re-spawns for whatever is still
    # untagged, and `sync --classify` is the last net).
    for session_id, agent_id in agents_pending:
        spawn_agent_classifier(db_file, session_id, agent_id, Path(db_file).parent)
    ensure_service()  # last: the audit is already durable by this point


def handle_session_end(payload) -> None:
    session_id = payload.get("session_id")
    workspace = payload.get("cwd") or os.getcwd()

    # Turns finalised only here never saw a Stop, so they are marked 'swept'.
    db_file, _ = sweep(payload, "swept")
    directory = Path(db_file).parent
    sweep_agents(payload, db_file)  # pending agents are recovered below

    conn = open_db(db_file)
    try:
        with conn:
            branch = conn.execute(
                "SELECT git_branch FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ?,"
                " git_branch = COALESCE(git_branch, ?) WHERE session_id = ?",
                (
                    utcnow(),
                    payload.get("reason"),
                    None if branch and branch["git_branch"] else git_branch(workspace),
                    session_id,
                ),
            )
        # Recover any turn whose classifier never landed.
        unclassified = [
            row["turn_id"]
            for row in conn.execute(
                "SELECT turn_id FROM turns WHERE session_id = ? AND tag_source IS NULL"
                " AND status != 'pending'",
                (session_id,),
            ).fetchall()
        ]
        # Same recovery for the session's subagents: one query catches both
        # the agents this sweep stored and any whose Stop-hook classifier
        # died on the way.
        unclassified_agents = [
            row["agent_id"]
            for row in conn.execute(
                "SELECT agent_id FROM agents WHERE session_id = ? AND tag IS NULL",
                (session_id,),
            ).fetchall()
        ]
    finally:
        conn.close()

    for turn_id in unclassified:
        spawn_classifier(db_file, turn_id, directory)
    for agent_id in unclassified_agents:
        spawn_agent_classifier(db_file, session_id, agent_id, directory)

    # One archive tree now serves every workspace on the machine, so the copy
    # is named the same way the sync names it: <workspace folder>-<session>.
    name = archive_basename(workspace, session_id)
    archive_transcript(payload.get("transcript_path"), directory, session_id, name=name)
    try:
        archive_subagents(
            payload.get("transcript_path"), directory, session_id, name=name
        )
    except Exception:
        log_error(
            directory,
            "subagent archive failed for {0}: {1}".format(
                session_id, traceback.format_exc().replace("\n", " | ")
            ),
        )

    ensure_service()  # last: the audit is already durable by this point


def archive_basename(workspace, session_id) -> str:
    """Archive name for one session: <workspace folder>-<session id>.

    A single machine-wide audit directory holds transcripts from every
    project, and session ids alone would say nothing about where a transcript
    came from -- so the workspace's folder name is prefixed. The hooks and the
    sync both go through here, so they name the same file.
    """
    folder = Path(workspace).name if workspace else ""
    return "{0}-{1}".format(folder or "root", session_id)


def archive_subagents(transcript_path, directory, session_id, name=None) -> int:
    """Copy the whole subagents tree next to the archived transcript.

    Mirrored verbatim under <audit>/transcripts/<session_id>-subagents/ with
    relative paths preserved, so the workflows/<run>/ structure (and each
    agent's meta.json and journal) survives Claude's own transcript cleanup.
    A file whose archived copy is already at least as long is left alone, so
    a repeated sync is cheap. Returns the number of files written.

    `name` overrides the archive's basename (see archive_basename: one archive
    directory serves every workspace, so the copies are prefixed with the
    workspace folder); the session id still locates the source tree.
    """
    root = subagents_dir(transcript_path, session_id)
    if root is None or not session_id:
        return 0
    target_root = Path(directory) / "transcripts" / "{0}-subagents".format(
        name or session_id
    )
    written = 0
    for source in sorted(root.rglob("*")):
        if not source.is_file():
            continue
        target = target_root / source.relative_to(root)
        try:
            if target.is_file() and target.stat().st_size >= source.stat().st_size:
                continue
        except OSError:
            pass
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                dst.write(chunk)
        written += 1
    return written


def archive_transcript(
    transcript_path, directory, session_id, skip_if_current=False, name=None
):
    """Copy the raw transcript verbatim -- ground truth that outlives Claude's
    own 30-day transcript cleanup.

    With `skip_if_current` an existing archive that is already at least as long
    as the source is left alone, so a repeated sync does not recopy hundreds of
    megabytes. `name` overrides the archive's basename (see archive_subagents).
    Returns True when a copy was written.
    """
    source = Path(transcript_path) if transcript_path else None
    if source is None or not source.is_file() or not session_id:
        return False
    target_dir = Path(directory) / "transcripts"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "{0}.jsonl".format(name or session_id)
    if skip_if_current:
        try:
            if target.is_file() and target.stat().st_size >= source.stat().st_size:
                return False
        except OSError:
            pass
    with source.open("rb") as src, target.open("wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
    return True


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


def agent_ref(session_id, agent_id) -> str:
    """The `classify --agent` selector for one subagent."""
    return "{0}:{1}".format(session_id, agent_id)


def parse_agent_ref(ref):
    """(session_id, agent_id) from a SESSION:AGENT selector, or (None, None).

    Split from the RIGHT: a session id is a uuid and carries no colon, so
    whatever precedes the last one is the session even if a future agent id
    grows a prefix of its own.
    """
    blob = (ref or "").strip()
    if ":" not in blob:
        return None, None
    session_id, agent_id = blob.rsplit(":", 1)
    session_id, agent_id = session_id.strip(), agent_id.strip()
    if not session_id or not agent_id:
        return None, None
    return session_id, agent_id


def spawn_classifier(db_file, turn_id, directory) -> None:
    """Fire-and-forget the tagging pass so the turn is never held up."""
    spawn_classify_process(db_file, ["--turn", str(turn_id)], directory)


def spawn_agent_classifier(db_file, session_id, agent_id, directory) -> None:
    """The subagent equivalent: same detached process, same never-block rule."""
    spawn_classify_process(
        db_file, ["--agent", agent_ref(session_id, agent_id)], directory
    )


def spawn_classify_process(db_file, selector, directory) -> None:
    """Detach one `classify` run against `db_file`. Never blocks, never raises
    anything the hook has to care about."""
    env = dict(os.environ)
    env["CLAUDE_AUDIT_SKIP"] = "1"  # the classifier's own claude run fires hooks
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        errlog = (directory / "hook-errors.log").open("a", encoding="utf-8")
    except Exception:
        errlog = subprocess.DEVNULL
    try:
        subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(__file__),
                "classify",
                "--db",
                str(db_file),
            ]
            + list(selector),
            cwd=str(directory),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=errlog,
            start_new_session=True,  # survives the hook process exiting
        )
    finally:
        if errlog is not subprocess.DEVNULL:
            errlog.close()


def strip_fences(text) -> str:
    """Unwrap ```json ... ``` and trim to the outermost JSON object."""
    blob = (text or "").strip()
    if blob.startswith("```"):
        lines = blob.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        blob = "\n".join(lines).strip()
    start, end = blob.find("{"), blob.rfind("}")
    if start != -1 and end > start:
        blob = blob[start : end + 1]
    return blob


def build_classify_prompt(prompt, response) -> str:
    return "{0}\n<turn>\n<user_prompt>\n{1}\n</user_prompt>\n<assistant_response>\n{2}\n</assistant_response>\n</turn>\n".format(
        CLASSIFY_INSTRUCTIONS, prompt, response
    )


def classify_with_haiku(prompt, response):
    """Ask Haiku for {tag, title}. Returns None on any failure."""
    env = dict(os.environ)
    env["CLAUDE_AUDIT_SKIP"] = "1"
    try:
        # Run somewhere disposable so this never lands in the workspace's own
        # transcript project directory.
        proc = subprocess.run(
            ["claude", "-p", "--model", HAIKU_MODEL, "--output-format", "json"],
            input=build_classify_prompt(prompt, response),
            cwd=str(classifier_workdir()),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=CLASSIFY_TIMEOUT_S,
            text=True,
        )
    except Exception:
        return None

    if proc.returncode != 0:
        return None
    try:
        envelope = json.loads(proc.stdout)
    except ValueError:
        return None
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        return None

    try:
        parsed = json.loads(strip_fences(envelope.get("result")))
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    tag = parsed.get("tag")
    if tag not in VALID_TAGS:
        return None
    title = parsed.get("title")
    if not isinstance(title, str) or not title.strip():
        return None
    return {"tag": tag, "title": title.strip()[:TITLE_MAX_CHARS]}


def is_question(prompt) -> bool:
    """True when the prompt reads as a question rather than an instruction."""
    blob = " ".join((prompt or "").split())
    if not blob:
        return False
    if blob.endswith("?"):
        return True
    first = re.sub(r"[^a-z]", "", blob.split(" ", 1)[0].lower())
    return first in QUESTION_OPENERS


def heuristic_classify(prompt, response, permission_mode, tool_calls=0):
    """Deterministic fallback used whenever the model call is unavailable.

    First match wins, in this order:
      1. plan mode -- the harness says outright that nothing may be changed;
      2. the keyword table, most specific tag first (see HEURISTIC_KEYWORDS);
      3. chat -- a question that spawned no tool call produced no work;
      4. build, the default for a turn that did something.
    """
    if (permission_mode or "") == "plan":
        return {"tag": "plan", "title": first_sentence(prompt)}

    blob = "{0}\n{1}".format(prompt or "", response or "").lower()
    tag = None
    for candidate, keywords in HEURISTIC_KEYWORDS:
        if any(keyword in blob for keyword in keywords):
            tag = candidate
            break
    if tag is None:
        tag = "chat" if (is_question(prompt) and not tool_calls) else "build"
    return {"tag": tag, "title": first_sentence(prompt)}


def classify_turn(conn, turn_id):
    """Tag and title one turn in-process. Returns the stored result, or None
    when there is no such turn."""
    row = conn.execute(
        "SELECT prompt, response, permission_mode, is_synthetic FROM turns"
        " WHERE turn_id = ?",
        (turn_id,),
    ).fetchone()
    if row is None:
        return None

    # Machine-generated prompts carry no engineering intent for Haiku to
    # classify; they get a deterministic system:* kind and title instead, so
    # the human tag ratios (see TAG_TAXONOMY) stay pure.
    if row["is_synthetic"]:
        kind, title = synthetic_kind(row["prompt"])
        result = {"tag": kind, "title": title}
        source = "system"
    else:
        prompt = truncate(row["prompt"], MAX_CLASSIFY_CHARS)
        response = truncate(row["response"], MAX_CLASSIFY_CHARS)

        result, source = None, "heuristic"
        if os.environ.get("CLAUDE_AUDIT_NO_LLM") != "1":
            result = classify_with_haiku(prompt, response)
        if result is not None:
            source = "haiku"
        else:
            # `chat` hinges on whether the turn actually did anything, which
            # the prompt text cannot say -- so the heuristic gets the count.
            tool_calls = conn.execute(
                "SELECT COUNT(*) AS n FROM tool_calls WHERE turn_id = ?", (turn_id,)
            ).fetchone()["n"]
            result = heuristic_classify(
                prompt, response, row["permission_mode"], tool_calls
            )

    with conn:
        conn.execute(
            "UPDATE turns SET tag = ?, title = ?, tag_source = ? WHERE turn_id = ?",
            (result["tag"], result["title"], source, turn_id),
        )
    return {"turn": turn_id, "tag_source": source, **result}


def classify_agent(conn, session_id, agent_id):
    """Tag and title one subagent, exactly as classify_turn does a turn.

    The subagent is put to the model as a turn -- its spawning instruction is
    the prompt, what it handed back is the response -- so the rubric, the
    VALID_TAGS validation, the title rule and the CLAUDE_AUDIT_NO_LLM opt-out
    are literally the same machinery, and an agent's tag is comparable with a
    turn's. There is no synthetic case: nothing spawns a subagent by accident.
    Returns the stored result, or None when there is no such agent.
    """
    row = conn.execute(
        "SELECT prompt, result, description, tool_call_count FROM agents"
        " WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    ).fetchone()
    if row is None:
        return None

    # A workflow step's own prompt can be empty (the step is described by its
    # label instead), and an unnamed agent is better titled by its label than
    # by nothing at all.
    prompt = truncate(row["prompt"] or row["description"] or "", MAX_CLASSIFY_CHARS)
    response = truncate(row["result"], MAX_CLASSIFY_CHARS)

    result, source = None, "heuristic"
    if os.environ.get("CLAUDE_AUDIT_NO_LLM") != "1":
        result = classify_with_haiku(prompt, response)
    if result is not None:
        source = "haiku"
    else:
        result = heuristic_classify(
            prompt, response, None, row["tool_call_count"] or 0
        )

    with conn:
        conn.execute(
            "UPDATE agents SET tag = ?, title = ?, tag_source = ?"
            " WHERE session_id = ? AND agent_id = ?",
            (result["tag"], result["title"], source, session_id, agent_id),
        )
    return {
        "agent": agent_ref(session_id, agent_id),
        "tag_source": source,
        **result,
    }


def cmd_classify(args) -> int:
    conn = open_db(default_db_path(args.db))
    try:
        if getattr(args, "agent", None):
            session_id, agent_id = parse_agent_ref(args.agent)
            if not session_id:
                sys.stderr.write(
                    "audit_log: --agent wants SESSION_ID:AGENT_ID, got {0}\n".format(
                        args.agent
                    )
                )
                return 2
            result = classify_agent(conn, session_id, agent_id)
            if result is None:
                sys.stderr.write("audit_log: no agent {0}\n".format(args.agent))
                return 1
        else:
            result = classify_turn(conn, args.turn)
            if result is None:
                sys.stderr.write("audit_log: no turn {0}\n".format(args.turn))
                return 1
        json.dump(result, sys.stdout)
        sys.stdout.write("\n")
        return 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Sync (backfill)
# --------------------------------------------------------------------------


def scan_transcript_meta(path):
    """One full pass over a transcript for session-level facts.

    The cursor-based extraction only ever sees the tail of a file, so the
    session row's own fields (when it started and ended, which branch and
    client version it ran on) are collected here instead. Timestamps are taken
    as min/max rather than literally first/last line so a file that is not
    perfectly ordered cannot produce a session that ended before it began.
    """
    meta = {
        "started_at": None,
        "ended_at": None,
        "git_branch": None,
        "cc_version": None,
        "workspace": None,
        "lines": 0,
        "parse_errors": 0,
    }
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            meta["lines"] += 1
            try:
                obj = json.loads(stripped)
            except ValueError:
                meta["parse_errors"] += 1
                continue
            if not isinstance(obj, dict):
                continue
            timestamp = obj.get("timestamp")
            if isinstance(timestamp, str) and timestamp:
                if meta["started_at"] is None or timestamp < meta["started_at"]:
                    meta["started_at"] = timestamp
                if meta["ended_at"] is None or timestamp > meta["ended_at"]:
                    meta["ended_at"] = timestamp
            for key, field in (
                ("git_branch", "gitBranch"),
                ("cc_version", "version"),
                ("workspace", "cwd"),
            ):
                if obj.get(field):
                    meta[key] = obj[field]
    return meta


def session_outline(path):
    """(lineage, interjections) for one transcript, in a single streamed pass.

    A deliberately minimal re-walk of the file: it draws turn boundaries with
    classify_line, exactly as assemble_turns does, but keeps nothing except
    the uuids and the interjection texts -- so a multi-hundred megabyte
    transcript costs one pass and a few kilobytes of memory.

    `lineage` is one {prompt_uuid, parent_uuid, last_uuid} per real turn.
    `interjections` is one {uuid, kind, text, addition, owner} per mid-turn
    message, where `owner` is the prompt_uuid of the turn it was folded into
    and `addition` is the exact block it contributes to that turn's prompt.
    Both are what the repair pass needs to tell a phantom row from a real one
    without assembling anything.
    """
    lineage = []
    interjections = []
    current = None
    open_turn = False
    pending = []
    for obj in iter_transcript_objects(path):
        verdict, payload = classify_line(obj, pending, open_turn)
        if verdict == "skip":
            continue
        if verdict == "start":
            open_turn = True
            current = None
            if obj.get("uuid"):
                current = {
                    "prompt_uuid": obj["uuid"],
                    "parent_uuid": obj.get("parentUuid"),
                    "last_uuid": obj["uuid"],
                }
                lineage.append(current)
            continue
        if verdict == "interject":
            addition = interjection_addition(payload[0], payload[1])
            if addition is not None:
                interjections.append(
                    {
                        "uuid": obj.get("uuid"),
                        "kind": payload[0],
                        "text": (payload[1] or "").strip(),
                        "addition": addition,
                        "owner": current["prompt_uuid"] if current else None,
                    }
                )
        if current is not None:
            chain_tip(current, obj)
    return lineage, interjections


def session_lineage(path):
    """Prompt parentUuid + final-line uuid for every turn in a transcript."""
    return session_outline(path)[0]


def relink_turns(conn, session_id, path) -> int:
    """Backfill turns.parent_uuid / turns.last_uuid for one session.

    Rows written before this release (and any row the live hooks opened but
    never saw a transcript line for) carry NULL lineage. When any of the
    session's turns is missing its last_uuid the transcript is re-read IN
    MEMORY -- the cursors table is not read and not written, and no column
    other than these two is touched, so a relink can never disturb the
    incremental sync, the tags or the token counts. Rows are matched on
    prompt_uuid, which is unique. Returns the number of rows updated.
    """
    stale = conn.execute(
        "SELECT 1 FROM turns WHERE session_id = ? AND last_uuid IS NULL LIMIT 1",
        (session_id,),
    ).fetchone()
    if stale is None:
        return 0
    relinked = 0
    for entry in session_lineage(path):
        cursor = conn.execute(
            "UPDATE turns SET parent_uuid = ?, last_uuid = ?"
            " WHERE session_id = ? AND prompt_uuid = ?",
            (
                entry["parent_uuid"],
                entry["last_uuid"],
                session_id,
                entry["prompt_uuid"],
            ),
        )
        relinked += cursor.rowcount
    return relinked


def renumber_tool_calls(conn, turn_id, turn) -> None:
    """Re-seq a turn's tool calls into the order the transcript really has.

    Called after a phantom's rows are re-pointed onto their enclosing turn,
    where they would otherwise keep the phantom's own 0..n numbering and
    interleave wrongly with the rows already there. Rows the re-assembly does
    not know (an id lost to truncation) keep their relative order at the end.
    """
    order = {
        entry["tool_use_id"]: seq
        for seq, entry in enumerate(turn["tools"])
        if entry["tool_use_id"]
    }
    rows = conn.execute(
        "SELECT tool_call_id, tool_use_id, seq FROM tool_calls WHERE turn_id = ?"
        " ORDER BY seq, tool_call_id",
        (turn_id,),
    ).fetchall()
    tail = len(order)
    updates = []
    for row in rows:
        seq = order.get(row["tool_use_id"])
        if seq is None:
            seq = tail
            tail += 1
        if seq != row["seq"]:
            updates.append((seq, row["tool_call_id"]))
    if updates:
        conn.executemany(
            "UPDATE tool_calls SET seq = ? WHERE tool_call_id = ?", updates
        )


def refresh_merged_turn(conn, row, turn) -> None:
    """Rewrite an enclosing turn from the re-assembled version of itself.

    Everything the phantom had stolen comes back: the prompt (now carrying the
    marked mid-turn addition), the response text, the token sums, the end of
    the turn and its lineage tip. tag/title/tag_source are dropped so the turn
    is classified again over the whole of what it actually did -- except for a
    synthetic turn, whose kind is derived from its prompt and never guessed.
    """
    reset = "" if row["is_synthetic"] else ", tag = NULL, title = NULL, tag_source = NULL"
    conn.execute(
        "UPDATE turns SET prompt = ?, response = ?, ts_end = ?, duration_s = ?,"
        " last_uuid = ?, interjections = ?, model = COALESCE(?, model),"
        " input_tokens = ?, output_tokens = ?, cache_read_tokens = ?,"
        " cache_creation_tokens = ?, thinking_tokens = ?{0}"
        " WHERE turn_id = ?".format(reset),
        (
            turn["prompt"],
            "\n\n".join(turn["texts"]),
            turn["ts_end"],
            duration_seconds(turn["ts_start"], turn["ts_end"]),
            turn["last_uuid"],
            turn["interjections"],
            turn["model"],
            turn["tokens"]["input_tokens"],
            turn["tokens"]["output_tokens"],
            turn["tokens"]["cache_read_tokens"],
            turn["tokens"]["cache_creation_tokens"],
            turn["tokens"]["thinking_tokens"],
            row["turn_id"],
        ),
    )


def repair_phantom_turns(conn, session_id, path) -> int:
    """Merge phantom turns back into the turn that was actually running.

    A phantom is a stored turn whose prompt line is NOT a turn boundary under
    the mid-flight rule (see classify_line) -- a message the human sent while
    the assistant was working, recorded as a turn of its own by an older
    release of this script or by the UserPromptSubmit hook, which fires before
    anyone can know the message will be absorbed. It owns a slice of the
    enclosing turn's tool calls and tokens, has no response, stays untagged,
    and floats unparented in the flow view.

    Two shapes are recognised, and only these two:
      * prompt_uuid names a line the new rule demotes to an interjection;
      * prompt_uuid is NULL and the prompt text is EXACTLY an interjection's,
        on a row still pending and empty -- a hook row. Guarded by time as
        well: the submission must fall inside the enclosing turn, so a fresh
        prompt that merely repeats an earlier message is never eaten.
    A row whose prompt_uuid the transcript does not mention is left alone, so
    a truncated or archived transcript can never delete history.

    Cursor-free and idempotent, like relink_turns: the file is re-read in
    memory, the cursors table is neither read nor written, and a second run
    finds no phantoms because the first one removed them. Returns the number
    of phantom rows merged away.
    """
    _, interjections = session_outline(path)
    if not interjections:
        return 0
    by_uuid = {i["uuid"]: i for i in interjections if i.get("uuid") and i.get("owner")}
    by_text = {}
    for entry in interjections:
        # Only a human message ever left a hook row behind to match by text;
        # a notification is never submitted through UserPromptSubmit.
        if entry["kind"] == "prompt" and entry["text"] and entry.get("owner"):
            by_text.setdefault(entry["text"], entry)

    candidates = []
    owners = {}
    for row in conn.execute(
        "SELECT turn_id, prompt_id, prompt_uuid, prompt, status, ts_start,"
        " is_synthetic FROM turns WHERE session_id = ?",
        (session_id,),
    ).fetchall():
        if row["prompt_uuid"]:
            owners[row["prompt_uuid"]] = row
            hit = by_uuid.get(row["prompt_uuid"])
        elif row["status"] == "pending":
            hit = by_text.get((row["prompt"] or "").strip())
        else:
            hit = None
        if hit is not None:
            candidates.append((row, hit["owner"]))

    # An enclosing turn stored before this rule existed is missing the text
    # even when its phantom is long gone -- the hook row was recycled onto an
    # unrelated turn instead of being left to float. Such a turn is not
    # deleted, only rewritten, and once it carries the marked addition this
    # test stops finding it.
    stale = {
        entry["owner"]
        for entry in interjections
        if entry.get("owner") in owners
        and entry["addition"] not in (owners[entry["owner"]]["prompt"] or "")
    }
    if not candidates and not stale:
        return 0

    # Only now is the full re-assembly worth its memory: the light outline
    # above already proved there is something to repair.
    rebuilt = {
        turn["prompt_uuid"]: turn
        for turn in assemble_turns(iter_transcript_records(path))
        if turn["prompt_uuid"]
    }

    merged = 0
    touched = {}
    for row, owner_uuid in candidates:
        turn = rebuilt.get(owner_uuid)
        if turn is None:
            continue
        owner = conn.execute(
            "SELECT turn_id, prompt_id, is_synthetic FROM turns"
            " WHERE session_id = ? AND prompt_uuid = ?",
            (session_id, owner_uuid),
        ).fetchone()
        if owner is None or owner["turn_id"] == row["turn_id"]:
            continue
        if not row["prompt_uuid"]:
            # A hook row carries no uuid, so time is the only evidence that
            # this submission belongs inside that turn rather than after it.
            if (row["ts_start"] or "") > (turn["ts_end"] or ""):
                continue
        # The same sweep may already have re-written the enclosing turn's tool
        # calls from the transcript (persist_turn deletes and re-inserts them),
        # in which case the calls the phantom holds are the SAME calls under an
        # older turn_id. Re-pointing them blindly would leave the merged turn
        # carrying each of them twice, with a duplicate (turn_id, seq) that no
        # later run cleans up. The freshly written copy wins; only a call the
        # enclosing turn does not already have is carried over.
        conn.execute(
            "DELETE FROM tool_calls WHERE turn_id = ? AND tool_use_id IS NOT NULL"
            " AND tool_use_id IN"
            " (SELECT tool_use_id FROM tool_calls WHERE turn_id = ?)",
            (row["turn_id"], owner["turn_id"]),
        )
        conn.execute(
            "UPDATE tool_calls SET turn_id = ? WHERE turn_id = ?",
            (owner["turn_id"], row["turn_id"]),
        )
        adopt_agents(conn, session_id, row, owner)
        conn.execute("DELETE FROM turns WHERE turn_id = ?", (row["turn_id"],))
        touched[owner["turn_id"]] = (owner, turn)
        merged += 1

    for owner_uuid in stale:
        turn = rebuilt.get(owner_uuid)
        if turn is not None:
            touched.setdefault(owners[owner_uuid]["turn_id"], (owners[owner_uuid], turn))

    for owner, turn in touched.values():
        refresh_merged_turn(conn, owner, turn)
        renumber_tool_calls(conn, owner["turn_id"], turn)
    return merged


def sync_session(conn, session_id, path, workspace, directory):
    """Backfill one transcript file.

    `workspace` is the project the session ran in (stored on the session row);
    `directory` is the audit directory of the open database, where per-agent
    failures are logged.

    Returns (counts, agent_counts, meta, is_new_session).
    """
    meta = scan_transcript_meta(path)
    existing = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()

    with conn:
        turns, consumed = read_turns(conn, session_id, path)
        branch = meta["git_branch"] or next(
            (t["git_branch"] for t in turns if t["git_branch"]), None
        )
        version = meta["cc_version"] or next(
            (t["cc_version"] for t in turns if t["cc_version"]), None
        )
        upsert_session(
            conn, session_id, meta["workspace"] or workspace, branch, version
        )
        conn.execute(
            "UPDATE sessions SET started_at = COALESCE(?, started_at),"
            " ended_at = COALESCE(?, ended_at) WHERE session_id = ?",
            (meta["started_at"], meta["ended_at"], session_id),
        )
        # At sync time the file is closed by EOF, so every assembled turn is
        # whole; 'swept' records that no Stop hook witnessed it live, while
        # persist_turn leaves anything already 'complete' alone.
        _, counts = persist_turns(conn, session_id, turns, consumed, "swept")
        # After the turns, so a parent turn written by this very pass is
        # already there for the promptId join.
        agent_counts = scan_agents(conn, session_id, path, directory)
        resolve_agent_turns(conn, session_id)
        # After the agents, so a phantom's agent rows are already attached and
        # can be handed over whole; cursor-free, like the relink below.
        counts["repaired"] = counts.get("repaired", 0) + repair_phantom_turns(
            conn, session_id, path
        )
        # Last, and cursor-free: whatever the rest of the sweep did or skipped,
        # every turn row that exists for this session now gets its lineage.
        counts["relinked"] = relink_turns(conn, session_id, path)
    return counts, agent_counts, meta, existing is None


TURN_TOTAL_KEYS = (
    "added",
    "updated",
    "present",
    "tool_calls",
    "reclassified",
    "relinked",
    "repaired",
)

PROBE_FILES = 3      # newest transcripts consulted when recovering a cwd
PROBE_LINES = 500    # lines read per transcript before giving up on it


def probe_workspace(project_dir):
    """Recover a project directory's real workspace path from its transcripts.

    The directory name is a LOSSY slug -- every non-alphanumeric character
    becomes a dash, so `-Users-mac-my-app` could be /Users/mac/my/app,
    /Users/mac/my-app or /Users/mac/my.app and no amount of guessing tells
    them apart. The transcripts themselves do not guess: each line carries the
    session's `cwd`. The newest few files are consulted (an old one may predate
    the field, or belong to a since-renamed path) and the first parseable line
    with a cwd wins. Returns (path string, None) or (None, reason).
    """
    try:
        files = [
            entry
            for entry in project_dir.iterdir()
            if entry.is_file() and entry.suffix == ".jsonl"
        ]
    except OSError as exc:
        return None, "unreadable: {0}".format(exc.strerror or exc)
    if not files:
        # Common and harmless: a directory holding only memory/ or a session
        # subdirectory whose transcript has not been written yet.
        return None, "no session transcripts"

    def mtime(path):
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files.sort(key=mtime, reverse=True)
    for path in files[:PROBE_FILES]:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for index, line in enumerate(fh):
                    if index >= PROBE_LINES:
                        break
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        obj = json.loads(stripped)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and isinstance(obj.get("cwd"), str):
                        if obj["cwd"].strip():
                            return obj["cwd"].strip(), None
        except OSError:
            continue
    return None, "no cwd recoverable from transcripts"


def discover_workspaces(root):
    """Every syncable (project_dir, workspace) pair under the projects root.

    Returns (targets, skipped) where `skipped` is [(dir name, reason)]. A
    project directory is skipped when no cwd can be recovered, when the
    workspace folder has since been deleted, or when it lives under the system
    temp directory -- that last one is this plugin's own classifier scratch
    session (see classifier_workdir), which must never audit itself.
    """
    targets = []
    skipped = []
    seen = {}
    try:
        entries = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError:
        return targets, skipped

    temp_root = tempfile.gettempdir()
    for project_dir in entries:
        workspace, reason = probe_workspace(project_dir)
        if not workspace:
            skipped.append((project_dir.name, reason))
            continue
        path = Path(workspace).expanduser()
        if path_under(path, temp_root):
            skipped.append((project_dir.name, "temp-dir workspace: {0}".format(path)))
            continue
        if not path.is_dir():
            skipped.append((project_dir.name, "workspace gone: {0}".format(path)))
            continue
        key = str(path.resolve())
        if key in seen:
            skipped.append(
                (project_dir.name, "duplicate of {0}".format(seen[key]))
            )
            continue
        seen[key] = project_dir.name
        targets.append((project_dir, path))
    return targets, skipped


def sync_workspace(conn, workspace, root, db_file):
    """Mirror every transcript of one workspace into an open database.

    `root` is the workspace's project directory (passed in rather than derived,
    because the discovery pass enumerates directories and the slug does not
    round-trip). One database serves every workspace, so archives are named
    <workspace-folder>-<session> and one archive directory holds them all
    without collisions -- the same names the SessionEnd hook writes.
    Returns a summary dict; never raises for a single bad transcript.
    """
    directory = Path(db_file).parent

    # Only *.jsonl directly in the project directory is a session transcript;
    # the sibling subdirectories hold subagent traffic, tool results and
    # per-session metadata.
    try:
        files = sorted(
            entry
            for entry in root.iterdir()
            if entry.is_file() and entry.suffix == ".jsonl"
        )
    except OSError:
        files = []

    summary = {
        "workspace": workspace,
        "root": root,
        "db_file": db_file,
        "sessions": len(files),
        "new_sessions": 0,
        "totals": {key: 0 for key in TURN_TOTAL_KEYS},
        "agent_totals": agent_counts_zero(),
        "archived": 0,
        "agent_archived": 0,
        "failures": [],
        "parse_errors": [],
        "remarked": 0,
        "classified": None,
    }

    for path in files:
        session_id = path.stem
        archive_name = archive_basename(workspace, session_id)
        try:
            counts, agent_counts, meta, is_new = sync_session(
                conn, session_id, path, workspace, directory
            )
        except Exception:
            summary["failures"].append(
                (path.name, traceback.format_exc().strip().splitlines()[-1])
            )
            log_error(
                directory,
                "sync failed for {0}: {1}".format(
                    path.name, traceback.format_exc().replace("\n", " | ")
                ),
            )
            continue
        for key in summary["totals"]:
            summary["totals"][key] += counts.get(key, 0)
        for key in summary["agent_totals"]:
            summary["agent_totals"][key] += agent_counts[key]
        summary["new_sessions"] += 1 if is_new else 0
        if meta["parse_errors"]:
            summary["parse_errors"].append((path.name, meta["parse_errors"]))
        try:
            if archive_transcript(
                path, directory, session_id, skip_if_current=True, name=archive_name
            ):
                summary["archived"] += 1
        except Exception:
            summary["failures"].append((path.name, "archive failed"))
            log_error(
                directory,
                "archive failed for {0}: {1}".format(
                    path.name, traceback.format_exc().replace("\n", " | ")
                ),
            )
        try:
            summary["agent_archived"] += archive_subagents(
                path, directory, session_id, name=archive_name
            )
        except Exception:
            summary["failures"].append((path.name, "subagent archive failed"))
            log_error(
                directory,
                "subagent archive failed for {0}: {1}".format(
                    path.name, traceback.format_exc().replace("\n", " | ")
                ),
            )
    return summary


def backfill_task_links(conn) -> int:
    """Give already-stored rows their background-task links.

    Cursor-free and transcript-free, exactly like the re-mark pass below: the
    evidence is text the mirror has ALREADY captured -- a spawning call's
    result and a notification turn's prompt -- so rows written before these
    columns existed converge on the next sync without re-reading a single
    JSONL file. Only rows still missing a link are considered, and only a row
    whose parse actually adds something is written, so a repeated sync writes
    nothing and reports zero. Returns the number of rows linked.
    """
    linked = 0
    placeholders = ", ".join("?" * len(TASK_LINK_TOOLS))
    for row in conn.execute(
        "SELECT tool_call_id, tool_name, result_text, task_id, run_ref"
        " FROM tool_calls WHERE (task_id IS NULL OR run_ref IS NULL)"
        " AND result_text IS NOT NULL AND tool_name IN ({0})".format(placeholders),
        TASK_LINK_TOOLS,
    ).fetchall():
        task_id, run_ref = parse_task_link(row["tool_name"], row["result_text"])
        # Whatever the row already knows wins: this pass only fills blanks, it
        # never revises a link (a Workflow result carries a run ref but no task
        # id, and that row must not be rewritten on every later sync).
        new_task = row["task_id"] or task_id
        new_ref = row["run_ref"] or run_ref
        if new_task == row["task_id"] and new_ref == row["run_ref"]:
            continue
        conn.execute(
            "UPDATE tool_calls SET task_id = ?, run_ref = ? WHERE tool_call_id = ?",
            (new_task, new_ref, row["tool_call_id"]),
        )
        linked += 1

    # The other end of the loop: the synthetic turn that reported the task
    # finished. The LIKE is only a prefilter -- prompt_task_id() re-checks for
    # a <task-notification> wrapper and a well-formed id.
    for row in conn.execute(
        "SELECT turn_id, prompt FROM turns WHERE task_id IS NULL"
        " AND prompt LIKE '%<task-id>%'"
    ).fetchall():
        task_id = prompt_task_id(row["prompt"])
        if not task_id:
            continue
        conn.execute(
            "UPDATE turns SET task_id = ? WHERE turn_id = ?", (task_id, row["turn_id"])
        )
        linked += 1

    conn.commit()
    return linked


# (table, primary key) for the two tables holding tool calls. They are tagged
# by identical rules -- a subagent's Bash call is the same kind of work a
# turn's is -- so the backfill walks both.
TOOL_CALL_TABLES = (("tool_calls", "tool_call_id"), ("agent_tool_calls", "atc_id"))
TOOL_TAG_BATCH = 500


def backfill_tool_tags(conn) -> int:
    """Tag and summarise stored tool calls that predate the columns.

    Cursor-free and transcript-free like the other backfills: the evidence is
    the input payload already in the row, and tool_tag_and_summary is the same
    function capture used, so this converges in one pass and a repeated sync
    writes nothing. Every row it touches comes out with a non-NULL tag (the
    mapping has a default for every tool), which is also what terminates the
    batching loop. Returns the number of rows tagged.
    """
    tagged = 0
    for table, key in TOOL_CALL_TABLES:
        select = (
            "SELECT {0}, tool_name, input_json FROM {1}"
            " WHERE tag IS NULL LIMIT {2}".format(key, table, TOOL_TAG_BATCH)
        )
        update = "UPDATE {0} SET tag = ?, summary = ? WHERE {1} = ?".format(table, key)
        while True:
            # Batched rather than fetchall()'d: a machine-wide audit holds
            # hundreds of thousands of tool calls, each with up to
            # MAX_INPUT_CHARS of input, and the first backfill must not try
            # to hold all of them at once.
            rows = conn.execute(select).fetchall()
            if not rows:
                break
            conn.executemany(
                update,
                [
                    tool_tag_and_summary(row["tool_name"], row["input_json"])
                    + (row[key],)
                    for row in rows
                ],
            )
            conn.commit()
            tagged += len(rows)
    return tagged


def finalize_db(conn, directory, classify):
    """Whole-database passes that run once per database, after the mirroring.

    One database serves every workspace, so these run once at the end of the
    sync rather than once per workspace. Returns {remarked, task_links,
    tool_tags, classified}.
    """
    # Rows written before synthetic marking existed (or before a marker
    # or system:* kind was added) converge here: every machine-generated
    # prompt is re-flagged and given its deterministic kind + title,
    # replacing any tag a classifier mistakenly assigned to machinery.
    remarked = 0
    for row in conn.execute(
        "SELECT turn_id, prompt, tag, tag_source, is_synthetic FROM turns"
        " WHERE prompt IS NOT NULL"
    ).fetchall():
        if not (row["is_synthetic"] or is_synthetic_prompt(row["prompt"])):
            continue
        kind, title = synthetic_kind(row["prompt"])
        if row["is_synthetic"] and row["tag"] == kind and row["tag_source"] == "system":
            continue
        conn.execute(
            "UPDATE turns SET is_synthetic = 1, tag = ?, title = ?,"
            " tag_source = 'system' WHERE turn_id = ?",
            [kind, title, row["turn_id"]],
        )
        remarked += 1
    conn.commit()

    # After the re-marking, so a turn only just recognised as machinery is
    # already in its final shape when its task link is read off the prompt.
    task_links = backfill_task_links(conn)
    # Independent of the classifier flag: tool-call tags are deterministic, so
    # there is no cost to keeping them complete on every sync.
    tool_tags = backfill_tool_tags(conn)

    classified = None
    if classify:
        classified = {"haiku": 0, "heuristic": 0, "synthetic": 0, "agents": 0}
        # `status != 'pending'` is load-bearing: a pending row is a prompt the
        # UserPromptSubmit hook opened whose response has not been written yet,
        # so there is nothing to classify and a tag assigned now would describe
        # an empty turn. Turns whose tag was just dropped for re-classification
        # are picked up by the same `tag IS NULL` clause.
        pending = [
            row["turn_id"]
            for row in conn.execute(
                "SELECT turn_id FROM turns WHERE tag IS NULL"
                " AND status != 'pending' ORDER BY turn_id ASC"
            ).fetchall()
        ]
        for turn_id in pending:
            try:
                result = classify_turn(conn, turn_id)
            except Exception:
                log_error(
                    directory,
                    "classify failed for turn {0}: {1}".format(
                        turn_id, traceback.format_exc().replace("\n", " | ")
                    ),
                )
                continue
            if result is not None:
                classified[result["tag_source"]] = (
                    classified.get(result["tag_source"], 0) + 1
                )

        # Subagents, same treatment and the same one-at-a-time pacing: each
        # classification is a headless `claude` run, and a machine-wide sync
        # must not fan hundreds of them out at once.
        for session_id, agent_id in [
            (row["session_id"], row["agent_id"])
            for row in conn.execute(
                "SELECT session_id, agent_id FROM agents WHERE tag IS NULL"
                " ORDER BY ts_start ASC, agent_id ASC"
            ).fetchall()
        ]:
            try:
                result = classify_agent(conn, session_id, agent_id)
            except Exception:
                log_error(
                    directory,
                    "classify failed for agent {0}: {1}".format(
                        agent_ref(session_id, agent_id),
                        traceback.format_exc().replace("\n", " | "),
                    ),
                )
                continue
            if result is not None:
                classified[result["tag_source"]] = (
                    classified.get(result["tag_source"], 0) + 1
                )
                classified["agents"] += 1
    return {
        "remarked": remarked,
        "task_links": task_links,
        "tool_tags": tool_tags,
        "classified": classified,
    }


def print_workspace_summary(out, summary) -> None:
    totals = summary["totals"]
    agents = summary["agent_totals"]
    out("audit sync: {0}\n".format(summary["workspace"]))
    out("  transcripts: {0}\n".format(summary["root"]))
    out("  database:    {0}\n".format(summary["db_file"]))
    out(
        "  sessions:    {0} found, {1} new\n".format(
            summary["sessions"], summary["new_sessions"]
        )
    )
    out(
        "  turns:       {0} added, {1} updated, {2} already present\n".format(
            totals["added"], totals["updated"], totals["present"]
        )
    )
    out("  tool_calls:  {0} written\n".format(totals["tool_calls"]))
    out(
        "  agents:      {0} new, {1} updated, {2} unchanged,"
        " {3} tool-backfilled\n".format(
            agents["new"],
            agents["updated"],
            agents["unchanged"],
            agents["backfilled"],
        )
    )
    out("  archives:    {0} written\n".format(summary["archived"]))
    out("  agent files: {0} archived\n".format(summary["agent_archived"]))
    if totals["reclassified"]:
        out(
            "  re-opened:   {0} turn(s) whose response grew since tagging\n".format(
                totals["reclassified"]
            )
        )
    if totals["relinked"]:
        out("  relinked:    {0} turn(s) given lineage uuids\n".format(totals["relinked"]))
    if totals["repaired"]:
        out(
            "  repaired:    {0} phantom turn(s) merged\n".format(totals["repaired"])
        )
    if summary["remarked"]:
        out(
            "  re-marked:   {0} turn(s) assigned a system:* kind\n".format(
                summary["remarked"]
            )
        )
    if summary["classified"] is not None:
        out(
            "  classified:  {0} haiku, {1} heuristic, {2} synthetic (untagged),"
            " {3} agent(s)\n".format(
                summary["classified"]["haiku"],
                summary["classified"]["heuristic"],
                summary["classified"]["synthetic"],
                summary["classified"].get("agents", 0),
            )
        )
    for name, count in summary["parse_errors"]:
        out("  parse errors: {0}: {1} unparseable line(s)\n".format(name, count))
    for name, message in summary["failures"]:
        out("  FAILED: {0}: {1}\n".format(name, message))


def sync_targets(args, root_dir):
    """Resolve --workspace / --all into [(project_dir, workspace)].

    Machine-wide is the model, so NEITHER flag means --all: a bare `sync`
    sweeps every workspace under the projects root. --workspace narrows the
    sweep to one project's transcripts; they still land in the same database.

    Returns (targets, skipped, exit_code). A nonzero exit code means the
    arguments named nothing syncable and the caller should stop.
    """
    if getattr(args, "all_workspaces", False) or not getattr(args, "workspace", None):
        if not root_dir.is_dir():
            sys.stderr.write(
                "audit_log: projects root is not a directory: {0}\n".format(root_dir)
            )
            return [], [], 2
        targets, skipped = discover_workspaces(root_dir)
        if not targets:
            sys.stderr.write(
                "audit_log: no syncable workspaces under {0}\n".format(root_dir)
            )
            return [], skipped, 2
        return targets, skipped, 0

    workspace = Path(args.workspace).expanduser()
    if not workspace.is_dir():
        sys.stderr.write(
            "audit_log: workspace is not a directory: {0}\n".format(workspace)
        )
        return [], [], 2
    project_dir = root_dir / project_slug(workspace)
    if not project_dir.is_dir():
        sys.stderr.write(
            "audit_log: no transcript directory for this workspace: {0}\n"
            "audit_log: (looked under {1} for slug {2})\n".format(
                project_dir, root_dir, project_slug(workspace)
            )
        )
        return [], [], 2
    return [(project_dir, workspace)], [], 0


def cmd_sync(args) -> int:
    root_dir = projects_root(args.projects_dir)
    targets, skipped, code = sync_targets(args, root_dir)
    if code:
        return code

    # Every workspace collapses into ONE database -- the machine-wide default,
    # or wherever --db / $CLAUDE_AUDIT_DB points. The schema separates
    # workspaces through sessions.workspace, and cursors live INSIDE the
    # database, so each database keeps its own independent cursor set: syncing
    # the same session into two different databases is fine, each advances its
    # own cursor and neither sees the other's.
    db_file = default_db_path(args.db)
    shared = open_db(db_file)

    summaries = []
    footer = None
    try:
        for project_dir, workspace in targets:
            try:
                summary = sync_workspace(shared, workspace, project_dir, db_file)
            except Exception:
                # One unusable workspace (an unreadable project directory, a
                # transcript that explodes the parser) must not abort a
                # machine-wide sweep.
                summaries.append(
                    {
                        "workspace": workspace,
                        "root": project_dir,
                        "db_file": db_file,
                        "sessions": 0,
                        "new_sessions": 0,
                        "totals": {key: 0 for key in TURN_TOTAL_KEYS},
                        "agent_totals": agent_counts_zero(),
                        "archived": 0,
                        "agent_archived": 0,
                        "failures": [
                            (
                                str(project_dir.name),
                                traceback.format_exc().strip().splitlines()[-1],
                            )
                        ],
                        "parse_errors": [],
                        "remarked": 0,
                        "classified": None,
                    }
                )
                continue
            summaries.append(summary)
        # DB-wide passes (re-marking synthetic turns, and classification) run
        # once at the end: one database now serves every workspace.
        footer = finalize_db(shared, db_file.parent, args.classify)
    finally:
        shared.close()

    out = sys.stdout.write
    for index, summary in enumerate(summaries):
        if index:
            out("\n")
        print_workspace_summary(out, summary)

    totals = {key: 0 for key in TURN_TOTAL_KEYS}
    agent_totals = agent_counts_zero()
    sessions = archived = agent_archived = new_sessions = 0
    failures = 0
    for summary in summaries:
        for key in totals:
            totals[key] += summary["totals"][key]
        for key in agent_totals:
            agent_totals[key] += summary["agent_totals"][key]
        sessions += summary["sessions"]
        new_sessions += summary["new_sessions"]
        archived += summary["archived"]
        agent_archived += summary["agent_archived"]
        failures += len(summary["failures"])

    out("\naudit sync totals: {0} workspace(s)\n".format(len(summaries)))
    out("  projects root: {0}\n".format(root_dir))
    out("  database:      {0}\n".format(db_file))
    out(
        "  archives:      {0}/transcripts/<workspace>-<session>.jsonl\n".format(
            db_file.parent
        )
    )
    out("  sessions:      {0} found, {1} new\n".format(sessions, new_sessions))
    out(
        "  turns:         {0} added, {1} updated, {2} already present\n".format(
            totals["added"], totals["updated"], totals["present"]
        )
    )
    out("  tool_calls:    {0} written\n".format(totals["tool_calls"]))
    out(
        "  agents:        {0} new, {1} updated, {2} unchanged,"
        " {3} tool-backfilled\n".format(
            agent_totals["new"],
            agent_totals["updated"],
            agent_totals["unchanged"],
            agent_totals["backfilled"],
        )
    )
    out(
        "  archives:      {0} transcripts, {1} agent files\n".format(
            archived, agent_archived
        )
    )
    out("  re-opened:     {0} turn(s) re-tagged after growing\n".format(
        totals["reclassified"]
    ))
    out("  relinked:      {0} turn(s) given lineage uuids\n".format(
        totals["relinked"]
    ))
    out("  repaired:      {0} phantom turn(s) merged\n".format(
        totals["repaired"]
    ))
    if footer is not None:
        out(
            "  re-marked:     {0} turn(s) assigned a system:* kind\n".format(
                footer["remarked"]
            )
        )
        out(
            "  task links:    {0} row(s) linked to a background task\n".format(
                footer["task_links"]
            )
        )
        out(
            "  tool tags:     {0} tool call(s) tagged and summarised\n".format(
                footer["tool_tags"]
            )
        )
        if footer["classified"] is not None:
            out(
                "  classified:    {0} haiku, {1} heuristic, {2} synthetic,"
                " {3} agent(s)\n".format(
                    footer["classified"]["haiku"],
                    footer["classified"]["heuristic"],
                    footer["classified"]["synthetic"],
                    footer["classified"].get("agents", 0),
                )
            )
    if skipped:
        out("  skipped:       {0} project dir(s)\n".format(len(skipped)))
        for name, reason in skipped:
            out("    {0}: {1}\n".format(name, reason))
    if failures:
        out("  failures:      {0}\n".format(failures))

    if failures and not totals["added"] and not totals["updated"]:
        return 1
    return 0


# --------------------------------------------------------------------------
# Merge (corporate restore)
# --------------------------------------------------------------------------
#
# The case this exists for: a virtual server is wiped and rebuilt, the machine
# starts logging again from nothing, and the backed-up audit database has to be
# folded back in WITHOUT losing what has been recorded since. So neither side
# is authoritative -- the merge is per-row, and for every row it keeps the more
# complete version.
#
# The one structural obstacle is turn_id: it is an AUTOINCREMENT rowid, private
# to each database, and the same turn almost certainly has different ids on the
# two sides. prompt_uuid is the real identity of a turn (it is the uuid Claude
# Code gave the prompt line), so turns are matched on it and a source ->
# destination turn_id map is built while they are matched; tool_calls.turn_id
# and agents.turn_id are rewritten through that map as they are imported.
#
# Every rule below is chosen so that merging the same source twice is a no-op:
# comparisons are strict (>), so an equal row never wins and nothing is
# rewritten on the second pass.

SESSION_MERGE_COLUMNS = (
    "session_id",
    "workspace",
    "git_branch",
    "cc_version",
    "started_at",
    "ended_at",
    "end_reason",
)
# The only session fields that can arrive late, and so are worth back-filling
# into a row the live machine already opened.
SESSION_FILL_COLUMNS = ("ended_at", "end_reason")

# Every turns column except turn_id, which is what the merge rewrites. A new
# column added here also belongs in MIGRATIONS, TURNS_REBUILD_DDL and
# TURN_COLUMN_LIST -- but a merge between mismatched schemas stays correct
# either way, because shared_columns() intersects this with both databases.
TURN_MERGE_COLUMNS = (
    "session_id",
    "prompt_id",
    "prompt_uuid",
    "ts_start",
    "ts_end",
    "duration_s",
    "model",
    "permission_mode",
    "prompt",
    "response",
    "tag",
    "title",
    "tag_source",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "thinking_tokens",
    "status",
    "is_synthetic",
    "parent_uuid",
    "last_uuid",
    "task_id",
    "interjections",
)
# Identity, not content: never overwritten on an existing destination row.
TURN_IDENTITY_COLUMNS = ("session_id", "prompt_uuid")

TOOL_CALL_MERGE_COLUMNS = (
    "turn_id",
    "seq",
    "tool_name",
    "tool_use_id",
    "input_json",
    "input_truncated",
    "result_text",
    "result_truncated",
    "is_error",
    "task_id",
    "run_ref",
    "tag",
    "summary",
)

# How finished a turn is, most finished last. A row the hooks opened but never
# filled ('pending') loses to anything; 'swept' (finalised by SessionEnd) loses
# to 'complete' (witnessed live by Stop).
STATUS_RANK = {"pending": 1, "swept": 2, "complete": 3}

MERGE_COUNT_KEYS = (
    "sessions_imported",
    "sessions_filled",
    "sessions_unchanged",
    "turns_imported",
    "turns_replaced",
    "turns_unchanged",
    "turns_unkeyed",
    "tool_calls_written",
    "agents_imported",
    "agents_replaced",
    "agents_unchanged",
    "agent_tool_calls_written",
    "cursors_imported",
)


def table_columns(conn, table):
    """The column names of `table`, or () when the table does not exist.

    A backup can predate columns the current schema has (or, after a rollback,
    carry ones it no longer has), and it is open read-only so it cannot be
    migrated. Every merge query is therefore built from the intersection of
    what both sides actually have.
    """
    try:
        rows = conn.execute("PRAGMA table_info({0})".format(table)).fetchall()
    except sqlite3.Error:
        return ()
    return tuple(row["name"] for row in rows)


def connect_readonly(uri) -> sqlite3.Connection:
    """Connect and force the file open, so a failure surfaces here."""
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except Exception:
        conn.close()
        raise
    return conn


def open_source_db(path) -> sqlite3.Connection:
    """Open the backup STRICTLY read-only. Raises on anything unusable.

    mode=ro is a real guarantee from SQLite, not a convention: the merge
    cannot alter, upgrade or WAL-checkpoint the file it was handed, whatever
    goes wrong downstream.

    The fallback matters for exactly the case this command exists for. These
    databases run in WAL mode, and a WAL reader needs a writable -shm
    companion; a backup is usually the .db file ALONE, so a plain mode=ro open
    of it fails outright with "unable to open database file". immutable=1
    promises SQLite the file cannot change and lets it read without the shared
    memory index. It is the second choice, not the first, because it also makes
    SQLite ignore any -wal that IS present -- so when the sidecars came along,
    the plain open above has already used them and seen the newest commits.
    """
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise OSError("not a file: {0}".format(resolved))
    uri = "file:{0}?mode=ro".format(quote(str(resolved.resolve())))
    try:
        conn = connect_readonly(uri)
    except sqlite3.Error:
        conn = connect_readonly(uri + "&immutable=1")
    if not table_columns(conn, "turns"):
        conn.close()
        raise sqlite3.DatabaseError(
            "no turns table: {0} is not an audit database".format(resolved)
        )
    return conn


def shared_columns(dest, source, table, wanted):
    """`wanted` narrowed to the columns both databases really have."""
    dest_cols = set(table_columns(dest, table))
    src_cols = set(table_columns(source, table))
    return [name for name in wanted if name in dest_cols and name in src_cols]


def insert_sql(table, columns) -> str:
    return "INSERT INTO {0} ({1}) VALUES ({2})".format(
        table, ", ".join(columns), ", ".join("?" * len(columns))
    )


def turn_completeness(row):
    """How much a turn row knows, as a sortable tuple.

    Ordered by what is hardest to recover: a finished status first, then
    whether it was ever classified, then the length of the response text (a
    turn whose transcript tail was lost mid-write has a shorter one).
    """
    status = row["status"] if "status" in row.keys() else None
    tag = row["tag"] if "tag" in row.keys() else None
    response = row["response"] if "response" in row.keys() else None
    return (
        STATUS_RANK.get(status or "", 0),
        1 if (tag or "").strip() else 0,
        len(response or ""),
    )


def merge_sessions(dest, source, counts) -> None:
    columns = shared_columns(dest, source, "sessions", SESSION_MERGE_COLUMNS)
    if "session_id" not in columns:
        return
    statement = insert_sql("sessions", columns)
    fills = [name for name in SESSION_FILL_COLUMNS if name in columns]

    for row in source.execute(
        "SELECT {0} FROM sessions".format(", ".join(columns))
    ).fetchall():
        session_id = row["session_id"]
        if not session_id:
            continue
        existing = dest.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if existing is None:
            dest.execute(statement, [row[name] for name in columns])
            counts["sessions_imported"] += 1
            continue
        # Present on both sides: the live row wins, except where it simply
        # does not know something yet. A session the backup saw end and the
        # live database never did gets its ending back.
        missing = {
            name: row[name]
            for name in fills
            if existing[name] is None and row[name] is not None
        }
        if not missing:
            counts["sessions_unchanged"] += 1
            continue
        dest.execute(
            "UPDATE sessions SET {0} WHERE session_id = ?".format(
                ", ".join("{0} = ?".format(name) for name in missing)
            ),
            list(missing.values()) + [session_id],
        )
        counts["sessions_filled"] += 1


def copy_tool_calls(dest, source, columns, src_turn_id, dest_turn_id) -> int:
    """Replace a destination turn's tool calls with the source turn's.

    Wholesale, exactly like the live sweep does it: tool calls have no stable
    identity of their own, so they belong to whichever side of the merge won
    the turn and are rewritten as a set.
    """
    rows = source.execute(
        "SELECT {0} FROM tool_calls WHERE turn_id = ? ORDER BY seq".format(
            ", ".join(columns)
        ),
        (src_turn_id,),
    ).fetchall()
    dest.execute("DELETE FROM tool_calls WHERE turn_id = ?", (dest_turn_id,))
    if not rows:
        return 0
    dest.executemany(
        insert_sql("tool_calls", ["turn_id"] + columns),
        [[dest_turn_id] + [row[name] for name in columns] for row in rows],
    )
    return len(rows)


def merge_turns(dest, source, counts):
    """Fold the source's turns in; returns {source turn_id: dest turn_id}."""
    columns = shared_columns(dest, source, "turns", TURN_MERGE_COLUMNS)
    if "prompt_uuid" not in columns:
        return {}
    mutable = [name for name in columns if name not in TURN_IDENTITY_COLUMNS]
    statement = insert_sql("turns", columns)
    update_sql = "UPDATE turns SET {0} WHERE turn_id = ?".format(
        ", ".join("{0} = ?".format(name) for name in mutable)
    )
    tool_columns = [
        name
        for name in shared_columns(dest, source, "tool_calls", TOOL_CALL_MERGE_COLUMNS)
        if name != "turn_id"
    ]

    id_map = {}
    for row in source.execute(
        "SELECT turn_id, {0} FROM turns".format(", ".join(columns))
    ).fetchall():
        prompt_uuid = row["prompt_uuid"]
        if not prompt_uuid:
            # Nothing to match it by, and importing it would duplicate it on
            # the next merge. These are rows a UserPromptSubmit opened and no
            # transcript sweep ever reached -- a prompt with no response.
            counts["turns_unkeyed"] += 1
            continue
        existing = dest.execute(
            "SELECT * FROM turns WHERE prompt_uuid = ?", (prompt_uuid,)
        ).fetchone()

        if existing is None:
            cursor = dest.execute(statement, [row[name] for name in columns])
            dest_turn_id = cursor.lastrowid
            id_map[row["turn_id"]] = dest_turn_id
            counts["turns_imported"] += 1
            counts["tool_calls_written"] += copy_tool_calls(
                dest, source, tool_columns, row["turn_id"], dest_turn_id
            )
            continue

        dest_turn_id = existing["turn_id"]
        id_map[row["turn_id"]] = dest_turn_id
        # Strictly more complete, or the live row stands -- which is what makes
        # a repeated merge a no-op.
        if turn_completeness(row) > turn_completeness(existing):
            dest.execute(
                update_sql, [row[name] for name in mutable] + [dest_turn_id]
            )
            counts["turns_replaced"] += 1
            counts["tool_calls_written"] += copy_tool_calls(
                dest, source, tool_columns, row["turn_id"], dest_turn_id
            )
        else:
            counts["turns_unchanged"] += 1
    return id_map


def copy_agent_tool_calls(dest, source, columns, session_id, agent_id) -> int:
    """Replace one agent's tool calls with the source agent's.

    Wholesale, exactly like copy_tool_calls does it for a turn: the rows have
    no identity of their own, they belong to whichever side of the merge won
    the agent, and they are keyed by (session_id, agent_id) -- which is the
    agents primary key and identical on both sides, so unlike turn_id nothing
    has to be remapped. `columns` empty means one of the databases predates the
    table, and then the destination's rows are left exactly as they are.
    """
    if not columns:
        return 0
    rows = source.execute(
        "SELECT {0} FROM agent_tool_calls WHERE session_id = ? AND agent_id = ?"
        " ORDER BY seq".format(", ".join(columns)),
        (session_id, agent_id),
    ).fetchall()
    dest.execute(
        "DELETE FROM agent_tool_calls WHERE session_id = ? AND agent_id = ?",
        (session_id, agent_id),
    )
    if not rows:
        return 0
    dest.executemany(
        insert_sql("agent_tool_calls", columns),
        [[row[name] for name in columns] for row in rows],
    )
    return len(rows)


def merge_agents(dest, source, id_map, counts) -> None:
    columns = shared_columns(dest, source, "agents", AGENT_COLUMNS)
    if "session_id" not in columns or "agent_id" not in columns:
        return
    statement = insert_sql("agents", columns)
    replace_sql = statement.replace("INSERT INTO", "INSERT OR REPLACE INTO", 1)
    # Keyed on the agent, so session_id/agent_id are copied across verbatim
    # rather than rewritten -- they are the join, not payload.
    atc_columns = shared_columns(
        dest, source, "agent_tool_calls", AGENT_TOOL_CALL_COLUMNS
    )
    if "session_id" not in atc_columns or "agent_id" not in atc_columns:
        atc_columns = []

    for row in source.execute(
        "SELECT {0} FROM agents".format(", ".join(columns))
    ).fetchall():
        session_id, agent_id = row["session_id"], row["agent_id"]
        if not session_id or not agent_id:
            continue
        values = {name: row[name] for name in columns}
        if "turn_id" in values:
            # Through the map, or nothing: a stale source turn_id would point
            # at an unrelated destination turn. resolve_agent_turns() re-joins
            # what it can afterwards, through session_id + prompt_id.
            values["turn_id"] = id_map.get(row["turn_id"])
        existing = dest.execute(
            "SELECT src_bytes FROM agents WHERE session_id = ? AND agent_id = ?",
            (session_id, agent_id),
        ).fetchone()
        params = [values[name] for name in columns]
        if existing is None:
            dest.execute(statement, params)
            counts["agents_imported"] += 1
            counts["agent_tool_calls_written"] += copy_agent_tool_calls(
                dest, source, atc_columns, session_id, agent_id
            )
            continue
        # src_bytes is the size of the agent transcript the row was parsed
        # from, and those files only ever grow: the bigger parse saw more.
        src_bytes = values.get("src_bytes") or 0
        if src_bytes > (existing["src_bytes"] or 0):
            dest.execute(replace_sql, params)
            counts["agents_replaced"] += 1
            # The flow follows the row: an agent replaced by the source keeps
            # the source's tool calls, never a mix of the two parses.
            counts["agent_tool_calls_written"] += copy_agent_tool_calls(
                dest, source, atc_columns, session_id, agent_id
            )
        else:
            counts["agents_unchanged"] += 1


def merge_cursors(dest, source, counts) -> None:
    """Copy only the cursors the destination has no opinion about.

    A cursor is a byte offset into a transcript that is still being appended
    to on THIS machine, so the live value is always the one to keep; a backup
    cursor is useful only for a session the live database has never swept.
    """
    columns = shared_columns(
        dest, source, "cursors", ("session_id", "byte_offset", "updated_at")
    )
    if "session_id" not in columns or "byte_offset" not in columns:
        return
    statement = insert_sql("cursors", columns).replace(
        "INSERT INTO", "INSERT OR IGNORE INTO", 1
    )
    for row in source.execute(
        "SELECT {0} FROM cursors".format(", ".join(columns))
    ).fetchall():
        if not row["session_id"]:
            continue
        cursor = dest.execute(statement, [row[name] for name in columns])
        if cursor.rowcount > 0:  # 0 == the destination already had one
            counts["cursors_imported"] += 1


def merge_databases(dest, source):
    """Fold `source` into `dest` in one transaction. Returns the counts."""
    counts = {key: 0 for key in MERGE_COUNT_KEYS}
    with dest:
        merge_sessions(dest, source, counts)
        id_map = merge_turns(dest, source, counts)
        merge_agents(dest, source, id_map, counts)
        merge_cursors(dest, source, counts)
        # Imported agents whose parent turn was not in the map (or not in the
        # source at all) can still find it by promptId now that both sides'
        # turns are in one table.
        resolve_agent_turns(dest)
    return counts


def cmd_merge(args) -> int:
    db_path = default_db_path(args.db)
    try:
        source = open_source_db(args.source)
    except Exception as exc:
        sys.stderr.write(
            "audit_log: cannot read source database {0}: {1}\n".format(args.source, exc)
        )
        return 1

    dest = None
    try:
        dest = open_db(db_path)
        counts = merge_databases(dest, source)
    except Exception:
        sys.stderr.write(
            "audit_log: merge failed: {0}\n".format(traceback.format_exc())
        )
        log_error(
            db_path.parent,
            "merge from {0} failed: {1}".format(
                args.source, traceback.format_exc().replace("\n", " | ")
            ),
        )
        return 1
    finally:
        source.close()
        if dest is not None:
            dest.close()

    out = sys.stdout.write
    out("audit merge: {0}\n".format(Path(args.source).expanduser()))
    out("  into:       {0}\n".format(db_path))
    out(
        "  sessions:   {0} imported, {1} completed, {2} unchanged\n".format(
            counts["sessions_imported"],
            counts["sessions_filled"],
            counts["sessions_unchanged"],
        )
    )
    out(
        "  turns:      {0} imported, {1} replaced, {2} unchanged\n".format(
            counts["turns_imported"],
            counts["turns_replaced"],
            counts["turns_unchanged"],
        )
    )
    if counts["turns_unkeyed"]:
        out(
            "  skipped:    {0} source turn(s) with no prompt_uuid\n".format(
                counts["turns_unkeyed"]
            )
        )
    out("  tool_calls: {0} written\n".format(counts["tool_calls_written"]))
    out(
        "  agents:     {0} imported, {1} replaced, {2} unchanged\n".format(
            counts["agents_imported"],
            counts["agents_replaced"],
            counts["agents_unchanged"],
        )
    )
    out(
        "  agent tools: {0} written\n".format(counts["agent_tool_calls_written"])
    )
    out("  cursors:    {0} imported\n".format(counts["cursors_imported"]))
    return 0


def cmd_init(args) -> int:
    db_path = default_db_path(args.db)
    conn = open_db(db_path)
    conn.close()
    json.dump({"db": str(db_path.resolve()), "schema_version": SCHEMA_VERSION}, sys.stdout)
    sys.stdout.write("\n")
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


DB_HELP = (
    "audit database (default: $CLAUDE_AUDIT_DB or ~/.claude/audit/audit.db);"
    " hook-errors.log and transcripts/ live beside it"
)


def run_subcommand(argv) -> int:
    parser = argparse.ArgumentParser(prog="audit_log.py")
    sub = parser.add_subparsers(dest="command")

    p_classify = sub.add_parser(
        "classify", help="tag and title one turn or one subagent"
    )
    p_classify.add_argument("--db", default=None, help=DB_HELP)
    # Exactly one subject per run: a turn id, or a subagent named the way the
    # agents table keys it.
    subject = p_classify.add_mutually_exclusive_group(required=True)
    subject.add_argument("--turn", type=int, help="turns.turn_id")
    subject.add_argument(
        "--agent",
        metavar="SESSION_ID:AGENT_ID",
        help="one subagent, as agents.session_id:agents.agent_id",
    )

    p_init = sub.add_parser("init", help="create or upgrade the schema")
    p_init.add_argument("--db", default=None, help=DB_HELP)

    p_merge = sub.add_parser(
        "merge", help="fold a backed-up audit database into this one"
    )
    p_merge.add_argument(
        "--from",
        dest="source",
        required=True,
        help="the backup to read (opened strictly read-only)",
    )
    p_merge.add_argument("--db", default=None, help=DB_HELP)

    p_sync = sub.add_parser(
        "sync", help="backfill the database from existing transcripts"
    )
    # Neither flag means every workspace: the audit is machine-wide.
    target = p_sync.add_mutually_exclusive_group()
    target.add_argument(
        "--workspace", help="narrow the sync to this one workspace's transcripts"
    )
    target.add_argument(
        "--all",
        dest="all_workspaces",
        action="store_true",
        help="sync every workspace found under the projects root (the default)",
    )
    p_sync.add_argument("--db", default=None, help=DB_HELP)
    p_sync.add_argument(
        "--classify",
        action="store_true",
        help="tag and title every still-untagged turn afterwards",
    )
    p_sync.add_argument(
        "--projects-dir",
        default=None,
        help="transcript root (default: $CLAUDE_AUDIT_PROJECTS_DIR or ~/.claude/projects)",
    )

    args = parser.parse_args(argv)
    return SUBCOMMANDS[args.command](args)


SUBCOMMANDS = {
    "classify": cmd_classify,
    "init": cmd_init,
    "merge": cmd_merge,
    "sync": cmd_sync,
}

HANDLERS = {
    "UserPromptSubmit": handle_prompt_submit,
    "Stop": handle_stop,
    "SessionEnd": handle_session_end,
}


def run_hook() -> int:
    """Read a hook payload from stdin and dispatch. Always exits 0."""
    try:
        payload = json.loads(sys.stdin.read())
    except Exception:
        return 0  # not a hook invocation
    if not isinstance(payload, dict):
        return 0

    try:
        handler = HANDLERS.get(payload.get("hook_event_name"))
        if handler is not None:
            handler(payload)
    except Exception:
        log_error(
            audit_dir(),
            "{0} failed: {1}".format(
                payload.get("hook_event_name"),
                traceback.format_exc().replace("\n", " | "),
            ),
        )
    return 0


def main() -> int:
    argv = sys.argv[1:]
    # Explicit subcommands are never hook invocations, so they run regardless of
    # CLAUDE_AUDIT_SKIP. This matters: the classifier is deliberately spawned
    # WITH that variable set (so the headless `claude` session it starts cannot
    # re-enter these hooks), and guarding the subcommand on it too would mean no
    # turn ever got classified.
    if argv and argv[0] in SUBCOMMANDS:
        return run_subcommand(argv)
    # Hook mode: bail out when we are running inside the classifier's own session.
    if os.environ.get("CLAUDE_AUDIT_SKIP") == "1":
        return 0
    return run_hook()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        # Absolute last resort -- a hook must never fail the turn. A CLI
        # subcommand is not a hook, so there the crash is reported honestly.
        log_error(audit_dir(), traceback.format_exc().replace("\n", " | "))
        if sys.argv[1:2] and sys.argv[1] in SUBCOMMANDS:
            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
