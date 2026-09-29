#!/usr/bin/env python3
"""Local read-only HTTP service for the Claude Code audit log (v2.0).

The viewer used to be a file you opened from disk and fed a database to by
hand. This service keeps that page exactly as it is but puts it behind a
loopback HTTP origin, so the browser can pull a consistent snapshot of the
audit database, watch it for changes, and open the files a turn touched --
none of which a `file://` page can do.

    python3 service.py serve             run in the foreground
    python3 service.py start             spawn a detached server (idempotent)
    python3 service.py status            is it up, where, which database
    python3 service.py stop              terminate the detached server
    python3 service.py install-startup   run it at login (launchd / systemd)
    python3 service.py uninstall-startup undo that
    python3 service.py print-startup     print the plist/unit without acting

Endpoints (all `X-Content-Type-Options: nosniff`)

    GET  /                the audit viewer          (../views/viewer.html)
    GET  /view            the artifact renderer     (../views/fileview.html)
    GET  /api/health      {ok, version, db, port}
    GET  /api/db          a consistent snapshot of the audit database
    GET  /api/events      SSE; `db-change` when the database moves
    GET  /api/files       files touched by Write/Edit/NotebookEdit/MultiEdit
    GET  /api/file?path=  one file's bytes, ONLY from inside a known workspace
    POST /api/merge       fold an uploaded audit database into the central one

Design rules (inherited from audit_log.py)
    * Bind 127.0.0.1 and nothing else. The hard-coded loopback address below
      is a security boundary, not a default: this process serves file bytes.
    * The database is resolved the same way the logger resolves it --
      $CLAUDE_AUDIT_DB, else ~/.claude/audit/audit.db -- and its directory
      owns the pidfile and logs, exactly as it owns hook-errors.log.
    * Stdlib only. No dependency may stand between a hook and a running
      service, because the Stop hook starts this process (see the watchdog in
      audit_log.py) and a hook must never fail the turn.
    * Reads are the rule; /api/merge is the single, deliberate exception.
      There is still no directory listing, the only path that can reach the
      filesystem is /api/file -- which serves a realpath'd file only when it
      sits inside a workspace the database itself has recorded -- and the only
      thing a write can do is hand bytes to the merge engine in audit_log.py.
      An upload never lands in a workspace: it is streamed to the service's
      own directory, validated, merged, and deleted.
"""

from __future__ import annotations

import argparse
import errno
import http.client
import importlib.util
import json
import mimetypes
import os
import plistlib
import re
import signal
import sqlite3
import stat as statmod
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

VERSION = "2.0.0"

# NOT configurable, deliberately. This process hands out file bytes; binding
# anything but loopback would publish one machine's entire audit trail --
# every prompt, every response, every touched file -- to its network.
BIND_HOST = "127.0.0.1"

DEFAULT_PORT = 4737
PORT_ENV_VAR = "CLAUDE_AUDIT_PORT"

# Same resolution as audit_log.default_db_path(). Duplicated rather than
# imported: this file is launched standalone by launchd/systemd and by a hook
# subprocess, and it must not depend on importing a 3000-line sibling.
DB_ENV_VAR = "CLAUDE_AUDIT_DB"
DEFAULT_AUDIT_DIR = Path.home() / ".claude" / "audit"
DEFAULT_DB_FILE = DEFAULT_AUDIT_DIR / "audit.db"

HERE = Path(__file__).resolve().parent
VIEWS_DIR = HERE.parent / "views"          # ../views, relative to THIS file
VIEWER_HTML = VIEWS_DIR / "viewer.html"
FILEVIEW_HTML = VIEWS_DIR / "fileview.html"

PIDFILE_NAME = "service.pid"
LOG_NAME = "service.log"
ERRLOG_NAME = "service-errors.log"

LAUNCHD_LABEL = "com.claude-audit.service"
SYSTEMD_UNIT = "claude-audit.service"

# The tools whose inputs name a file. MultiEdit and Edit use file_path;
# NotebookEdit uses notebook_path; Write uses file_path.
FILE_TOOLS = ("Write", "Edit", "NotebookEdit", "MultiEdit")
PATH_KEYS = ("file_path", "notebook_path")
FILES_LIMIT = 5000

ROOT_CACHE_TTL_S = 30.0     # how often the workspace root set is re-read
POLL_INTERVAL_S = 1.0       # SSE: how often the change signal is sampled
HEARTBEAT_S = 15.0          # SSE: comment-line keepalive
STREAM_CHUNK = 64 * 1024

START_TIMEOUT_S = 15.0
STOP_TIMEOUT_S = 10.0
HEALTH_TIMEOUT_S = 1.0

# ---- /api/merge ----------------------------------------------------------
AUDIT_LOG_SCRIPT = HERE / "audit_log.py"    # the merge engine lives there

MERGE_MAX_BYTES = 512 * 1024 * 1024         # 512 MiB; anything larger is 413
MERGE_UPLOAD_DIRNAME = "merge-incoming"     # a subdir of the audit directory
MERGE_STALE_UPLOAD_S = 3600.0               # sweep leftovers older than this
# How long a second upload waits for the first merge to finish before it is
# told 409. Short on purpose: the merge itself is seconds, and a caller that
# has already spent a minute uploading would rather be told to retry than sit
# on a socket for the length of somebody else's transaction.
MERGE_LOCK_WAIT_S = 5.0
MERGE_HEADER_LIMIT = 64 * 1024              # multipart part headers, at most
MERGE_READ_CHUNK = 1024 * 1024

SQLITE_MAGIC = b"SQLite format 3\x00"
# The tables a file must have before it is allowed anywhere near the merge.
REQUIRED_SOURCE_TABLES = ("sessions", "turns")

# Every route that answers GET/HEAD and nothing else -- so a POST to one of
# them is told 405 rather than 404, which is the difference between "you used
# the wrong verb" and "you misread the docs".
GET_ONLY_ROUTES = frozenset(
    (
        "/",
        "/index.html",
        "/viewer.html",
        "/view",
        "/view.html",
        "/fileview.html",
        "/api/health",
        "/api/db",
        "/api/events",
        "/api/files",
        "/api/file",
        "/favicon.ico",
    )
)

# A truncated tool input (audit_log caps input_json at 10k chars) is no longer
# parseable JSON, but the path is almost always in the first few hundred
# bytes -- so a failed json.loads falls back to this.
PATH_RE = re.compile(r'"(?:file_path|notebook_path)"\s*:\s*"((?:[^"\\]|\\.)*)"')

TEXTUAL_SUFFIXES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".py": "text/x-python",
    ".rs": "text/plain",
    ".go": "text/plain",
    ".ts": "text/plain",
    ".tsx": "text/plain",
    ".jsx": "text/plain",
    ".toml": "text/plain",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".ini": "text/plain",
    ".cfg": "text/plain",
    ".sh": "text/x-shellscript",
    ".zsh": "text/x-shellscript",
    ".sql": "text/plain",
    ".jsonl": "text/plain",
    ".env": "text/plain",
}


# --------------------------------------------------------------------------
# Paths, config, small helpers
# --------------------------------------------------------------------------


def default_db_path(override=None) -> Path:
    """The database this invocation reads. See audit_log.default_db_path."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get(DB_ENV_VAR)
    if env and env.strip():
        return Path(env.strip()).expanduser()
    return DEFAULT_DB_FILE


def audit_dir(db_override=None) -> Path:
    """The directory owning a database's sidecars -- pidfile, logs."""
    return default_db_path(db_override).parent


def service_port(override=None) -> int:
    """Port precedence: an explicit --port, then $CLAUDE_AUDIT_PORT, then 4737."""
    for candidate in (override, os.environ.get(PORT_ENV_VAR)):
        if candidate is None or str(candidate).strip() == "":
            continue
        try:
            port = int(str(candidate).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= port <= 65535:
            return port
    return DEFAULT_PORT


def pidfile_path(db_override=None) -> Path:
    return audit_dir(db_override) / PIDFILE_NAME


def log_paths(db_override=None):
    directory = audit_dir(db_override)
    return directory / LOG_NAME, directory / ERRLOG_NAME


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_error(directory, message) -> None:
    """Best-effort error trail beside the database. Must never raise."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ERRLOG_NAME).open("a", encoding="utf-8") as fh:
            fh.write("{0} {1}\n".format(utcnow(), message))
    except Exception:
        pass


def path_is_within(path: Path, parent: Path) -> bool:
    """True when `path` is `parent` or lives beneath it. Both already real."""
    try:
        return path == parent or parent in path.parents
    except Exception:
        return False


# --------------------------------------------------------------------------
# Workspace roots -- the whole of /api/file's authority
# --------------------------------------------------------------------------


class WorkspaceRoots:
    """The set of directories /api/file is allowed to serve from.

    Nothing here is configured: the roots ARE the workspaces the audit
    database has recorded in sessions.workspace, realpath'd. A file the audit
    never saw a session in is not servable, full stop. The set is re-read on a
    short TTL so a session started after the server came up becomes visible
    without a restart.
    """

    def __init__(self, db_path: Path, ttl=ROOT_CACHE_TTL_S):
        self.db_path = Path(db_path)
        self.ttl = ttl
        self._roots = ()
        self._loaded_at = 0.0
        self._lock = threading.Lock()

    def _load(self):
        roots = []
        if not self.db_path.is_file():
            return ()
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("PRAGMA busy_timeout=5000")
            rows = conn.execute(
                "SELECT DISTINCT workspace FROM sessions"
                " WHERE workspace IS NOT NULL AND workspace != ''"
            ).fetchall()
        except sqlite3.Error:
            return self._roots  # keep whatever we had; a locked DB is not a revocation
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

        for (workspace,) in rows:
            try:
                real = Path(os.path.realpath(str(workspace)))
            except (OSError, ValueError):
                continue
            # A workspace of "/" would make every file on the machine
            # servable. Refuse it rather than honour it.
            if str(real) == os.sep or not real.is_dir():
                continue
            roots.append(real)
        return tuple(sorted(set(roots)))

    def roots(self):
        now = time.monotonic()
        with self._lock:
            if now - self._loaded_at >= self.ttl or not self._roots:
                self._roots = self._load()
                self._loaded_at = now
            return self._roots

    def allows(self, real_path: Path) -> bool:
        return any(path_is_within(real_path, root) for root in self.roots())


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """A connection used for SELECTs only. Never writes, never migrates."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def snapshot_db(db_path: Path) -> Path:
    """A consistent copy of the audit database, as a temp file.

    Straight-copying a live WAL database can capture a half-applied write (or
    miss the -wal entirely) and hand the browser a file sql.js refuses. The
    backup API takes the same lock SQLite itself takes, so the copy is always
    a coherent point in time. The caller owns the returned file and must
    unlink it.
    """
    handle, tmp_name = tempfile.mkstemp(prefix="claude-audit-snapshot-", suffix=".db")
    os.close(handle)
    tmp_path = Path(tmp_name)
    src = dst = None
    try:
        src = open_readonly(db_path)
        dst = sqlite3.connect(str(tmp_path))
        src.backup(dst)
        dst.commit()
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    finally:
        for conn in (dst, src):
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
    return tmp_path


def change_signal(conn, db_path: Path):
    """A cheap value that moves whenever the database does.

    PRAGMA data_version is SQLite's own answer to "has another connection
    committed since I last looked" -- it is the reliable half. The file's
    mtime/size (database and -wal alike) are the fallback for the cases
    data_version cannot see: a file swapped underneath us, a copy restored
    over the top, a database that has just appeared.
    """
    version = None
    try:
        row = conn.execute("PRAGMA data_version").fetchone()
        version = row[0] if row else None
    except sqlite3.Error:
        version = None

    stamps = []
    for path in (db_path, Path(str(db_path) + "-wal")):
        try:
            info = path.stat()
            stamps.append((round(info.st_mtime, 3), info.st_size))
        except OSError:
            stamps.append(None)
    return (version, tuple(stamps))


def extract_paths(input_json, tool_name):
    """Every file path named by one tool call's stored input.

    Returns a list (MultiEdit names one file, but an unexpected shape may name
    several and dropping them would be worse than listing them).
    """
    if not input_json:
        return []
    found = []
    try:
        parsed = json.loads(input_json)
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        for key in PATH_KEYS:
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                found.append(value)
    if not found:
        # Truncated input (see PATH_RE): scrape the path out of the prefix.
        for match in PATH_RE.finditer(input_json):
            try:
                value = json.loads('"{0}"'.format(match.group(1)))
            except ValueError:
                value = match.group(1)
            if isinstance(value, str) and value.strip():
                found.append(value)
    return found


def stat_entry(path_text, cache):
    """(exists, size, mtime) for a path, memoised within one request."""
    if path_text in cache:
        return cache[path_text]
    result = (False, None, None)
    try:
        info = os.stat(path_text)
        result = (
            True,
            int(info.st_size),
            datetime.fromtimestamp(info.st_mtime, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
    except OSError:
        pass
    cache[path_text] = result
    return result


def list_touched_files(db_path: Path, turn_id=None):
    """Files touched by file-writing tools, newest turn first.

    De-duplicated on (path, turn_id, tool_name): a turn that edits one file
    six times is one line in the answer, but the same file edited in two turns
    stays two entries, because the turn is what the caller is looking at.
    """
    if not db_path.is_file():
        return []
    placeholders = ", ".join("?" * len(FILE_TOOLS))
    sql = (
        "SELECT turn_id, tool_name, input_json FROM tool_calls"
        " WHERE tool_name IN ({0})".format(placeholders)
    )
    params = list(FILE_TOOLS)
    if turn_id is not None:
        sql += " AND turn_id = ?"
        params.append(turn_id)
    sql += " ORDER BY turn_id DESC, seq ASC LIMIT ?"
    params.append(FILES_LIMIT)

    conn = open_readonly(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass

    seen = set()
    cache = {}
    entries = []
    for row_turn, tool_name, input_json in rows:
        for path_text in extract_paths(input_json, tool_name):
            key = (path_text, row_turn, tool_name)
            if key in seen:
                continue
            seen.add(key)
            exists, size, mtime = stat_entry(path_text, cache)
            entries.append(
                {
                    "path": path_text,
                    "turn_id": row_turn,
                    "tool_name": tool_name,
                    "exists": exists,
                    "size": size,
                    "mtime": mtime,
                }
            )
    return entries


def guess_content_type(path: Path) -> str:
    """A sensible Content-Type, biased towards "this is text"."""
    suffix = path.suffix.lower()
    if suffix in TEXTUAL_SUFFIXES:
        return "{0}; charset=utf-8".format(TEXTUAL_SUFFIXES[suffix])
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        if guessed.startswith("text/") or guessed in (
            "application/javascript",
            "application/json",
            "application/xml",
        ):
            return "{0}; charset=utf-8".format(guessed)
        return guessed
    # Unknown extension: sniff. NUL bytes or undecodable content mean binary.
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
        if b"\x00" not in head:
            head.decode("utf-8")
            return "text/plain; charset=utf-8"
    except (OSError, UnicodeDecodeError):
        pass
    return "application/octet-stream"


# --------------------------------------------------------------------------
# /api/merge -- the one write
# --------------------------------------------------------------------------
#
# The dedup rules are not reimplemented here, and must never be: this endpoint
# is a transport in front of audit_log.merge_databases(), the exact function
# `audit_log.py merge --from X --db Y` calls. audit_log.py is imported as a
# module rather than shelled out to because it factors the merge cleanly --
# open_source_db() does the read-only open, merge_databases() returns a counts
# dict -- so the alternative would mean scraping that dict back out of the
# human-readable summary the CLI prints. Parsing prose to recover numbers the
# function already returned is how the two copies drift apart.
#
# The import is lazy for the reason the header of this file gives: a hook
# spawns this process, and `serve` must keep starting even if the sibling is
# missing or unparseable. Nothing but a merge request pays for it.

_audit_log_module = None
_audit_log_lock = threading.Lock()

# One merge at a time, process-wide. The merge is a single transaction against
# the central database; two of them interleaved would race on the same rows and
# the turn_id map one of them built. The lock is taken AFTER the upload is
# streamed and validated, so a slow uploader never blocks a fast one -- only
# the seconds of actual merging are serialised.
_merge_lock = threading.Lock()


class MergeRequestError(Exception):
    """A request this endpoint refuses, with the status to refuse it with."""

    def __init__(self, status, message):
        Exception.__init__(self, message)
        self.status = status
        self.message = message


def load_audit_log():
    """Import the sibling audit_log.py, once, by path.

    By path rather than `import audit_log` because sys.path[0] is not
    dependable for a process launched by launchd, systemd or a hook: the file
    that must be loaded is the one beside THIS file, never whatever else on
    the path happens to answer to that name.
    """
    global _audit_log_module
    with _audit_log_lock:
        if _audit_log_module is not None:
            return _audit_log_module
        if not AUDIT_LOG_SCRIPT.is_file():
            raise MergeRequestError(503, "merge engine is unavailable")
        try:
            spec = importlib.util.spec_from_file_location(
                "claude_audit_log", str(AUDIT_LOG_SCRIPT)
            )
            module = importlib.util.module_from_spec(spec)
            # Registered before exec so a self-referential import inside the
            # module finds the half-built one instead of loading it twice.
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop("claude_audit_log", None)
            log_error(
                audit_dir(),
                "cannot import merge engine: {0}".format(
                    traceback.format_exc().replace("\n", " | ")
                ),
            )
            raise MergeRequestError(503, "merge engine is unavailable")
        for name in ("open_source_db", "open_db", "merge_databases", "table_columns"):
            if not hasattr(module, name):
                raise MergeRequestError(503, "merge engine is unavailable")
        _audit_log_module = module
        return module


def upload_dir(db_path: Path) -> Path:
    """Where an upload is spooled: the service's own dir, never a workspace.

    Beside the database rather than in it -- a subdirectory, so a half-written
    upload can never be mistaken for the audit database itself -- and never in
    the system temp dir, because a 512 MiB spool belongs on the volume that
    already holds the thing it is being merged into.
    """
    directory = Path(db_path).parent / MERGE_UPLOAD_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sweep_stale_uploads(directory: Path) -> None:
    """Best-effort: drop spool files a killed process never cleaned up."""
    cutoff = time.time() - MERGE_STALE_UPLOAD_S
    try:
        entries = list(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            pass


def parse_content_type(raw):
    """('multipart/form-data', {'boundary': b'...'}) from a header value."""
    if not raw:
        return "", {}
    parts = str(raw).split(";")
    mime = parts[0].strip().lower()
    params = {}
    for item in parts[1:]:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        params[key.strip().lower()] = value
    return mime, params


class BodyReader:
    """Reads at most Content-Length bytes off the wire, and not one more."""

    def __init__(self, rfile, length):
        self.rfile = rfile
        self.remaining = int(length)

    def read(self, size):
        if self.remaining <= 0:
            return b""
        chunk = self.rfile.read(min(size, self.remaining))
        self.remaining -= len(chunk)
        return chunk


def spool_raw_body(reader, out_fh) -> int:
    """Stream an application/octet-stream body to disk. Returns bytes written."""
    total = 0
    while True:
        chunk = reader.read(MERGE_READ_CHUNK)
        if not chunk:
            break
        out_fh.write(chunk)
        total += len(chunk)
    return total


def spool_multipart_body(reader, out_fh, boundary: bytes) -> int:
    """Stream the FIRST part of a multipart body to disk.

    A hand-rolled parser rather than email.parser because that one wants the
    whole message in memory, and the whole message here can be half a gigabyte.
    The rule it implements is the one from RFC 2046: a part ends at the CRLF
    that precedes the next `--boundary`, and that CRLF belongs to the
    delimiter, not to the file.

    Exactly one file field is expected. Anything after the first part is read
    off the wire and discarded rather than refused -- a browser that appends a
    stray text field should not cost the caller the upload.
    """
    delim = b"--" + boundary
    buf = b""

    # 1. The first part's headers: everything up to the blank line after the
    #    opening delimiter. The preamble before it is discarded, per the RFC.
    body_start = -1
    while body_start < 0:
        start = buf.find(delim)
        if start >= 0:
            blank = buf.find(b"\r\n\r\n", start)
            if blank >= 0:
                body_start = blank + 4
                break
        if len(buf) > MERGE_HEADER_LIMIT:
            raise MergeRequestError(400, "malformed multipart body")
        chunk = reader.read(MERGE_READ_CHUNK)
        if not chunk:
            raise MergeRequestError(400, "malformed multipart body")
        buf += chunk

    # 2. The part's bytes, holding back enough of a tail that a delimiter
    #    straddling two chunks is still found whole.
    keep = len(delim) + 4
    pending = buf[body_start:]
    total = 0
    closed = False
    while True:
        index = pending.find(delim)
        if index >= 0:
            end = index - 2          # drop the CRLF owned by the delimiter
            if end >= 0:
                out_fh.write(pending[:end])
                total += end
            else:
                # The delimiter's CRLF was written in an earlier pass: take it
                # back off the file rather than leave two stray bytes.
                total = max(0, total + end)
                out_fh.flush()
                out_fh.truncate(total)
                out_fh.seek(total)
            closed = True
            break
        if len(pending) > keep:
            out_fh.write(pending[:-keep])
            total += len(pending) - keep
            pending = pending[-keep:]
        chunk = reader.read(MERGE_READ_CHUNK)
        if not chunk:
            break
        pending += chunk

    if not closed:
        raise MergeRequestError(400, "multipart body ended before its boundary")

    while reader.read(MERGE_READ_CHUNK):     # drain the trailing parts
        pass
    return total


def validate_source_db(audit_log, path: Path):
    """Open an uploaded file as an audit database, or refuse it.

    Three gates, cheapest first: the magic header, a read-only open, and the
    core tables. open_source_db() is the CLI's own opener -- URI mode=ro with
    the immutable=1 fallback a WAL database uploaded without its sidecars
    needs -- so nothing this endpoint accepts could be opened read-write.
    """
    try:
        with path.open("rb") as fh:
            header = fh.read(len(SQLITE_MAGIC))
    except OSError:
        raise MergeRequestError(400, "uploaded file could not be read back")
    if header != SQLITE_MAGIC:
        raise MergeRequestError(400, "not a SQLite database (bad magic header)")

    try:
        conn = audit_log.open_source_db(str(path))
    except Exception:
        # The message would name the spool path and SQLite internals; the
        # caller gets the fact, the log beside the database gets the detail.
        log_error(
            audit_dir(),
            "merge: unreadable upload: {0}".format(
                traceback.format_exc().replace("\n", " | ")
            ),
        )
        raise MergeRequestError(
            400, "file is not a readable SQLite database"
        )

    missing = [
        name
        for name in REQUIRED_SOURCE_TABLES
        if not audit_log.table_columns(conn, name)
    ]
    if missing:
        try:
            conn.close()
        except sqlite3.Error:
            pass
        raise MergeRequestError(
            400,
            "not an audit database: missing table(s) {0}".format(
                ", ".join(missing)
            ),
        )
    return conn


def run_merge(audit_log, db_path: Path, source_conn):
    """merge_databases() against the central DB. Returns the CLI's counts."""
    dest = None
    try:
        dest = audit_log.open_db(db_path)
        return audit_log.merge_databases(dest, source_conn)
    finally:
        if dest is not None:
            try:
                dest.close()
            except sqlite3.Error:
                pass


def merge_report(counts, received, elapsed, db_path: Path):
    """The counts dict, rearranged into per-table {imported, replaced, skipped}.

    The mapping, once, here:
      * sessions   -- `replaced` is a back-fill. A session present on both
                      sides keeps the live row; only ended_at/end_reason are
                      taken from the upload, and only when the live row has
                      none. `skipped` is a row the merge left exactly as it was.
      * turns      -- `skipped` counts both the rows the live database already
                      had more completely AND the rows with no prompt_uuid,
                      which have no identity to match on and are never
                      imported; `unkeyed` breaks the second group out.
      * tool_calls and agent_tool_calls have no identity of their own: they
                      travel with the turn or agent that won, and are rewritten
                      as a set. Every row written is therefore an insert, so
                      `imported` carries the count and `written` repeats it.
      * cursors    -- imported only where the destination had no opinion.
    """
    def triple(imported=0, replaced=0, skipped=0, **extra):
        row = {"imported": imported, "replaced": replaced, "skipped": skipped}
        row.update(extra)
        return row

    written_tools = counts.get("tool_calls_written", 0)
    written_agent_tools = counts.get("agent_tool_calls_written", 0)
    tables = {
        "sessions": triple(
            imported=counts.get("sessions_imported", 0),
            replaced=counts.get("sessions_filled", 0),
            skipped=counts.get("sessions_unchanged", 0),
        ),
        "turns": triple(
            imported=counts.get("turns_imported", 0),
            replaced=counts.get("turns_replaced", 0),
            skipped=counts.get("turns_unchanged", 0)
            + counts.get("turns_unkeyed", 0),
            unkeyed=counts.get("turns_unkeyed", 0),
        ),
        "tool_calls": triple(imported=written_tools, written=written_tools),
        "agents": triple(
            imported=counts.get("agents_imported", 0),
            replaced=counts.get("agents_replaced", 0),
            skipped=counts.get("agents_unchanged", 0),
        ),
        "agent_tool_calls": triple(
            imported=written_agent_tools, written=written_agent_tools
        ),
        "cursors": triple(imported=counts.get("cursors_imported", 0)),
    }
    totals = {
        key: sum(table[key] for table in tables.values())
        for key in ("imported", "replaced", "skipped")
    }
    totals["changed"] = totals["imported"] + totals["replaced"]
    return {
        "ok": True,
        "db": str(db_path),
        "received_bytes": received,
        "tables": tables,
        "totals": totals,
        "elapsed_s": round(elapsed, 3),
        # The engine's own vocabulary, unrearranged, for anything that would
        # rather read the CLI's numbers than this endpoint's summary of them.
        "counts": dict(counts),
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


class AuditHandler(BaseHTTPRequestHandler):
    """The whole exposure surface. Every method here is a read."""

    server_version = "claude-audit/{0}".format(VERSION)
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # Injected by build_server().
    db_path = DEFAULT_DB_FILE
    port = DEFAULT_PORT
    roots = None

    # ---- plumbing ------------------------------------------------------

    def end_headers(self):
        # Every response, including the ones BaseHTTPRequestHandler generates
        # for itself. Nothing this server returns should ever be sniffed into
        # a type it did not declare.
        self.send_header("X-Content-Type-Options", "nosniff")
        BaseHTTPRequestHandler.end_headers(self)

    def log_message(self, fmt, *args):
        sys.stderr.write(
            "{0} {1} {2}\n".format(utcnow(), self.address_string(), fmt % args)
        )

    def _send_headers(self, status, ctype, length=None, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self._started = True

    def _send_bytes(self, status, ctype, body: bytes, extra=None):
        self._send_headers(status, ctype, len(body), extra)
        if not self._head:
            self.wfile.write(body)

    def _send_json(self, status, obj, extra=None):
        body = json.dumps(obj, default=str).encode("utf-8")
        headers = {"Cache-Control": "no-store"}
        headers.update(extra or {})
        self._send_bytes(status, "application/json; charset=utf-8", body, headers)

    def _send_error_json(self, status, message):
        self._send_json(status, {"error": message, "status": status})

    def _send_view(self, path: Path):
        try:
            body = path.read_bytes()
        except OSError:
            self._send_error_json(
                404, "view not found: {0}".format(path.name)
            )
            return
        self._send_bytes(
            200,
            "text/html; charset=utf-8",
            body,
            {"Cache-Control": "no-store"},
        )

    # ---- routing -------------------------------------------------------

    def do_GET(self):
        self._head = False
        self._dispatch("GET")

    def do_HEAD(self):
        self._head = True
        self._dispatch("GET")

    def do_POST(self):
        self._head = False
        self._dispatch("POST")

    def _dispatch(self, method):
        self._started = False
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if method == "POST":
                self._route_post(route)
            else:
                self._route_get(route, query)
        except MergeRequestError as exc:
            # A refusal this endpoint chose: the status and the sentence are
            # both deliberate, and neither names an internal path.
            if not self._started:
                self._send_error_json(exc.status, exc.message)
            # The rest of the upload may still be in flight and there is no
            # point draining half a gigabyte to keep the socket reusable.
            self.close_connection = True
            return
        except (BrokenPipeError, ConnectionResetError):
            return  # the browser walked away mid-response; not our problem
        except Exception:
            # A traceback names source paths and internals. It goes to the log
            # beside the database, never down the wire.
            log_error(
                audit_dir(),
                "{0} failed: {1}".format(
                    route, traceback.format_exc().replace("\n", " | ")
                ),
            )
            if self._started:
                self.close_connection = True
                return
            try:
                self._send_error_json(500, "internal server error")
            except Exception:
                self.close_connection = True
            if method == "POST":
                # Same reasoning as the MergeRequestError arm: an undrained
                # request body makes the next request on this socket garbage.
                self.close_connection = True

    def _route_get(self, route, query):
        if route in ("/", "/index.html", "/viewer.html"):
            self._send_view(VIEWER_HTML)
        elif route in ("/view", "/view.html", "/fileview.html"):
            self._send_view(FILEVIEW_HTML)
        elif route == "/api/health":
            self._api_health()
        elif route == "/api/db":
            self._api_db()
        elif route == "/api/events":
            self._api_events()
        elif route == "/api/files":
            self._api_files(query)
        elif route == "/api/file":
            self._api_file(query)
        elif route == "/favicon.ico":
            self._send_bytes(200, "image/svg+xml", FAVICON_SVG)
        elif route == "/api/merge":
            self._send_method_not_allowed("POST")
        else:
            self._send_error_json(404, "no such endpoint")

    def _route_post(self, route):
        """POST reaches exactly one endpoint. Everything else is refused."""
        if route == "/api/merge":
            self._api_merge()
        elif route in GET_ONLY_ROUTES:
            self._send_method_not_allowed("GET, HEAD")
        else:
            self._send_error_json(404, "no such endpoint")

    def _send_method_not_allowed(self, allow):
        self._send_json(
            405,
            {"error": "method not allowed", "status": 405, "allow": allow},
            {"Allow": allow},
        )

    # ---- endpoints -----------------------------------------------------

    def _api_health(self):
        self._send_json(
            200,
            {
                "ok": True,
                "version": VERSION,
                "db": str(self.db_path),
                "port": self.port,
            },
        )

    def _api_db(self):
        if not self.db_path.is_file():
            self._send_error_json(404, "no audit database at {0}".format(self.db_path))
            return
        snapshot = snapshot_db(self.db_path)
        try:
            size = snapshot.stat().st_size
            self._send_headers(
                200,
                "application/octet-stream",
                size,
                {
                    "Cache-Control": "no-store",
                    "Content-Disposition": 'attachment; filename="audit.db"',
                },
            )
            if self._head:
                return
            with snapshot.open("rb") as fh:
                while True:
                    chunk = fh.read(STREAM_CHUNK)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        finally:
            try:
                snapshot.unlink()
            except OSError:
                pass

    def _api_events(self):
        """Server-sent events. One long-lived response per browser tab.

        SSE rather than websockets on purpose: the payload is a single
        one-way "something changed, re-fetch" ping, EventSource reconnects by
        itself, and it costs no dependency -- which matters, because the whole
        service is stdlib.
        """
        self._send_headers(
            200,
            "text/event-stream; charset=utf-8",
            None,
            {
                "Cache-Control": "no-store",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )
        self.close_connection = True
        if self._head:
            return

        conn = None
        try:
            if self.db_path.is_file():
                conn = open_readonly(self.db_path)
            last = change_signal(conn, self.db_path) if conn else None
            self._sse_write(": connected {0}\n\n".format(utcnow()))
            last_beat = time.monotonic()
            while True:
                time.sleep(POLL_INTERVAL_S)
                if conn is None and self.db_path.is_file():
                    conn = open_readonly(self.db_path)      # it appeared
                    last = change_signal(conn, self.db_path)
                    self._sse_write(
                        "event: db-change\ndata: {0}\n\n".format(
                            json.dumps({"at": utcnow(), "reason": "db-appeared"})
                        )
                    )
                    last_beat = time.monotonic()
                    continue
                if conn is not None:
                    current = change_signal(conn, self.db_path)
                    if current != last:
                        last = current
                        self._sse_write(
                            "event: db-change\ndata: {0}\n\n".format(
                                json.dumps({"at": utcnow(), "reason": "data-version"})
                            )
                        )
                        last_beat = time.monotonic()
                        continue
                now = time.monotonic()
                if now - last_beat >= HEARTBEAT_S:
                    self._sse_write(": keepalive {0}\n\n".format(utcnow()))
                    last_beat = now
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            return  # client closed the tab: the normal way this ends
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    def _sse_write(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def _api_files(self, query):
        raw_turn = (query.get("turn") or [None])[0]
        turn_id = None
        if raw_turn not in (None, ""):
            try:
                turn_id = int(raw_turn)
            except (TypeError, ValueError):
                self._send_error_json(400, "turn must be an integer")
                return
        entries = list_touched_files(self.db_path, turn_id)
        self._send_json(
            200, {"files": entries, "count": len(entries), "turn": turn_id}
        )

    def _api_file(self, query):
        """The only endpoint that touches the filesystem.

        The check is: realpath the request, then require the result to sit
        inside a realpath'd workspace root taken from the database. Doing it
        on the canonical path is the point -- `..`, a symlink out of the
        workspace and a bare `/etc/passwd` all collapse to the same test, and
        all three fail it.
        """
        raw = (query.get("path") or [None])[0]
        if not raw:
            self._send_error_json(400, "path is required")
            return
        try:
            real = Path(os.path.realpath(os.path.expanduser(raw)))
        except (OSError, ValueError):
            self._send_error_json(400, "unreadable path")
            return

        if not self.roots.allows(real):
            # Deliberately uninformative and identical for "outside the
            # workspaces" and "does not exist": a 403/404 split here would
            # turn this endpoint into a filesystem oracle.
            self._send_error_json(403, "path is outside every audited workspace")
            return

        try:
            info = os.stat(str(real))
        except OSError:
            self._send_error_json(404, "no such file")
            return
        if statmod.S_ISDIR(info.st_mode):
            self._send_error_json(403, "directories are not served")
            return
        if not statmod.S_ISREG(info.st_mode):
            self._send_error_json(403, "not a regular file")
            return

        ctype = guess_content_type(real)
        self._send_headers(
            200,
            ctype,
            info.st_size,
            {
                "Cache-Control": "no-store",
                "Last-Modified": formatdate(info.st_mtime, usegmt=True),
                # Defence in depth for the Raw link: opened directly in a tab,
                # a served .html would otherwise run on this origin and could
                # read /api/db. Under CSP sandbox it runs in an opaque origin,
                # exactly as it does inside fileview's iframe.
                "Content-Security-Policy": "sandbox allow-scripts allow-downloads",
            },
        )
        if self._head:
            return
        with real.open("rb") as fh:
            while True:
                chunk = fh.read(STREAM_CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ---- the one write -------------------------------------------------

    def _reject_cross_site(self):
        """Refuse a POST a foreign page made the browser send.

        Loopback is not an authorisation boundary in a browser: any page in
        any tab can post a multipart form to http://127.0.0.1:4737 without
        tripping CORS, because a form submission is not a request the browser
        preflights. It cannot READ the answer, but this endpoint's effect --
        folding an attacker-chosen database into the audit log -- happens
        regardless. Two checks close it, and neither inconveniences curl,
        which sends neither header: an Origin that is not this service's own,
        and Sec-Fetch-Site saying the request came from another site.
        """
        origin = self.headers.get("Origin")
        if origin:
            allowed = {
                "http://{0}:{1}".format(host, self.port)
                for host in (BIND_HOST, "localhost")
            }
            if origin not in allowed:
                raise MergeRequestError(403, "cross-origin upload refused")
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site and site not in ("same-origin", "none"):
            raise MergeRequestError(403, "cross-site upload refused")

    def _merge_body_length(self):
        """The declared body size, refused early when it cannot be honoured."""
        if (self.headers.get("Transfer-Encoding") or "").strip().lower():
            # Chunked would mean streaming without knowing the size, and the
            # cap below is the whole point. Every client that matters here
            # (curl, fetch with a Blob, requests) sends a length.
            raise MergeRequestError(411, "Content-Length is required")
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise MergeRequestError(411, "Content-Length is required")
        try:
            length = int(str(raw).strip())
        except (TypeError, ValueError):
            raise MergeRequestError(400, "Content-Length is not a number")
        if length < 0:
            raise MergeRequestError(400, "Content-Length is not a number")
        if length == 0:
            raise MergeRequestError(400, "empty body")
        if length > MERGE_MAX_BYTES:
            # Refused on the declared size, before a single byte is spooled.
            raise MergeRequestError(
                413,
                "upload exceeds the {0} MiB limit".format(
                    MERGE_MAX_BYTES // (1024 * 1024)
                ),
            )
        return length

    def _api_merge(self):
        """Fold an uploaded audit database into the central one.

        The shape of the thing: spool to disk, validate, take the lock, merge,
        always delete the spool. Nothing is held in memory, nothing is written
        outside the service's own directory, and the merging itself is done by
        audit_log.merge_databases() -- the same function, on the same rows,
        with the same most-complete-wins rules as `audit_log.py merge`.

        The viewer needs no help noticing: the merge commits on a connection
        of its own, which moves both PRAGMA data_version and the file's mtime,
        so the /api/events poller emits `db-change` on its next tick exactly
        as it does for a hook's write.
        """
        started = time.monotonic()
        self._reject_cross_site()
        length = self._merge_body_length()

        mime, params = parse_content_type(self.headers.get("Content-Type"))
        boundary = None
        if mime == "multipart/form-data":
            raw_boundary = params.get("boundary")
            if not raw_boundary:
                raise MergeRequestError(400, "multipart body has no boundary")
            boundary = raw_boundary.encode("utf-8", "replace")
        # Any other content type is taken at its word and treated as raw
        # bytes. Being lenient costs nothing: the magic-header check below is
        # what actually decides, and a caller who sent the wrong type but the
        # right bytes should not be made to guess why.

        audit_log = load_audit_log()
        directory = upload_dir(self.db_path)
        sweep_stale_uploads(directory)
        handle, tmp_name = tempfile.mkstemp(
            prefix="upload-", suffix=".db", dir=str(directory)
        )
        tmp_path = Path(tmp_name)
        source = None
        try:
            reader = BodyReader(self.rfile, length)
            with os.fdopen(handle, "wb") as out_fh:
                if boundary is None:
                    received = spool_raw_body(reader, out_fh)
                    if reader.remaining > 0:
                        # A client that hung up mid-upload leaves a file that
                        # can still be a structurally valid SQLite prefix, and
                        # merging half a database is worse than merging none.
                        # The multipart path needs no such check: its closing
                        # delimiter is a stronger end-of-file than a count.
                        raise MergeRequestError(
                            400, "upload ended before Content-Length"
                        )
                else:
                    received = spool_multipart_body(reader, out_fh, boundary)
            if received <= 0:
                raise MergeRequestError(400, "no database in the request body")
            if received > MERGE_MAX_BYTES:
                raise MergeRequestError(
                    413,
                    "upload exceeds the {0} MiB limit".format(
                        MERGE_MAX_BYTES // (1024 * 1024)
                    ),
                )

            source = validate_source_db(audit_log, tmp_path)

            # Serialised here and not a line earlier: the upload is the slow
            # part and it does not touch the central database. A caller that
            # arrives while another merge is committing waits a few seconds
            # and is then told 409 -- a retry it can decide about, rather than
            # an open socket it cannot.
            if not _merge_lock.acquire(timeout=MERGE_LOCK_WAIT_S):
                raise MergeRequestError(
                    409, "another merge is in progress; retry shortly"
                )
            try:
                counts = run_merge(audit_log, self.db_path, source)
            finally:
                _merge_lock.release()
        finally:
            if source is not None:
                try:
                    source.close()
                except sqlite3.Error:
                    pass
            # The spool file goes, on every path out of here: success, a
            # refusal, a crash inside the merge, a client that hung up.
            try:
                tmp_path.unlink()
            except OSError:
                pass

        self._send_json(
            200,
            merge_report(
                counts, received, time.monotonic() - started, self.db_path
            ),
        )


FAVICON_SVG = (
    b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
    b'<rect width="16" height="16" rx="3" fill="#3a5bd9"/>'
    b'<path d="M4 5h8M4 8h8M4 11h5" stroke="#fff" stroke-width="1.6"'
    b' stroke-linecap="round"/></svg>'
)


class AuditServer(ThreadingHTTPServer):
    daemon_threads = True      # an SSE reader must never hold up shutdown
    allow_reuse_address = True


def build_server(db_path: Path, port: int) -> AuditServer:
    handler = type(
        "BoundAuditHandler",
        (AuditHandler,),
        {
            "db_path": Path(db_path),
            "port": port,
            "roots": WorkspaceRoots(db_path),
        },
    )
    return AuditServer((BIND_HOST, port), handler)


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def health_check(port, timeout=HEALTH_TIMEOUT_S):
    """The running service's /api/health payload, or None."""
    conn = None
    try:
        conn = http.client.HTTPConnection(BIND_HOST, port, timeout=timeout)
        conn.request("GET", "/api/health")
        response = conn.getresponse()
        body = response.read()
        if response.status != 200:
            return None
        payload = json.loads(body.decode("utf-8", "replace"))
        return payload if isinstance(payload, dict) and payload.get("ok") else None
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def read_pidfile(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text.split()[0])
    except (ValueError, IndexError):
        return None


def process_alive(pid) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


def write_pidfile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{0}\n".format(os.getpid()), encoding="utf-8")


def clear_pidfile(path: Path) -> None:
    """Remove the pidfile, but only while it is still ours."""
    try:
        if read_pidfile(path) == os.getpid():
            path.unlink()
    except OSError:
        pass


def cmd_serve(args) -> int:
    db_path = default_db_path(args.db)
    port = service_port(args.port)
    pidfile = pidfile_path(args.db)

    existing = health_check(port)
    if existing is not None:
        sys.stderr.write(
            "claude-audit: already serving on http://{0}:{1}\n".format(BIND_HOST, port)
        )
        return 0

    try:
        server = build_server(db_path, port)
    except OSError as exc:
        sys.stderr.write(
            "claude-audit: cannot bind {0}:{1}: {2}\n".format(BIND_HOST, port, exc)
        )
        return 1

    write_pidfile(pidfile)

    def shutdown(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, shutdown)
        except (ValueError, OSError):
            pass  # not the main thread, or the platform disagrees

    sys.stderr.write(
        "{0} claude-audit {1} serving http://{2}:{3} db={4}\n".format(
            utcnow(), VERSION, BIND_HOST, port, db_path
        )
    )
    sys.stderr.flush()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except Exception:
            pass
        clear_pidfile(pidfile)
    return 0


def spawn_detached(args) -> int:
    """Start `serve` in its own session, logging beside the database."""
    directory = audit_dir(args.db)
    directory.mkdir(parents=True, exist_ok=True)
    out_path, _ = log_paths(args.db)
    try:
        log = out_path.open("a", encoding="utf-8")
    except OSError:
        log = subprocess.DEVNULL

    argv = [sys.executable, str(Path(__file__).resolve()), "serve"]
    if args.db:
        argv += ["--db", str(args.db)]
    if args.port:
        argv += ["--port", str(args.port)]
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(directory),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    finally:
        if log is not subprocess.DEVNULL:
            log.close()
    return proc.pid


def cmd_start(args) -> int:
    port = service_port(args.port)
    existing = health_check(port)
    if existing is not None:
        print(
            json.dumps(
                {
                    "started": False,
                    "already_running": True,
                    "url": "http://{0}:{1}/".format(BIND_HOST, port),
                    "port": port,
                    "db": existing.get("db"),
                    "version": existing.get("version"),
                }
            )
        )
        return 0

    pid = spawn_detached(args)
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        health = health_check(port, timeout=0.5)
        if health is not None:
            print(
                json.dumps(
                    {
                        "started": True,
                        "already_running": False,
                        "pid": read_pidfile(pidfile_path(args.db)) or pid,
                        "url": "http://{0}:{1}/".format(BIND_HOST, port),
                        "port": port,
                        "db": health.get("db"),
                        "version": health.get("version"),
                    }
                )
            )
            return 0
        time.sleep(0.25)

    out_path, _ = log_paths(args.db)
    print(
        json.dumps(
            {
                "started": False,
                "already_running": False,
                "error": "service did not become healthy within {0}s".format(
                    int(START_TIMEOUT_S)
                ),
                "port": port,
                "log": str(out_path),
            }
        )
    )
    return 1


def cmd_status(args) -> int:
    port = service_port(args.port)
    health = health_check(port)
    pid = read_pidfile(pidfile_path(args.db))
    payload = {
        "running": health is not None,
        "port": port,
        "url": "http://{0}:{1}/".format(BIND_HOST, port),
        "pid": pid if process_alive(pid) else None,
        "db": str(default_db_path(args.db)),
        "db_exists": default_db_path(args.db).is_file(),
        "version": (health or {}).get("version", VERSION),
        "pidfile": str(pidfile_path(args.db)),
        "log": str(log_paths(args.db)[0]),
    }
    print(json.dumps(payload))
    return 0 if health is not None else 1


def cmd_stop(args) -> int:
    port = service_port(args.port)
    pidfile = pidfile_path(args.db)
    pid = read_pidfile(pidfile)
    running = health_check(port) is not None

    if not running and not process_alive(pid):
        try:
            pidfile.unlink()
        except OSError:
            pass
        print(json.dumps({"stopped": False, "running": False, "port": port}))
        return 0

    if not pid or not process_alive(pid):
        print(
            json.dumps(
                {
                    "stopped": False,
                    "running": True,
                    "port": port,
                    "error": "service is answering but no pidfile names it;"
                    " stop it where it was started",
                    "pidfile": str(pidfile),
                }
            )
        )
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(json.dumps({"stopped": False, "error": str(exc), "pid": pid}))
        return 1

    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not process_alive(pid) and health_check(port, timeout=0.3) is None:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    try:
        pidfile.unlink()
    except OSError:
        pass
    print(json.dumps({"stopped": True, "pid": pid, "port": port}))
    return 0


# --------------------------------------------------------------------------
# Startup installation (launchd / systemd --user)
# --------------------------------------------------------------------------


def startup_env(args):
    """The environment the supervisor must reproduce.

    A login agent inherits almost nothing, so anything that would change which
    database or port this service uses has to be written into the unit -- or
    the service started at login would quietly serve a different database from
    the one the shell serves.
    """
    env = {}
    for var, value in (
        (DB_ENV_VAR, str(Path(args.db).expanduser()) if args.db else os.environ.get(DB_ENV_VAR)),
        (PORT_ENV_VAR, str(args.port) if args.port else os.environ.get(PORT_ENV_VAR)),
    ):
        if value and str(value).strip():
            env[var] = str(value).strip()
    return env


def launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "{0}.plist".format(LAUNCHD_LABEL)


def systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def startup_spec(args):
    """(path, text, description) for this platform's startup definition."""
    python = sys.executable or "python3"
    script = str(Path(__file__).resolve())
    directory = audit_dir(args.db)
    out_log, err_log = log_paths(args.db)
    env = startup_env(args)

    if sys.platform == "darwin":
        payload = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": [python, script, "serve"],
            "RunAtLoad": True,
            # Restart a crash, but respect a clean `stop`: a service that
            # exited 0 was told to go away.
            "KeepAlive": {"SuccessfulExit": False},
            "StandardOutPath": str(out_log),
            "StandardErrorPath": str(err_log),
            "WorkingDirectory": str(directory),
            "ProcessType": "Background",
        }
        if env:
            payload["EnvironmentVariables"] = env
        text = plistlib.dumps(payload).decode("utf-8")
        return launchd_plist_path(), text, "launchd user agent"

    if sys.platform.startswith("linux"):
        lines = [
            "[Unit]",
            "Description=Claude Code audit log service (127.0.0.1 only)",
            "After=default.target",
            "",
            "[Service]",
            "Type=simple",
            "ExecStart={0} {1} serve".format(python, script),
            "WorkingDirectory={0}".format(directory),
            "Restart=on-failure",
            "RestartSec=5",
        ]
        for key in sorted(env):
            lines.append("Environment={0}={1}".format(key, env[key]))
        lines += [
            "StandardOutput=append:{0}".format(out_log),
            "StandardError=append:{0}".format(err_log),
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
        return systemd_unit_path(), "\n".join(lines), "systemd user unit"

    return None, None, "unsupported platform: {0}".format(sys.platform)


def run_quiet(argv):
    """Run a supervisor command; return (ok, combined output)."""
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return proc.returncode == 0, (proc.stdout or "").strip()


def cmd_print_startup(args) -> int:
    path, text, description = startup_spec(args)
    if path is None:
        sys.stderr.write("claude-audit: {0}\n".format(description))
        return 1
    if args.json:
        print(
            json.dumps(
                {
                    "platform": sys.platform,
                    "kind": description,
                    "path": str(path),
                    "label": LAUNCHD_LABEL
                    if sys.platform == "darwin"
                    else SYSTEMD_UNIT,
                    "text": text,
                    "log": str(log_paths(args.db)[0]),
                    "error_log": str(log_paths(args.db)[1]),
                    "python": sys.executable,
                    "script": str(Path(__file__).resolve()),
                }
            )
        )
        return 0
    # The unit text alone goes to stdout, so `print-startup > unit` is exactly
    # installable and a test can compare it byte for byte. The paths are
    # commentary, and commentary goes to stderr.
    sys.stderr.write("# {0}\n# path: {1}\n# log:  {2}\n".format(
        description, path, log_paths(args.db)[0]
    ))
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


def cmd_install_startup(args) -> int:
    path, text, description = startup_spec(args)
    if path is None:
        print(json.dumps({"installed": False, "error": description}))
        return 1

    path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir(args.db).mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    steps = []
    if sys.platform == "darwin":
        uid = os.getuid()
        # Unload any previous copy first, or bootstrap answers "already
        # bootstrapped" and the edited plist is never read.
        run_quiet(["launchctl", "bootout", "gui/{0}/{1}".format(uid, LAUNCHD_LABEL)])
        ok, output = run_quiet(
            ["launchctl", "bootstrap", "gui/{0}".format(uid), str(path)]
        )
        steps.append({"cmd": "launchctl bootstrap", "ok": ok, "output": output})
        if not ok:
            ok, output = run_quiet(["launchctl", "load", "-w", str(path)])
            steps.append({"cmd": "launchctl load -w", "ok": ok, "output": output})
    else:
        ok, output = run_quiet(["systemctl", "--user", "daemon-reload"])
        steps.append({"cmd": "systemctl --user daemon-reload", "ok": ok, "output": output})
        ok, output = run_quiet(
            ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT]
        )
        steps.append(
            {"cmd": "systemctl --user enable --now", "ok": ok, "output": output}
        )

    port = service_port(args.port)
    health = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and health is None:
        health = health_check(port, timeout=0.5)
        if health is None:
            time.sleep(0.3)

    print(
        json.dumps(
            {
                "installed": True,
                "kind": description,
                "path": str(path),
                "loaded": bool(steps and steps[-1]["ok"]),
                "healthy": health is not None,
                "url": "http://{0}:{1}/".format(BIND_HOST, port),
                "steps": steps,
            }
        )
    )
    return 0


def cmd_uninstall_startup(args) -> int:
    path, _, description = startup_spec(args)
    if path is None:
        print(json.dumps({"uninstalled": False, "error": description}))
        return 1

    steps = []
    if sys.platform == "darwin":
        uid = os.getuid()
        ok, output = run_quiet(
            ["launchctl", "bootout", "gui/{0}/{1}".format(uid, LAUNCHD_LABEL)]
        )
        steps.append({"cmd": "launchctl bootout", "ok": ok, "output": output})
        if not ok and path.is_file():
            ok, output = run_quiet(["launchctl", "unload", "-w", str(path)])
            steps.append({"cmd": "launchctl unload -w", "ok": ok, "output": output})
    else:
        ok, output = run_quiet(
            ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT]
        )
        steps.append({"cmd": "systemctl --user disable --now", "ok": ok, "output": output})

    removed = False
    try:
        path.unlink()
        removed = True
    except OSError:
        pass

    if sys.platform.startswith("linux"):
        ok, output = run_quiet(["systemctl", "--user", "daemon-reload"])
        steps.append({"cmd": "systemctl --user daemon-reload", "ok": ok, "output": output})

    print(
        json.dumps(
            {
                "uninstalled": True,
                "kind": description,
                "path": str(path),
                "removed": removed,
                "steps": steps,
            }
        )
    )
    return 0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


DB_HELP = (
    "audit database (default: $CLAUDE_AUDIT_DB or ~/.claude/audit/audit.db);"
    " the pidfile and logs live beside it"
)
PORT_HELP = "loopback port (default: $CLAUDE_AUDIT_PORT or {0})".format(DEFAULT_PORT)

SUBCOMMANDS = {
    "serve": cmd_serve,
    "start": cmd_start,
    "status": cmd_status,
    "stop": cmd_stop,
    "install-startup": cmd_install_startup,
    "uninstall-startup": cmd_uninstall_startup,
    "print-startup": cmd_print_startup,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="service.py", description="Local audit viewer service (127.0.0.1 only)"
    )
    sub = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("serve", "run the server in the foreground"),
        ("start", "spawn a detached server (no-op if one is already healthy)"),
        ("status", "report whether the service is answering"),
        ("stop", "terminate the detached server"),
        ("install-startup", "run the service at login"),
        ("uninstall-startup", "remove the login item"),
        ("print-startup", "print the plist/unit text without installing it"),
    ):
        child = sub.add_parser(name, help=help_text)
        child.add_argument("--db", default=None, help=DB_HELP)
        child.add_argument("--port", default=None, type=int, help=PORT_HELP)
        if name == "print-startup":
            child.add_argument(
                "--json",
                action="store_true",
                help="emit {path, text, label, ...} as JSON instead of raw text",
            )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] or ["serve"])
    handler = SUBCOMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
