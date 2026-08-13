"""Tests for the SessionDB read-path split (per-thread read-only connections).

The gateway shares ONE SessionDB across every agent, so recall/browse reads
used to queue behind writer flushes on self._lock — a measured production
convoy (a 0.2s FTS query stretched to 112s while 6-8 concurrent turns
flushed tool results). These tests pin the new contract: reads run on a
per-thread read-only connection under WAL, never touch self._lock, and fall
back to the legacy locked path when WAL or the read connection is missing.
"""

import sqlite3
import threading

import pytest

from hermes_state import SessionDB


def _run_checked_thread(target):
    failures = []

    def checked_target():
        try:
            target()
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=checked_target)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "reader thread did not finish"
    if failures:
        raise failures[0]


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="hello graphiti world")
    d.append_message("s1", role="assistant", content="the neo4j daemon is healthy")
    yield d
    d.close()


@pytest.mark.requires_wal
def test_read_conn_is_per_thread(db):
    conns = {}

    def grab(key):
        conns[key] = db._get_read_conn()

    t1 = threading.Thread(target=grab, args=(1,))
    t2 = threading.Thread(target=grab, args=(2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert conns[1] is not None and conns[2] is not None
    assert conns[1] is not conns[2]


def test_read_conn_reused_within_thread(db):
    assert db._get_read_conn() is db._get_read_conn()


@pytest.mark.requires_wal
def test_reads_do_not_take_writer_lock(db):
    """Reads must complete while another thread holds self._lock."""
    acquired = db._lock.acquire()
    assert acquired
    try:
        done = {}

        def reader():
            done["session"] = db.get_session("s1")
            done["search"] = db.search_messages("graphiti", limit=10)
            done["messages"] = db.get_messages("s1")

        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=5.0)
        assert not t.is_alive(), "read path blocked on writer lock"
        assert done["session"]["id"] == "s1"
        assert any("graphiti" in (m.get("snippet") or "") for m in done["search"])
        assert len(done["messages"]) == 2
    finally:
        db._lock.release()




def test_read_your_writes(db):
    """A fresh committed write must be visible to the read connection."""
    db.append_message("s1", role="user", content="zanzibar checkpoint")
    rows = db.search_messages("zanzibar", limit=5)
    assert rows, "committed write invisible to read connection"




def test_non_wal_uses_locked_path(db):
    db._wal_active = False
    assert db._get_read_conn() is None
    # And queries still work via the legacy path.
    assert db.get_session("s1")["id"] == "s1"


@pytest.mark.requires_wal
def test_read_conn_open_failure_marks_thread(db, monkeypatch, tmp_path):
    """A failed read-conn open must not retry per query; fallback still works."""
    import sqlite3 as _sqlite3

    calls = {"n": 0}
    real_connect = _sqlite3.connect

    def failing_connect(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("file:") and "mode=ro" in a[0]:
            calls["n"] += 1
            raise _sqlite3.OperationalError("simulated open failure")
        return real_connect(*a, **k)

    fresh = SessionDB(db_path=tmp_path / "state2.db")
    try:
        fresh.create_session(session_id="x", source="cli", model="m")
        monkeypatch.setattr("hermes_state.sqlite3.connect", failing_connect)
        assert fresh.get_session("x")["id"] == "x"
        assert fresh.get_session("x")["id"] == "x"
        assert calls["n"] == 1, "open failure should be remembered per thread"
    finally:
        fresh.close()


@pytest.mark.requires_wal
def test_anchored_view_and_around_use_read_path(db):
    msgs = db.get_messages("s1")
    anchor = msgs[0]["id"]
    acquired = db._lock.acquire()
    try:
        done = {}

        def reader():
            done["around"] = db.get_messages_around("s1", anchor, window=2)
            done["view"] = db.get_anchored_view("s1", anchor, window=2, bookend=1)

        t = threading.Thread(target=reader)
        t.start(); t.join(timeout=5.0)
        assert not t.is_alive(), "anchored reads blocked on writer lock"
        assert done["around"]["window"]
        assert done["view"]["window"]
    finally:
        db._lock.release()


@pytest.mark.requires_wal
def test_session_resume_reads_do_not_take_writer_lock(db):
    """session.resume's three read paths must not convoy behind writer flushes.

    get_messages_as_conversation / get_resume_conversations /
    get_ancestor_display_prefix are the hottest reads in the file — every
    resume across the gateway, CLI, and ACP adapter goes through one of
    them — so they must use the same per-thread read-only connection as
    get_messages, not the legacy self._lock path.
    """
    db.create_session(session_id="parent1", source="cli", model="m")
    db.append_message("parent1", role="user", content="parent turn")
    db.append_message("parent1", role="assistant", content="parent reply")
    db.create_session(session_id="child1", source="cli", model="m", parent_session_id="parent1")
    db.append_message("child1", role="user", content="child turn")
    db.append_message("child1", role="assistant", content="child reply")

    acquired = db._lock.acquire()
    try:
        done = {}

        def reader():
            done["conversation"] = db.get_messages_as_conversation("s1")
            done["resume"] = db.get_resume_conversations("child1")
            done["ancestor_prefix"] = db.get_ancestor_display_prefix("child1")

        t = threading.Thread(target=reader)
        t.start(); t.join(timeout=5.0)
        assert not t.is_alive(), "session resume reads blocked on writer lock"
        assert len(done["conversation"]) == 2
        model_history, display_history = done["resume"]
        assert len(model_history) == 2
        assert len(display_history) == 4
        assert len(done["ancestor_prefix"]) == 2
    finally:
        db._lock.release()


def test_completed_reader_threads_do_not_accumulate_connections(db):
    """A traffic burst must not pin one SQLite connection per retired worker."""
    db._wal_active = True
    stable_conn = db._get_read_conn()
    assert stable_conn is not None

    for _ in range(30):
        _run_checked_thread(lambda: db.get_session("s1"))

    assert len(db._read_conns) <= 2
    assert db._get_read_conn() is stable_conn


def test_live_reader_thread_cache_is_bounded(db):
    """Executor threads may stay alive after a burst; their cache is capped."""
    db._wal_active = True
    release = threading.Event()
    ready = [threading.Event() for _ in range(30)]
    failures = []

    def reader(index):
        try:
            db.get_session("s1")
            ready[index].set()
            release.wait(timeout=5.0)
        except BaseException as exc:
            failures.append(exc)
            ready[index].set()

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(30)]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(timeout=5.0) for event in ready)
        if failures:
            raise failures[0]
        assert len(db._read_conns) <= db._MAX_READ_CONNECTIONS
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5.0)
            assert not thread.is_alive(), "reader thread did not finish"


def test_close_waits_for_active_read_context(db):
    """Final teardown must not close a connection during its SELECT scope."""
    db._wal_active = True
    reading = threading.Event()
    finish_read = threading.Event()
    close_returned = threading.Event()
    failures = []
    read_conn = []

    def reader():
        try:
            with db._read_ctx() as conn:
                read_conn.append(conn)
                reading.set()
                assert finish_read.wait(timeout=5.0)
                assert conn.execute("SELECT 1").fetchone()[0] == 1
        except BaseException as exc:
            failures.append(exc)

    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    assert reading.wait(timeout=5.0)

    def closer():
        try:
            db.close()
            close_returned.set()
        except BaseException as exc:
            failures.append(exc)

    close_thread = threading.Thread(target=closer)
    close_thread.start()
    try:
        with db._read_conns_lock:
            assert db._read_conns_closed
            assert not close_returned.is_set()
    finally:
        finish_read.set()

    reader_thread.join(timeout=5.0)
    close_thread.join(timeout=5.0)
    assert not reader_thread.is_alive()
    assert not close_thread.is_alive()
    if failures:
        raise failures[0]
    assert close_returned.is_set()
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        read_conn[0].execute("SELECT 1")
