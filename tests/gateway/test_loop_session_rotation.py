"""Persistent /loop state follows gateway-created session successors."""

from datetime import datetime, timedelta
from pathlib import Path
import threading

import pytest

from gateway.config import GatewayConfig, SessionResetPolicy
from gateway.platforms.base import Platform, SessionSource
from gateway.session import SessionStore
from hermes_cli import loops
from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _source(*, profile=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="loop-rotation-chat",
        user_id="loop-rotation-user",
        profile=profile,
    )


def _store(*, policy=None, multiplex=False):
    config = GatewayConfig(
        default_reset_policy=policy or SessionResetPolicy(),
        multiplex_profiles=multiplex,
    )
    return SessionStore(Path(get_hermes_home()) / "sessions", config)


def _set_loop(session_id, *, prompt="watch the deploy", route=None):
    state = loops.LoopManager(session_id=session_id).set(
        prompt,
        interval_seconds=300,
        times=12,
        until="deploy is healthy",
        route=route,
    )
    state.ticks_fired = 4
    state.last_fired_at = 123.5
    state.last_response_digest = "digest-4"
    loops.save_loop(session_id, state)
    return state


def _expire_daily(store, entry):
    with store._lock:
        entry.updated_at = datetime.now() - timedelta(days=2)
        store._save()


def test_daily_auto_reset_migrates_active_loop_with_all_state():
    store = _store(policy=SessionResetPolicy(mode="daily", at_hour=0))
    source = _source()
    old_entry = store.get_or_create_session(source)
    route = {
        "platform": "telegram",
        "chat_id": source.chat_id,
        "chat_type": source.chat_type,
        "user_id": source.user_id,
    }
    expected = _set_loop(old_entry.session_id, route=route)
    _expire_daily(store, old_entry)

    new_entry = store.get_or_create_session(source)

    assert new_entry.session_id != old_entry.session_id
    assert new_entry.auto_reset_reason == "daily"
    migrated = loops.load_loop(new_entry.session_id)
    assert migrated is not None
    assert migrated.status == "active"
    assert migrated.prompt == expected.prompt
    assert migrated.route == route
    assert migrated.times == 12
    assert migrated.ticks_fired == 4
    assert migrated.last_fired_at == 123.5
    assert migrated.last_response_digest == "digest-4"
    assert migrated.until == "deploy is healthy"
    assert loops.load_loop(old_entry.session_id).status == "cleared"


@pytest.mark.parametrize(
    ("reason", "policy"),
    [
        ("idle", SessionResetPolicy(mode="idle", idle_minutes=1)),
        ("suspended", SessionResetPolicy(mode="none")),
        (
            "resume_pending_expired",
            SessionResetPolicy(mode="idle", idle_minutes=999999),
        ),
    ],
)
def test_other_auto_reset_paths_migrate_active_loop(reason, policy, monkeypatch):
    store = _store(policy=policy)
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id)

    with store._lock:
        if reason == "idle":
            old_entry.updated_at = datetime.now() - timedelta(minutes=5)
        elif reason == "suspended":
            old_entry.suspended = True
        else:
            monkeypatch.setattr(
                "gateway.session.auto_continue_freshness_window", lambda: 3600
            )
            old_entry.resume_pending = True
            old_entry.last_resume_marked_at = datetime.now() - timedelta(hours=2)
        store._save()

    new_entry = store.get_or_create_session(source)

    assert new_entry.auto_reset_reason == reason
    assert loops.load_loop(new_entry.session_id).prompt == "watch the deploy"
    assert loops.load_loop(old_entry.session_id).status == "cleared"


def test_auto_reset_without_loop_leaves_both_sessions_without_loop_state():
    store = _store(policy=SessionResetPolicy(mode="daily", at_hour=0))
    source = _source()
    old_entry = store.get_or_create_session(source)
    _expire_daily(store, old_entry)

    new_entry = store.get_or_create_session(source)

    assert new_entry.session_id != old_entry.session_id
    assert loops.load_loop(old_entry.session_id) is None
    assert loops.load_loop(new_entry.session_id) is None


def test_explicit_reset_session_migrates_active_loop():
    store = _store()
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id, prompt="poll the queue")

    new_entry = store.reset_session(old_entry.session_key)

    assert new_entry is not None
    assert loops.load_loop(new_entry.session_id).prompt == "poll the queue"
    assert loops.load_loop(old_entry.session_id).status == "cleared"


@pytest.mark.parametrize("second_rotation", ["reset", "force_new", "daily"])
def test_overlapping_rotations_serialize_loop_migration_through_route_publication(
    monkeypatch, second_rotation
):
    policy = (
        SessionResetPolicy(mode="daily", at_hour=0)
        if second_rotation == "daily"
        else None
    )
    store = _store(policy=policy)
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id, prompt="follow every rotation")

    real_migrate = loops.migrate_loop_to_session
    first_migration_started = threading.Event()
    release_first_migration = threading.Event()
    second_reset_finished = threading.Event()
    rotation_results = {}
    reset_errors = []

    def pause_first_migration(old_session_id, new_session_id, **kwargs):
        if old_session_id == old_entry.session_id:
            first_migration_started.set()
            assert release_first_migration.wait(timeout=5)
        return real_migrate(old_session_id, new_session_id, **kwargs)

    def first_reset():
        try:
            rotation_results["first"] = store.reset_session(old_entry.session_key)
        except BaseException as exc:  # pragma: no cover - surfaced below
            reset_errors.append(exc)

    def second_reset():
        try:
            if second_rotation == "reset":
                result = store.reset_session(old_entry.session_key)
            else:
                result = store.get_or_create_session(
                    source,
                    force_new=second_rotation == "force_new",
                )
            rotation_results["second"] = result
        except BaseException as exc:  # pragma: no cover - surfaced below
            reset_errors.append(exc)
        finally:
            second_reset_finished.set()

    monkeypatch.setattr(loops, "migrate_loop_to_session", pause_first_migration)
    first = threading.Thread(target=first_reset)
    second = threading.Thread(target=second_reset)

    first.start()
    assert first_migration_started.wait(timeout=5)
    if second_rotation == "daily":
        with store._lock:
            store._entries[old_entry.session_key].updated_at = (
                datetime.now() - timedelta(days=2)
            )
            store._save()
    second.start()
    second_finished_while_first_was_paused = second_reset_finished.wait(timeout=0.5)
    release_first_migration.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert reset_errors == []
    assert second_finished_while_first_was_paused is False
    assert set(rotation_results) == {"first", "second"}
    current = store.lookup_by_session_key(old_entry.session_key)
    assert current is not None
    assert (
        rotation_results["second"].session_id
        != rotation_results["first"].session_id
    )
    assert current.session_id == rotation_results["second"].session_id
    assert (
        loops.load_loop(old_entry.session_id).migrated_to
        == rotation_results["first"].session_id
    )
    assert (
        loops.load_loop(rotation_results["first"].session_id).migrated_to
        == current.session_id
    )
    assert [
        (session_id, state.prompt)
        for session_id, state in loops.list_active_loops()
    ] == [(current.session_id, "follow every rotation")]


@pytest.mark.parametrize("first_migration", ["compression", "reset"])
def test_compression_reset_overlap_keeps_loop_on_current_route(first_migration):
    store = _store()
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id, prompt="follow compression and reset")
    compression_child_id = f"{old_entry.session_id}_compression"

    compression_started = threading.Event()
    allow_compression_migration = threading.Event()
    compression_migrated = threading.Event()
    allow_compression_route_advance = threading.Event()
    compression_result = {}
    compression_errors = []

    def compress_then_advance_route():
        try:
            compression_started.set()
            assert allow_compression_migration.wait(timeout=5)
            compression_result["migrated"] = loops.migrate_loop_to_session(
                old_entry.session_id,
                compression_child_id,
                reason="compression",
            )
            compression_migrated.set()
            assert allow_compression_route_advance.wait(timeout=5)
            compression_result["advanced"] = store.advance_compression_session(
                old_entry.session_key,
                old_entry.session_id,
                compression_child_id,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            compression_errors.append(exc)

    compression = threading.Thread(target=compress_then_advance_route)
    compression.start()
    assert compression_started.wait(timeout=5)
    try:
        if first_migration == "compression":
            allow_compression_migration.set()
            assert compression_migrated.wait(timeout=5)
            reset_entry = store.reset_session(old_entry.session_key)
        else:
            reset_entry = store.reset_session(old_entry.session_key)
            allow_compression_migration.set()
            assert compression_migrated.wait(timeout=5)
    finally:
        allow_compression_migration.set()
        allow_compression_route_advance.set()
        compression.join(timeout=5)

    assert not compression.is_alive()
    assert compression_errors == []
    assert reset_entry is not None
    assert compression_result["migrated"] is (first_migration == "compression")
    assert compression_result["advanced"] is None
    current = store.lookup_by_session_key(old_entry.session_key)
    assert current is not None
    assert current.session_id == reset_entry.session_id
    assert [
        (session_id, state.prompt)
        for session_id, state in loops.list_active_loops()
    ] == [(current.session_id, "follow compression and reset")]


def test_force_new_rotation_migrates_active_loop():
    store = _store()
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id, prompt="follow forced rotation")

    new_entry = store.get_or_create_session(source, force_new=True)

    assert new_entry.session_id != old_entry.session_id
    assert loops.load_loop(new_entry.session_id).prompt == "follow forced rotation"
    assert loops.load_loop(old_entry.session_id).status == "cleared"


def test_loop_migration_failure_never_breaks_successor_creation(monkeypatch):
    store = _store()
    source = _source()
    old_entry = store.get_or_create_session(source)
    _set_loop(old_entry.session_id)

    def fail_migration(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loops, "migrate_loop_to_session", fail_migration)

    new_entry = store.reset_session(old_entry.session_key)

    assert new_entry is not None
    assert store._db_for_key(old_entry.session_key).get_session(new_entry.session_id)
    assert loops.load_loop(old_entry.session_id).status == "active"


def test_profile_scoped_auto_reset_migrates_only_in_owning_session_db(
    tmp_path, monkeypatch
):
    import hermes_state

    root = tmp_path / "hermes"
    profile_home = root / "profiles" / "jaros-app"
    (root / "sessions").mkdir(parents=True)
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )

    store = _store(
        policy=SessionResetPolicy(mode="daily", at_hour=0), multiplex=True
    )
    source = _source(profile="jaros-app")
    old_entry = store.get_or_create_session(source)
    token = set_hermes_home_override(profile_home)
    try:
        _set_loop(old_entry.session_id, prompt="profile-only loop")
    finally:
        reset_hermes_home_override(token)
    _expire_daily(store, old_entry)
    profile_db = store._db_for_key(old_entry.session_key)
    real_migrate = loops.migrate_loop_to_session
    migration_dbs = []

    def capture_migration_db(*args, **kwargs):
        migration_dbs.append(kwargs.get("db"))
        return real_migrate(*args, **kwargs)

    monkeypatch.setattr(loops, "migrate_loop_to_session", capture_migration_db)

    # Exercise the unscoped background-watcher shape: the routing key, not
    # ambient HERMES_HOME, must select the profile that owns the session.
    new_entry = store.get_or_create_session(source)

    assert migration_dbs == [profile_db]

    token = set_hermes_home_override(profile_home)
    try:
        assert loops.load_loop(new_entry.session_id).prompt == "profile-only loop"
        assert loops.load_loop(old_entry.session_id).status == "cleared"
    finally:
        reset_hermes_home_override(token)
    assert loops.load_loop(old_entry.session_id) is None
    assert loops.load_loop(new_entry.session_id) is None
