"""Tests for hermes_cli/loops.py — /loop recurring in-session wakeups."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import asdict
from unittest.mock import patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# Interval / argument parsing
# ──────────────────────────────────────────────────────────────────────


class TestParseIntervalToken:
    def test_minutes(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("5m") == 300

    def test_seconds(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("30s") == 30

    def test_hours(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("2h") == 7200

    def test_compound(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("1h30m") == 5400

    def test_case_insensitive(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("5M") == 300

    def test_bare_number_is_not_interval(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("3") is None

    def test_prose_is_not_interval(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("check") is None
        assert parse_interval_token("") is None
        assert parse_interval_token("5x") is None

    def test_zero_rejected(self):
        from hermes_cli.loops import parse_interval_token

        assert parse_interval_token("0m") is None
        assert parse_interval_token("0s") is None


class TestParseLoopArgs:
    def test_fixed_interval(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m check the deploy status")
        assert p["interval_seconds"] == 300
        assert p["prompt"] == "check the deploy status"
        assert p["error"] is None

    def test_every_sugar(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("every 10m /recap")
        assert p["interval_seconds"] == 600
        assert p["prompt"] == "/recap"

    def test_self_paced(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("keep refining the failing test until the suite passes")
        assert p["interval_seconds"] is None
        assert p["prompt"].startswith("keep refining")

    def test_times_flag(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll CI --times 30")
        assert p["interval_seconds"] == 120
        assert p["prompt"] == "poll CI"
        assert p["times"] == 30

    def test_until_flag(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m watch the queue --until queue depth reaches zero")
        assert p["interval_seconds"] == 300
        assert p["prompt"] == "watch the queue"
        assert p["until"] == "queue depth reaches zero"

    def test_until_and_times_together(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll --times 5 --until it is green")
        assert p["times"] == 5
        assert p["until"] == "it is green"
        assert p["prompt"] == "poll"

    def test_bad_times(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("2m poll --times zero")
        assert p["error"] is not None

    def test_no_start_now_flag_anymore(self):
        from hermes_cli.loops import parse_loop_args

        # Immediate first wakeup is the default now; --start-now was never
        # released, so the token is NOT parsed as a flag.
        p = parse_loop_args("1h check the deploy status")
        assert "start_now" not in p
        assert p["interval_seconds"] == 3600
        assert p["prompt"] == "check the deploy status"
        assert p["error"] is None

    def test_start_now_word_in_prompt_kept_verbatim(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("1h read the file called start-now.md")
        assert p["prompt"] == "read the file called start-now.md"

    def test_interval_only_is_error(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("5m")
        assert p["error"] is not None

    def test_empty(self):
        from hermes_cli.loops import parse_loop_args

        assert parse_loop_args("")["error"] == "empty"

    def test_prompt_with_leading_number_not_eaten(self):
        from hermes_cli.loops import parse_loop_args

        p = parse_loop_args("3 things to verify in the repo")
        assert p["interval_seconds"] is None
        assert p["prompt"].startswith("3 things")


class TestFormatInterval:
    def test_render(self):
        from hermes_cli.loops import format_interval

        assert format_interval(30) == "30s"
        assert format_interval(300) == "5m"
        assert format_interval(5400) == "1h30m"
        assert format_interval(90) == "1m30s"
        assert format_interval(0) == "0s"


# ──────────────────────────────────────────────────────────────────────
# LOOP_COMPLETE marker
# ──────────────────────────────────────────────────────────────────────


class TestResponseSignalsComplete:
    def test_marker_on_own_line(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("Deploy is live.\nLOOP_COMPLETE") is True

    def test_marker_with_trailing_period(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("done\nLOOP_COMPLETE.") is True

    def test_marker_mid_sentence_does_not_count(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("I will emit LOOP_COMPLETE when finished") is False

    def test_no_marker(self):
        from hermes_cli.loops import response_signals_complete

        assert response_signals_complete("still building") is False
        assert response_signals_complete("") is False


# ──────────────────────────────────────────────────────────────────────
# LoopState round-trip
# ──────────────────────────────────────────────────────────────────────


class TestLoopStateSerde:
    def test_round_trip(self):
        from hermes_cli.loops import LoopState

        s = LoopState(
            prompt="check CI",
            mode="interval",
            interval_seconds=300.0,
            current_delay=300.0,
            times=5,
            until="ci is green",
            ticks_fired=2,
            route={"platform": "telegram", "chat_id": "123"},
        )
        s2 = LoopState.from_json(s.to_json())
        assert s2.prompt == "check CI"
        assert s2.interval_seconds == 300.0
        assert s2.times == 5
        assert s2.until == "ci is green"
        assert s2.ticks_fired == 2
        assert s2.route == {"platform": "telegram", "chat_id": "123"}

    def test_old_row_missing_fields(self):
        from hermes_cli.loops import LoopState

        s = LoopState.from_json('{"prompt": "p"}')
        assert s.prompt == "p"
        assert s.status == "active"
        assert s.route == {}


# ──────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_load_clear(self, hermes_home):
        from hermes_cli.loops import LoopManager, load_loop

        mgr = LoopManager(session_id="sess-1")
        mgr.set("check the deploy", interval_seconds=300)
        loaded = load_loop("sess-1")
        assert loaded is not None
        assert loaded.prompt == "check the deploy"
        assert loaded.status == "active"

        assert mgr.clear() is True
        cleared = load_loop("sess-1")
        assert cleared is not None and cleared.status == "cleared"

        restarted = mgr.set("check it again", interval_seconds=300)
        assert restarted.status == "active"
        assert load_loop("sess-1").prompt == "check it again"

    def test_list_active_loops(self, hermes_home):
        from hermes_cli.loops import LoopManager, list_active_loops

        LoopManager(session_id="a").set("task a", interval_seconds=60)
        mgr_b = LoopManager(session_id="b")
        mgr_b.set("task b", interval_seconds=60)
        mgr_b.pause()

        active = dict(list_active_loops())
        assert "a" in active
        assert "b" not in active

    def test_migrate_to_session(self, hermes_home):
        from hermes_cli.loops import (
            LoopManager,
            list_active_loops,
            load_loop,
            migrate_loop_to_session,
            save_loop,
        )

        expected = LoopManager(session_id="parent").set(
            "watch it",
            interval_seconds=60,
            times=7,
            until="the deploy is healthy",
            route={"platform": "telegram", "chat_id": "42"},
        )
        expected.current_delay = 91.0
        expected.ticks_fired = 3
        expected.last_fired_at = 123.5
        expected.next_due_at = 456.75
        expected.awaiting_response = True
        expected.last_response_digest = "digest-3"
        expected.paused_reason = "preserved metadata"
        expected.last_stop_reason = "also preserved"
        save_loop("parent", expected)
        expected_fields = asdict(expected)
        expected_generation = expected_fields.pop("generation")

        assert migrate_loop_to_session("parent", "child", reason="compression") is True
        child = load_loop("child")
        assert child is not None
        child_fields = asdict(child)
        child_generation = child_fields.pop("generation")
        assert child_fields == expected_fields
        assert child_generation == expected_generation + 1
        parent = load_loop("parent")
        assert parent is not None and parent.status == "cleared"
        assert parent.generation == child_generation
        assert [session_id for session_id, _state in list_active_loops()] == ["child"]

    def test_migrate_successor_write_failure_keeps_predecessor_active(
        self, hermes_home
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="parent-write-fail").set(
            "watch it", interval_seconds=60
        )
        db = loops._get_session_db()
        db._execute_write(
            lambda conn: conn.execute(
                """
                CREATE TRIGGER fail_loop_successor_write
                BEFORE INSERT ON state_meta
                WHEN NEW.key = 'loop:child-write-fail'
                BEGIN
                    SELECT RAISE(FAIL, 'injected successor write failure');
                END
                """
            )
        )

        assert (
            loops.migrate_loop_to_session(
                "parent-write-fail", "child-write-fail", reason="test"
            )
            is False
        )
        assert loops.load_loop("parent-write-fail").status == "active"
        assert loops.load_loop("child-write-fail") is None

    def test_migrate_predecessor_clear_failure_rolls_back_successor(
        self, hermes_home
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="parent-clear-fail").set(
            "watch it", interval_seconds=60
        )
        db = loops._get_session_db()
        db._execute_write(
            lambda conn: conn.execute(
                """
                CREATE TRIGGER fail_loop_predecessor_clear
                BEFORE UPDATE OF value ON state_meta
                WHEN OLD.key = 'loop:parent-clear-fail'
                BEGIN
                    SELECT RAISE(FAIL, 'injected predecessor clear failure');
                END
                """
            )
        )

        assert (
            loops.migrate_loop_to_session(
                "parent-clear-fail", "child-clear-fail", reason="test"
            )
            is False
        )
        assert loops.load_loop("parent-clear-fail").status == "active"
        assert loops.load_loop("child-clear-fail") is None

    def test_stale_manager_cannot_reactivate_migrated_predecessor(self, hermes_home):
        from hermes_cli import loops

        loops.LoopManager(session_id="parent-stale").set(
            "watch it", interval_seconds=60
        )
        stale = loops.LoopManager(session_id="parent-stale")

        assert loops.migrate_loop_to_session("parent-stale", "child-stale") is True
        stale.state.next_due_at = time.time() - 1

        assert stale.fire_tick() is None
        with pytest.raises(loops.LoopPersistenceError, match="migrated"):
            stale.set("reactivate predecessor", interval_seconds=60)

        assert loops.load_loop("parent-stale").status == "cleared"
        assert loops.load_loop("parent-stale").migrated_to == "child-stale"
        assert loops.load_loop("child-stale").status == "active"
        assert [session_id for session_id, _state in loops.list_active_loops()] == [
            "child-stale"
        ]

    def test_gateway_migration_follows_compression_descendant_chain(
        self, hermes_home
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="chain-root").set(
            "follow the route", interval_seconds=60
        )
        assert loops.migrate_loop_to_session("chain-root", "compression-one")
        assert loops.migrate_loop_to_session("compression-one", "compression-two")

        assert loops.migrate_loop_to_session(
            "chain-root",
            "gateway-target",
            follow_migrated=True,
        )

        assert loops.load_loop("chain-root").migrated_to == "compression-one"
        assert loops.load_loop("compression-one").migrated_to == "compression-two"
        assert loops.load_loop("compression-two").migrated_to == "gateway-target"
        assert [session_id for session_id, _state in loops.list_active_loops()] == [
            "gateway-target"
        ]

    def test_stale_compression_does_not_follow_gateway_migration(
        self, hermes_home
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="gateway-first-root").set(
            "stay current", interval_seconds=60
        )
        assert loops.migrate_loop_to_session(
            "gateway-first-root",
            "gateway-first-target",
            follow_migrated=True,
        )

        assert not loops.migrate_loop_to_session(
            "gateway-first-root",
            "stale-compression-target",
        )
        assert loops.load_loop("stale-compression-target") is None
        assert [session_id for session_id, _state in loops.list_active_loops()] == [
            "gateway-first-target"
        ]

    def test_gateway_migration_rejects_forwarding_cycle(self, hermes_home):
        from hermes_cli import loops

        db = loops._get_session_db()
        db.set_meta(
            loops._meta_key("cycle-a"),
            loops.LoopState(
                prompt="cycle",
                status="cleared",
                migrated_to="cycle-b",
            ).to_json(),
        )
        db.set_meta(
            loops._meta_key("cycle-b"),
            loops.LoopState(
                prompt="cycle",
                status="cleared",
                migrated_to="cycle-a",
            ).to_json(),
        )

        assert not loops.migrate_loop_to_session(
            "cycle-a",
            "cycle-target",
            follow_migrated=True,
            db=db,
        )
        assert loops.load_loop("cycle-target") is None

    def test_gateway_migration_bounds_forwarding_chain(self, hermes_home):
        from hermes_cli import loops

        db = loops._get_session_db()
        nodes = [
            f"bounded-{index}"
            for index in range(loops._MIGRATION_CHAIN_MAX_HOPS + 2)
        ]
        for current, successor in zip(nodes, nodes[1:]):
            db.set_meta(
                loops._meta_key(current),
                loops.LoopState(
                    prompt="bounded",
                    status="cleared",
                    migrated_to=successor,
                ).to_json(),
            )
        loops.LoopManager(session_id=nodes[-1]).set(
            "bounded active", interval_seconds=60
        )

        assert not loops.migrate_loop_to_session(
            nodes[0],
            "bounded-target",
            follow_migrated=True,
            db=db,
        )
        assert loops.load_loop("bounded-target") is None
        assert [session_id for session_id, _state in loops.list_active_loops()] == [
            nodes[-1]
        ]

    def test_migrate_no_source(self, hermes_home):
        from hermes_cli.loops import migrate_loop_to_session

        assert migrate_loop_to_session("nope", "child2") is False
        assert migrate_loop_to_session("same", "same") is False

    def test_migrate_target_wins_and_predecessor_is_cleared(self, hermes_home):
        from hermes_cli.loops import (
            LoopManager,
            list_active_loops,
            load_loop,
            migrate_loop_to_session,
            save_loop,
        )

        LoopManager(session_id="p2").set("parent loop", interval_seconds=60)
        target = LoopManager(session_id="c2").set(
            "child loop", interval_seconds=120, times=9
        )
        target.ticks_fired = 4
        target.route = {"platform": "telegram", "chat_id": "target"}
        save_loop("c2", target)
        target_raw = target.to_json()

        assert migrate_loop_to_session("p2", "c2") is True
        assert load_loop("c2").to_json() == target_raw
        assert load_loop("p2").status == "cleared"
        assert [session_id for session_id, _state in list_active_loops()] == ["c2"]

    def test_tick_committed_before_migration_is_carried_to_successor(
        self, hermes_home
    ):
        from hermes_cli import loops

        mgr = loops.LoopManager(session_id="tick-wins-parent")
        mgr.set("poll", interval_seconds=60)
        assert mgr.fire_tick() is not None
        claimed = loops.load_loop("tick-wins-parent")

        assert loops.migrate_loop_to_session(
            "tick-wins-parent", "tick-wins-child"
        )
        child = loops.load_loop("tick-wins-child")
        assert child.ticks_fired == claimed.ticks_fired == 1
        assert child.awaiting_response is claimed.awaiting_response is True
        assert child.last_fired_at == claimed.last_fired_at
        assert child.next_due_at == claimed.next_due_at
        assert loops.load_loop("tick-wins-parent").status == "cleared"

    def test_migration_retries_bounded_cas_conflict(
        self, hermes_home, monkeypatch
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="retry-parent").set(
            "poll", interval_seconds=60
        )
        db = loops._get_session_db()
        real_compare = db.compare_and_set_meta_batch
        calls = 0

        def conflict_once(expected, updates):
            nonlocal calls
            calls += 1
            if calls == 1:
                return False
            return real_compare(expected, updates)

        monkeypatch.setattr(db, "compare_and_set_meta_batch", conflict_once)

        assert loops.migrate_loop_to_session(
            "retry-parent", "retry-child", db=db
        )
        assert calls == 2
        assert loops.load_loop("retry-parent").status == "cleared"
        assert loops.load_loop("retry-child").status == "active"

    def test_migration_stops_after_bounded_cas_conflicts(
        self, hermes_home, monkeypatch
    ):
        from hermes_cli import loops

        loops.LoopManager(session_id="retry-bound-parent").set(
            "poll", interval_seconds=60
        )
        db = loops._get_session_db()
        calls = 0

        def always_conflict(_expected, _updates):
            nonlocal calls
            calls += 1
            return False

        monkeypatch.setattr(db, "compare_and_set_meta_batch", always_conflict)

        assert loops.migrate_loop_to_session(
            "retry-bound-parent", "retry-bound-child", db=db
        ) is False
        assert calls == 3
        assert loops.load_loop("retry-bound-parent").status == "active"
        assert loops.load_loop("retry-bound-child") is None

    def test_gateway_follows_migration_winning_final_cas_conflict(
        self, hermes_home, monkeypatch
    ):
        from hermes_cli import loops

        root = "final-conflict-root"
        compression_child = "final-conflict-compression"
        gateway_target = "final-conflict-gateway"
        loops.LoopManager(session_id=root).set("poll", interval_seconds=60)
        db = loops._get_session_db()
        real_compare = db.compare_and_set_meta_batch
        calls = 0

        def compression_wins_final_attempt(expected, updates):
            nonlocal calls
            calls += 1
            if calls < loops._MIGRATION_CAS_ATTEMPTS:
                return False
            if calls == loops._MIGRATION_CAS_ATTEMPTS:
                root_key = loops._meta_key(root)
                child_key = loops._meta_key(compression_child)
                root_raw = db.get_meta(root_key)
                source = loops.LoopState.from_json(root_raw)
                next_generation = source.generation + 1
                predecessor = loops.LoopState.from_json(root_raw)
                predecessor.status = "cleared"
                predecessor.generation = next_generation
                predecessor.migrated_to = compression_child
                successor = loops.LoopState.from_json(root_raw)
                successor.generation = next_generation
                assert real_compare(
                    {root_key: root_raw, child_key: None},
                    {
                        child_key: successor.to_json(),
                        root_key: predecessor.to_json(),
                    },
                )
                return False
            return real_compare(expected, updates)

        monkeypatch.setattr(
            db,
            "compare_and_set_meta_batch",
            compression_wins_final_attempt,
        )

        assert loops.migrate_loop_to_session(
            root,
            gateway_target,
            db=db,
            follow_migrated=True,
        )
        assert calls == loops._MIGRATION_CAS_ATTEMPTS + 1
        assert [session_id for session_id, _state in loops.list_active_loops()] == [
            gateway_target
        ]

    def test_atomic_meta_batch_conflict_applies_no_updates(self, hermes_home):
        from hermes_cli import loops

        db = loops._get_session_db()
        db.set_meta("batch:first", "old-first")

        assert db.compare_and_set_meta_batch(
            {"batch:first": "wrong", "batch:second": None},
            {"batch:first": "new-first", "batch:second": "new-second"},
        ) is False
        assert db.get_meta("batch:first") == "old-first"
        assert db.get_meta("batch:second") is None

    def test_atomic_meta_batch_write_failure_rolls_back_all_updates(
        self, hermes_home
    ):
        from hermes_cli import loops

        db = loops._get_session_db()
        db.set_meta("batch:rollback-first", "old-first")
        db.set_meta("batch:rollback-second", "old-second")
        db._execute_write(
            lambda conn: conn.execute(
                """
                CREATE TRIGGER fail_batch_second_update
                BEFORE UPDATE OF value ON state_meta
                WHEN OLD.key = 'batch:rollback-second'
                BEGIN
                    SELECT RAISE(FAIL, 'injected batch write failure');
                END
                """
            )
        )

        with pytest.raises(sqlite3.IntegrityError, match="injected batch write failure"):
            db.compare_and_set_meta_batch(
                {
                    "batch:rollback-first": "old-first",
                    "batch:rollback-second": "old-second",
                },
                {
                    "batch:rollback-first": "new-first",
                    "batch:rollback-second": "new-second",
                },
            )
        assert db.get_meta("batch:rollback-first") == "old-first"
        assert db.get_meta("batch:rollback-second") == "old-second"


# ──────────────────────────────────────────────────────────────────────
# Persistence failure / stale-manager contracts
# ──────────────────────────────────────────────────────────────────────


class TestPersistenceFailureContracts:
    def test_fire_tick_db_failure_emits_no_prompt_without_durable_claim(
        self, hermes_home, monkeypatch
    ):
        from hermes_cli import loops

        mgr = loops.LoopManager(session_id="fire-write-failure")
        state = mgr.set("poll", interval_seconds=60)
        state.next_due_at = time.time() - 1
        db = loops._get_session_db()

        def fail_compare(*_args, **_kwargs):
            raise OSError("injected CAS persistence failure")

        monkeypatch.setattr(db, "compare_and_set_meta_batch", fail_compare)

        assert mgr.fire_tick() is None
        assert mgr.state.ticks_fired == 0
        assert mgr.state.awaiting_response is False

    def test_stale_set_raises_instead_of_returning_unsaved_state(self, hermes_home):
        from hermes_cli import loops

        loops.LoopManager(session_id="stale-set-parent").set(
            "original", interval_seconds=60
        )
        stale = loops.LoopManager(session_id="stale-set-parent")
        assert loops.migrate_loop_to_session(
            "stale-set-parent", "stale-set-child"
        )

        with pytest.raises(loops.LoopPersistenceError, match="persist"):
            stale.set("replacement", interval_seconds=60)

        assert stale.state.status == "cleared"
        assert loops.load_loop("stale-set-parent").status == "cleared"
        assert loops.load_loop("stale-set-child").prompt == "original"

    @pytest.mark.parametrize(
        ("operation", "expected"),
        [
            ("pause", None),
            ("resume", None),
            ("clear", False),
            ("mark_done", False),
        ],
    )
    def test_stale_control_does_not_report_success_or_overwrite(
        self, hermes_home, operation, expected
    ):
        from hermes_cli import loops

        parent = f"stale-{operation}-parent"
        child = f"stale-{operation}-child"
        loops.LoopManager(session_id=parent).set("original", interval_seconds=60)
        stale = loops.LoopManager(session_id=parent)
        assert loops.migrate_loop_to_session(parent, child)

        if operation == "pause":
            result = stale.pause()
        elif operation == "resume":
            result = stale.resume()
        elif operation == "clear":
            result = stale.clear()
        else:
            result = stale.mark_done("stale completion")

        assert result is expected
        assert stale.state.status == "cleared"
        assert loops.load_loop(parent).status == "cleared"
        assert loops.load_loop(child).status == "active"

    @pytest.mark.parametrize("operation", ["release", "abandon"])
    def test_stale_inflight_control_fails_closed(
        self, hermes_home, operation
    ):
        from hermes_cli import loops

        parent = f"stale-{operation}-parent"
        child = f"stale-{operation}-child"
        owner = loops.LoopManager(session_id=parent)
        owner.set("original", interval_seconds=60)
        assert owner.fire_tick() is not None
        stale = loops.LoopManager(session_id=parent)
        assert loops.migrate_loop_to_session(parent, child)

        if operation == "release":
            result = stale.release_stale_tick_claim(
                now=stale.state.next_due_at + 1
            )
        else:
            result = stale.abandon_tick()

        assert result is False
        assert stale.state.status == "cleared"
        assert loops.load_loop(parent).status == "cleared"
        assert loops.load_loop(child).awaiting_response is True

    @pytest.mark.parametrize(
        "branch",
        ["marker", "until_done", "until_blocked", "times", "max_ticks", "continue"],
    )
    def test_every_stale_complete_tick_branch_fails_closed(
        self, hermes_home, branch
    ):
        from hermes_cli import loops

        parent = f"stale-complete-{branch}-parent"
        child = f"stale-complete-{branch}-child"
        until = "deploy healthy" if branch.startswith("until_") else ""
        times = 1 if branch == "times" else 0
        owner = loops.LoopManager(session_id=parent)
        state = owner.set(
            "original", interval_seconds=60, times=times, until=until
        )
        if branch == "max_ticks":
            state.max_ticks = 1
            loops.save_loop(parent, state)
        assert owner.fire_tick() is not None
        stale = loops.LoopManager(session_id=parent)
        assert loops.migrate_loop_to_session(parent, child)

        response = "done\nLOOP_COMPLETE" if branch == "marker" else "still running"
        judge = (
            ("done", "healthy", False, None, False)
            if branch == "until_done"
            else ("blocked", "impossible", False, None, False)
        )
        with patch("hermes_cli.goals.judge_goal", return_value=judge):
            decision = stale.complete_tick(response)

        assert decision["persisted"] is False
        assert decision["status"] == "cleared"
        assert decision["stopped"] is True
        assert decision["message"] == ""
        assert loops.load_loop(parent).status == "cleared"
        assert loops.load_loop(child).status == "active"


# ──────────────────────────────────────────────────────────────────────
# Tick lifecycle
# ──────────────────────────────────────────────────────────────────────


class TestTickLifecycle:
    def test_due_immediately_on_create(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t1")
        mgr.set("poll", interval_seconds=300)
        assert mgr.is_due() is True

    def test_due_after_interval(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        assert mgr.is_due() is True

    def test_not_due_once_rescheduled(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2a")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() + 300
        assert mgr.is_due() is False

    def test_self_paced_due_immediately_on_create(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t2c")
        state = mgr.set("keep refining")
        assert state.mode == "self_paced"
        assert mgr.is_due() is True

    def test_fire_marks_awaiting_and_blocks_refire(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t3")
        state = mgr.set("poll the build", interval_seconds=300)
        state.next_due_at = time.time() - 1
        wakeup = mgr.fire_tick()
        assert wakeup is not None
        assert "[/loop wakeup #1" in wakeup
        assert "poll the build" in wakeup
        assert "LOOP_COMPLETE" in wakeup
        assert mgr.state.awaiting_response is True
        assert mgr.is_due() is False  # can't double-fire mid-turn
        assert mgr.fire_tick() is None

    def test_slash_prompt_returned_raw(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t4")
        state = mgr.set("/recap", interval_seconds=300)
        state.next_due_at = time.time() - 1
        assert mgr.fire_tick() == "/recap"

    def test_abandon_tick_rolls_back(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t5")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.abandon_tick()
        assert mgr.state.awaiting_response is False
        assert mgr.state.ticks_fired == 0

    def test_release_stale_tick_claim_preserves_prior_tick_and_due_cadence(
        self, hermes_home
    ):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t5-stale")
        state = mgr.set("poll", interval_seconds=300)
        state.ticks_fired = 1
        state.awaiting_response = True
        state.next_due_at = time.time() - 1
        prior_due_at = state.next_due_at

        assert mgr.release_stale_tick_claim(now=time.time()) is True
        assert mgr.state.awaiting_response is False
        assert mgr.state.ticks_fired == 1
        assert mgr.state.next_due_at == prior_due_at
        assert mgr.fire_tick() is not None
        assert mgr.state.ticks_fired == 2

    @pytest.mark.parametrize(
        ("times", "max_ticks", "expected_status"),
        [(1, 100, "done"), (0, 1, "paused")],
    )
    def test_release_stale_tick_claim_honors_tick_caps(
        self, hermes_home, times, max_ticks, expected_status
    ):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id=f"t5-stale-cap-{expected_status}")
        state = mgr.set("poll", interval_seconds=300, times=times)
        state.max_ticks = max_ticks
        state.ticks_fired = 1
        state.awaiting_response = True
        state.next_due_at = time.time() - 1

        assert mgr.release_stale_tick_claim(now=time.time()) is True
        assert mgr.state.status == expected_status
        assert mgr.state.awaiting_response is False
        assert mgr.state.ticks_fired == 1
        assert mgr.fire_tick() is None

    def test_complete_tick_marker_stops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t6")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("The deploy is live.\nLOOP_COMPLETE")
        assert decision["stopped"] is True
        assert decision["status"] == "done"

    def test_complete_tick_times_cap(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t7")
        state = mgr.set("poll", interval_seconds=300, times=1)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is True
        assert decision["status"] == "done"
        assert "1/1" in decision["message"]

    def test_complete_tick_continues_and_reschedules(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t8")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is False
        assert decision["status"] == "active"
        assert mgr.state.awaiting_response is False
        assert mgr.state.next_due_at > time.time() + 250

    def test_complete_tick_max_ticks_pauses(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t9")
        state = mgr.set("poll", interval_seconds=300)
        state.max_ticks = 1
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        decision = mgr.complete_tick("still building")
        assert decision["stopped"] is True
        assert decision["status"] == "paused"

    def test_until_judge_done_stops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t10")
        state = mgr.set("poll", interval_seconds=300, until="the suite is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("done", "suite green", False, None, False)):
            decision = mgr.complete_tick("All 500 tests passed.")
        assert decision["stopped"] is True
        assert decision["status"] == "done"

    def test_until_judge_continue_loops(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t11")
        state = mgr.set("poll", interval_seconds=300, until="the suite is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("continue", "3 failures", False, None, False)):
            decision = mgr.complete_tick("3 tests still failing")
        assert decision["stopped"] is False

    def test_until_judge_blocked_pauses(self, hermes_home):
        """An unachievable stop condition pauses the loop instead of spinning to the tick budget."""
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t11b")
        state = mgr.set("poll", interval_seconds=300, until="the deleted repo's CI is green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", return_value=("blocked", "repo no longer exists", False, None, False)):
            decision = mgr.complete_tick("The repository was deleted; there is no CI to watch.")
        assert decision["stopped"] is True
        assert decision["status"] == "paused"
        assert "unachievable" in decision["message"]

    def test_until_judge_error_fails_open(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="t12")
        state = mgr.set("poll", interval_seconds=300, until="green")
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        with patch("hermes_cli.goals.judge_goal", side_effect=RuntimeError("api down")):
            decision = mgr.complete_tick("some output")
        assert decision["stopped"] is False  # fail-open: keep looping


class TestSelfPacedBackoff:
    def test_backoff_doubles_on_unchanged_and_resets_on_change(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="sp1")
        state = mgr.set("watch the queue")  # self-paced
        floor = state.current_delay
        assert state.mode == "self_paced"

        # Tick 1: response A → delay stays at floor (change from empty digest).
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor

        # Tick 2: same response → backoff doubles.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor * 2

        # Tick 3: same again → doubles again.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 5")
        assert mgr.state.current_delay == floor * 4

        # Tick 4: changed response → snaps back to floor.
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("queue depth is 2 — draining")
        assert mgr.state.current_delay == floor

    def test_timestamp_only_changes_do_not_reset(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="sp2")
        state = mgr.set("watch")
        floor = state.current_delay
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("Still building. Checked at 14:02:33")
        mgr.state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.complete_tick("Still building. Checked at 14:07:33")
        assert mgr.state.current_delay == floor * 2  # digest ignored the clock


# ──────────────────────────────────────────────────────────────────────
# Controls (pause/resume/clear) + min interval + status
# ──────────────────────────────────────────────────────────────────────


class TestControls:
    def test_min_interval_enforced(self, hermes_home):
        from hermes_cli.loops import LoopManager, min_interval_seconds

        mgr = LoopManager(session_id="c1")
        state = mgr.set("poll", interval_seconds=1)
        assert state.interval_seconds >= min_interval_seconds()

    def test_pause_resume(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c2")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.pause()
        assert mgr.is_active() is False
        assert mgr.is_due() is False
        mgr.resume()
        assert mgr.is_active() is True

    def test_paused_mid_tick_clears_awaiting(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c3")
        state = mgr.set("poll", interval_seconds=300)
        state.next_due_at = time.time() - 1
        mgr.fire_tick()
        mgr.pause(reason="user-interrupted")
        assert mgr.state.awaiting_response is False

    def test_status_line_shapes(self, hermes_home):
        from hermes_cli.loops import LoopManager

        mgr = LoopManager(session_id="c4")
        assert "No loop set" in mgr.status_line()
        mgr.set("poll the build", interval_seconds=300)
        assert "active" in mgr.status_line()
        assert "poll the build" in mgr.status_line()
        mgr.pause()
        assert "paused" in mgr.status_line()
        mgr.clear()
        assert "No loop set" in mgr.status_line()


# ──────────────────────────────────────────────────────────────────────
# /goal mixing
# ──────────────────────────────────────────────────────────────────────


class TestGoalMixing:
    def test_active_goal_blocks_tick(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        GoalManager(session_id="g1").set("finish the migration")
        assert goal_blocks_loop_tick("g1") is True

    def test_no_goal_does_not_block(self, hermes_home):
        from hermes_cli.loops import goal_blocks_loop_tick

        assert goal_blocks_loop_tick("g2") is False

    def test_paused_goal_does_not_block(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        gm = GoalManager(session_id="g3")
        gm.set("finish it")
        gm.pause()
        assert goal_blocks_loop_tick("g3") is False

    def test_parked_goal_does_not_block(self, hermes_home):
        from hermes_cli.goals import GoalManager
        from hermes_cli.loops import goal_blocks_loop_tick

        gm = GoalManager(session_id="g4")
        gm.set("finish it")
        gm.wait_for_seconds(3600, reason="waiting on CI")
        assert goal_blocks_loop_tick("g4") is False


# ──────────────────────────────────────────────────────────────────────
# dispatch_loop_command (shared slash handler)
# ──────────────────────────────────────────────────────────────────────


class TestDispatchLoopCommand:
    def test_create_fixed(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d1")
        result = dispatch_loop_command(mgr, "5m check the deploy")
        assert result["created"] is True
        assert "Loop set" in result["output"]
        assert "every 5m" in result["output"]

    def test_create_self_paced(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d2")
        result = dispatch_loop_command(mgr, "keep fixing the tests")
        assert result["created"] is True
        assert "Self-paced" in result["output"]

    def test_create_fires_immediately(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d2a")
        result = dispatch_loop_command(mgr, "1h check the deploy")
        assert result["created"] is True
        assert "Loop set" in result["output"]
        assert "fires now" in result["output"]
        assert mgr.is_due() is True

    def test_status_empty(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d3")
        result = dispatch_loop_command(mgr, "")
        assert result["created"] is False
        assert "No loop set" in result["output"]

    def test_pause_resume_stop(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d4")
        dispatch_loop_command(mgr, "5m poll")
        assert "paused" in dispatch_loop_command(mgr, "pause")["output"].lower()
        assert "resumed" in dispatch_loop_command(mgr, "resume")["output"].lower()
        assert "stopped" in dispatch_loop_command(mgr, "stop")["output"].lower()
        assert "No active loop" in dispatch_loop_command(mgr, "stop")["output"]

    def test_route_stored(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command, load_loop

        mgr = LoopManager(session_id="d5")
        route = {"platform": "telegram", "chat_id": "42", "chat_type": "private"}
        dispatch_loop_command(mgr, "5m ping", route=route)
        assert load_loop("d5").route == route

    def test_help(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d6")
        out = dispatch_loop_command(mgr, "help")["output"]
        assert "Usage" in out
        assert "--times" in out

    def test_bad_times_error(self, hermes_home):
        from hermes_cli.loops import LoopManager, dispatch_loop_command

        mgr = LoopManager(session_id="d7")
        result = dispatch_loop_command(mgr, "5m poll --times banana")
        assert result["created"] is False
        assert "--times" in result["output"]


# ──────────────────────────────────────────────────────────────────────
# Command registry
# ──────────────────────────────────────────────────────────────────────


class TestCommandRegistry:
    def test_loop_registered_with_proactive_alias(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("loop")
        assert cmd is not None
        assert cmd.name == "loop"
        alias = resolve_command("proactive")
        assert alias is not None
        assert alias.name == "loop"


# ──────────────────────────────────────────────────────────────────────
# SessionDB.list_meta_prefix
# ──────────────────────────────────────────────────────────────────────


class TestListMetaPrefix:
    def test_prefix_scan(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        db.set_meta("lmp-test:aaa", "1")
        db.set_meta("lmp-test:bbb", "2")
        db.set_meta("other:aaa", "3")
        rows = dict(db.list_meta_prefix("lmp-test:"))
        assert rows == {"lmp-test:aaa": "1", "lmp-test:bbb": "2"}

    def test_wildcards_escaped(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        db.set_meta("pre%fix:x", "1")
        db.set_meta("prefix:y", "2")
        assert db.list_meta_prefix("pre%") == [("pre%fix:x", "1")]

    def test_empty_prefix(self, hermes_home):
        from hermes_state import SessionDB

        db = SessionDB()
        assert db.list_meta_prefix("") == []
