"""Gateway /loop command tests — dispatch, routing capture, mid-run guard."""

import asyncio
import logging
import threading
import time
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.profile_routing import ProfileRoute
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from hermes_constants import get_hermes_home
from hermes_cli import goals, loops


class _FakeSessionEntry:
    session_id = "sid-gateway-loop"


class _FakeSessionStore:
    def __init__(self):
        self.entry = _FakeSessionEntry()

    def get_or_create_session(self, source, *, touch_activity=True):
        return self.entry

    def _generate_session_key(self, source):
        return "agent:main:discord:channel:loop-test"


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from this sync (non-loop) context. Inside
    # the async tests, a cold cache makes GoalManager.set() kick the bounded
    # background bootstrap (loop-thread path) and wait only
    # _DB_BOOTSTRAP_INIT_WAIT_S — on a loaded CI runner the init overruns the
    # window, the goal is never persisted, and the active-goal assertion
    # flakes (main run 33455779041). Warming here removes the race entirely.
    goals._get_session_db()
    yield home
    goals._DB_CACHE.clear()


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="token")}
    )
    runner.session_store = _FakeSessionStore()
    runner.adapters = {}
    runner._queued_events = {}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chat-loop",
            chat_type="channel",
            thread_id="thread-9",
            user_id="user-loop",
        ),
        message_id="msg-loop",
    )


@pytest.mark.asyncio
async def test_gateway_loop_create_captures_route(loop_env):
    runner = _make_runner()
    event = _make_event("/loop 5m check the deploy")
    event.source.scope_id = "guild-7"
    event.source.guild_id = "guild-7"
    event.source.parent_chat_id = "parent-channel"
    event.source.profile = "coder"

    response = await GatewayRunner._handle_loop_command(runner, event)
    assert "Loop set" in response
    assert "every 5m" in response

    state = loops.load_loop("sid-gateway-loop")
    assert state is not None
    assert state.prompt == "check the deploy"
    assert state.route["platform"] == "discord"
    assert state.route["chat_id"] == "chat-loop"
    assert state.route["thread_id"] == "thread-9"
    assert state.route["scope_id"] == "guild-7"
    assert state.route["guild_id"] == "guild-7"
    assert state.route["parent_chat_id"] == "parent-channel"
    assert state.route["profile"] == "coder"


@pytest.mark.asyncio
async def test_gateway_loop_status_pause_stop(loop_env):
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))

    status = await GatewayRunner._handle_loop_command(runner, _make_event("/loop status"))
    assert "poll CI" in status

    paused = await GatewayRunner._handle_loop_command(runner, _make_event("/loop pause"))
    assert "paused" in paused.lower()

    stopped = await GatewayRunner._handle_loop_command(runner, _make_event("/loop stop"))
    assert "stopped" in stopped.lower()


@pytest.mark.asyncio
async def test_gateway_loop_goal_note_when_goal_active(loop_env):
    from hermes_cli.goals import GoalManager

    GoalManager(session_id="sid-gateway-loop").set("finish the migration")
    runner = _make_runner()
    response = await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))
    assert "active /goal" in response


@pytest.mark.asyncio
async def test_post_turn_loop_completion_completes_inflight_tick(loop_env):
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))

    mgr = loops.LoopManager(session_id="sid-gateway-loop")
    mgr.state.next_due_at = time.time() - 1
    assert mgr.fire_tick() is not None

    entry = _FakeSessionEntry()
    await GatewayRunner._post_turn_loop_completion(
        runner,
        session_entry=entry,
        source=None,
        final_response="CI is done.\nLOOP_COMPLETE",
    )
    reloaded = loops.load_loop("sid-gateway-loop")
    assert reloaded.status == "done"


@pytest.mark.asyncio
async def test_post_turn_loop_completion_noop_without_inflight_tick(loop_env):
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))
    entry = _FakeSessionEntry()
    # No tick fired — the ordinary user turn must not consume loop state.
    await GatewayRunner._post_turn_loop_completion(
        runner,
        session_entry=entry,
        source=None,
        final_response="regular reply LOOP_COMPLETE",
    )
    reloaded = loops.load_loop("sid-gateway-loop")
    assert reloaded.status == "active"
    assert reloaded.ticks_fired == 0


def test_streamed_already_sent_none_recovers_text_for_hooks():
    """Streamed turns return None. Hooks must still see the delivered reply."""
    event = _make_event("wakeup")
    event._streamed_final_response = "CI is green.\nLOOP_COMPLETE"
    assert GatewayRunner._final_text_for_post_turn_hooks(None, event) == (
        "CI is green.\nLOOP_COMPLETE"
    )
    assert GatewayRunner._final_text_for_post_turn_hooks(None, _make_event("x")) == ""
    assert (
        GatewayRunner._final_text_for_post_turn_hooks(
            {"final_response": "from dict"}, event
        )
        == "from dict"
    )


@pytest.mark.asyncio
async def test_streamed_already_sent_completes_loop_tick(loop_env):
    """A streamed wakeup must not leave awaiting_response stuck."""
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))

    mgr = loops.LoopManager(session_id="sid-gateway-loop")
    mgr.state.next_due_at = time.time() - 1
    assert mgr.fire_tick() is not None
    assert mgr.state.awaiting_response is True
    assert mgr.is_due() is False

    event = _make_event("wakeup")
    event._streamed_final_response = "CI is done.\nLOOP_COMPLETE"
    # Same inputs the already_sent branch leaves for _handle_message.
    final_text = GatewayRunner._final_text_for_post_turn_hooks(None, event)
    assert final_text.strip()

    await GatewayRunner._post_turn_loop_completion(
        runner,
        session_entry=_FakeSessionEntry(),
        source=None,
        final_response=final_text,
    )
    reloaded = loops.load_loop("sid-gateway-loop")
    assert reloaded.awaiting_response is False
    assert reloaded.status == "done"


@pytest.mark.asyncio
async def test_empty_agent_result_releases_inflight_loop_tick(loop_env):
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))

    mgr = loops.LoopManager(session_id="sid-gateway-loop")
    mgr.state.next_due_at = time.time() - 1
    assert mgr.fire_tick() is not None
    assert mgr.state.awaiting_response is True

    runner._post_turn_goal_continuation = AsyncMock()
    await GatewayRunner._run_post_turn_hooks(
        runner,
        agent_result={"final_response": ""},
        source=_make_event("wakeup").source,
        is_internal=True,
    )

    runner._post_turn_goal_continuation.assert_not_awaited()
    reloaded = loops.load_loop("sid-gateway-loop")
    assert reloaded.awaiting_response is False
    assert reloaded.status == "active"
    assert reloaded.next_due_at > time.time()


@pytest.mark.asyncio
async def test_goal_hook_failure_does_not_block_loop_completion(loop_env, caplog):
    runner = _make_runner()
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))

    mgr = loops.LoopManager(session_id="sid-gateway-loop")
    mgr.state.next_due_at = time.time() - 1
    assert mgr.fire_tick() is not None

    runner._post_turn_goal_continuation = AsyncMock(side_effect=RuntimeError("judge failed"))
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        await GatewayRunner._run_post_turn_hooks(
            runner,
            agent_result={"final_response": "still working"},
            source=_make_event("wakeup").source,
            is_internal=True,
        )

    reloaded = loops.load_loop("sid-gateway-loop")
    assert reloaded.awaiting_response is False
    assert "goal continuation hook failed: judge failed" in caplog.text


@pytest.mark.asyncio
async def test_post_turn_loop_completion_persists_in_multiplex_profile(
    loop_env, monkeypatch
):
    """Completion must clear the same profile DB that the watcher claimed."""
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "coder"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)

    with gateway_run._profile_runtime_scope(profile_home):
        await asyncio.to_thread(goals._get_session_db)
        mgr = loops.LoopManager(session_id="sid-gateway-loop")
        state = mgr.set("poll CI", interval_seconds=300)
        state.next_due_at = time.time() - 1
        assert mgr.fire_tick() is not None
        assert state.awaiting_response is True

    runner = _make_runner()
    runner.config.multiplex_profiles = True
    runner.config.multiplex_profile_allowlist = [profile_name]
    source = _make_event("wakeup").source
    source.profile = profile_name

    await GatewayRunner._run_post_turn_hooks(
        runner,
        agent_result={"final_response": "still working"},
        source=source,
        is_internal=True,
    )

    with gateway_run._profile_runtime_scope(profile_home):
        reloaded = loops.load_loop("sid-gateway-loop")
        assert reloaded is not None
        assert reloaded.awaiting_response is False
        assert reloaded.status == "active"
        assert reloaded.next_due_at > time.time()
        assert loops.LoopManager(session_id="sid-gateway-loop").is_due(
            now=reloaded.next_due_at
        )
    assert loops.load_loop("sid-gateway-loop") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary_live", [True, False], ids=["live", "missing"])
async def test_loop_wakeup_watcher_scopes_secondary_profile_and_adapter(
    loop_env, monkeypatch, secondary_live
):
    import hermes_state

    # The suite-wide isolation fixture pins every SessionDB to one temp path.
    # Restore dynamic HERMES_HOME resolution for this profile-scope regression.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "coder"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)
    session_id = "secondary-loop"

    # Persist the only active loop in the secondary profile's existing store.
    with gateway_run._profile_runtime_scope(profile_home):
        await asyncio.to_thread(goals._get_session_db)
        manager = loops.LoopManager(session_id=session_id)
        state = manager.set(
            "check the secondary deploy",
            interval_seconds=300,
            route={
                "platform": "discord",
                "chat_id": "secondary-channel",
                "chat_type": "channel",
                "user_id": "secondary-user",
                "profile": profile_name,
            },
        )
        state.next_due_at = time.time() - 1
        loops.save_loop(session_id, state)

    assert get_hermes_home() == loop_env
    await asyncio.to_thread(goals._get_session_db)
    assert loops.load_loop(session_id) is None

    default_adapter = Mock()
    default_adapter.handle_message = AsyncMock()
    secondary_adapter = Mock()
    secondary_adapter.handle_message = AsyncMock()

    runner = _make_runner()
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=[profile_name],
    )
    runner.session_store = None
    runner.adapters = {Platform.DISCORD: default_adapter}
    runner._profile_adapters = (
        {profile_name: {Platform.DISCORD: secondary_adapter}}
        if secondary_live
        else {}
    )
    runner._running_agents = {}
    runner._running = True

    warmed_homes = []

    async def _record_warm(_label):
        warmed_homes.append(get_hermes_home())

    runner._warm_goals_session_db = _record_warm

    sleep_calls = 0

    async def _finish_after_one_scan(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            runner._running = False

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

    await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

    default_adapter.handle_message.assert_not_awaited()
    assert warmed_homes == [loop_env, profile_home]
    if not secondary_live:
        secondary_adapter.handle_message.assert_not_awaited()
        with gateway_run._profile_runtime_scope(profile_home):
            secondary_state = loops.load_loop(session_id)
            assert secondary_state is not None
            assert secondary_state.awaiting_response is False
            assert secondary_state.ticks_fired == 0
        assert loops.load_loop(session_id) is None
        return

    secondary_adapter.handle_message.assert_awaited_once()
    event = secondary_adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.source.profile == profile_name
    assert event.source.thread_id is None
    assert runner._session_key_for_source(event.source).startswith(
        f"agent:{profile_name}:discord:channel:secondary-channel"
    )

    with gateway_run._profile_runtime_scope(profile_home):
        secondary_state = loops.load_loop(session_id)
        assert secondary_state is not None
        assert secondary_state.awaiting_response is True
        assert secondary_state.ticks_fired == 1
    assert loops.load_loop(session_id) is None


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_ignores_foreign_profile_row_in_launch_db(
    loop_env, monkeypatch
):
    """A copied legacy row without local ownership proof must fail closed."""
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "coder"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)
    session_id = "foreign-profile-loop"

    await asyncio.to_thread(goals._get_session_db)
    manager = loops.LoopManager(session_id=session_id)
    state = manager.set(
        "must stay in coder",
        interval_seconds=300,
        route={
            "platform": "discord",
            "chat_id": "coder-channel",
            "chat_type": "channel",
            "user_id": "coder-user",
        },
    )
    state.ticks_fired = 1
    state.awaiting_response = True
    state.next_due_at = time.time() - 1
    loops.save_loop(session_id, state)

    shared_adapter = Mock()
    shared_adapter.handle_message = AsyncMock()
    runner = _make_runner()
    runner.config = GatewayConfig(
        multiplex_profiles=True,
        multiplex_profile_allowlist=[profile_name],
    )
    runner.session_store = None
    runner.adapters = {Platform.DISCORD: shared_adapter}
    runner._profile_adapters = {}
    runner._running_agents = {}
    runner._running = True
    runner._warm_goals_session_db = AsyncMock()

    sleep_calls = 0

    async def _finish_after_one_scan(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            runner._running = False

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

    await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

    shared_adapter.handle_message.assert_not_awaited()
    reloaded = loops.load_loop(session_id)
    assert reloaded is not None
    assert reloaded.awaiting_response is True
    assert reloaded.ticks_fired == 1
    with gateway_run._profile_runtime_scope(profile_home):
        await asyncio.to_thread(goals._get_session_db)
        assert loops.load_loop(session_id) is None


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_uses_named_launch_profile_as_scan_owner(
    loop_env, monkeypatch
):
    """The unscoped launch scan may itself belong to a named profile."""
    import hermes_state
    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "launch-profile"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)
    session_id = "session-launch-profile-loop"
    token = set_hermes_home_override(profile_home)
    try:
        await asyncio.to_thread(goals._get_session_db)
        manager = loops.LoopManager(session_id)
        state = manager.set(
            "continue launch work",
            interval_seconds=60,
            route={
                "platform": "discord",
                "chat_id": "launch-channel",
                "chat_type": "channel",
                "user_id": "launch-user",
                "profile": profile_name,
            },
        )
        state.next_due_at = time.time() - 1
        loops.save_loop(session_id, state)

        shared_adapter = Mock()
        shared_adapter.handle_message = AsyncMock()
        runner = _make_runner()
        runner.config = GatewayConfig(
            multiplex_profiles=True,
            multiplex_profile_allowlist=[],
        )
        runner.session_store = None
        runner.adapters = {Platform.DISCORD: shared_adapter}
        runner._profile_adapters = {}
        runner._running_agents = {}
        runner._running = True
        runner._warm_goals_session_db = AsyncMock()

        sleep_calls = 0

        async def _finish_after_one_scan(_delay):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                runner._running = False

        monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

        await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

        shared_adapter.handle_message.assert_awaited_once()
        event = shared_adapter.handle_message.await_args.args[0]
        assert event.source.profile == profile_name
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_uses_shared_adapter_for_matching_profile_route(
    loop_env, monkeypatch
):
    import hermes_state

    # Match the restart topology: the loop route predates richer scope capture,
    # while the durable session origin still has the complete routed identity.
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "coder"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)
    sessions_dir = loop_env / "gateway-sessions"
    config = GatewayConfig(
        sessions_dir=sessions_dir,
        multiplex_profiles=True,
        multiplex_profile_allowlist=[profile_name],
        profile_routes=[
            ProfileRoute(
                name="coder-discord",
                platform="discord",
                profile=profile_name,
                guild_id="guild-7",
                chat_id="secondary-channel",
            )
        ],
    )
    persisted_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="secondary-channel",
        chat_type="group",
        thread_id=None,
        user_id="secondary-user",
        scope_id="guild-7",
        parent_chat_id=None,
        profile=profile_name,
    )

    with gateway_run._profile_runtime_scope(profile_home):
        await asyncio.to_thread(goals._get_session_db)
        pre_restart_store = SessionStore(sessions_dir, config)
        session_entry = pre_restart_store.get_or_create_session(persisted_source)
        session_key = session_entry.session_key
        manager = loops.LoopManager(session_id=session_entry.session_id)
        state = manager.set(
            "check the routed deploy",
            interval_seconds=300,
            route={
                "platform": "discord",
                "chat_id": "secondary-channel",
                "chat_type": "group",
                "user_id": "secondary-user",
            },
        )
        state.next_due_at = time.time() - 1
        loops.save_loop(session_entry.session_id, state)
        pre_restart_store.close_all_db_handles()

    shared_adapter = Mock()
    shared_adapter.handle_message = AsyncMock()

    runner = _make_runner()
    runner.config = config
    runner.session_store = SessionStore(sessions_dir, config)
    runner.adapters = {Platform.DISCORD: shared_adapter}
    runner._profile_adapters = {}
    runner._running_agents = {}
    runner._running = True
    runner._warm_goals_session_db = AsyncMock()

    sleep_calls = 0

    async def _finish_after_one_scan(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            runner._running = False

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

    try:
        await GatewayRunner._loop_wakeup_watcher(runner, interval=0)
    finally:
        runner.session_store.close_all_db_handles()

    shared_adapter.handle_message.assert_awaited_once()
    event = shared_adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.source.profile == profile_name
    assert event.source.scope_id == "guild-7"
    assert event.source.guild_id == "guild-7"
    assert event.source.thread_id is None
    assert runner._session_key_for_source(event.source) == session_key
    assert event.source._transport_adapter_ref() is shared_adapter
    assert runner._adapter_for_source(event.source) is shared_adapter


@pytest.mark.asyncio
async def test_loop_wakeup_keeps_transport_owner_when_route_targets_another_profile(
    loop_env, monkeypatch
):
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    runtime_profile = "coder"
    transport_profile = "alerts"
    runtime_home = loop_env / "profiles" / runtime_profile
    runtime_home.mkdir(parents=True)
    (loop_env / "profiles" / transport_profile).mkdir(parents=True)
    sessions_dir = loop_env / "gateway-sessions"
    config = GatewayConfig(
        sessions_dir=sessions_dir,
        multiplex_profiles=True,
        multiplex_profile_allowlist=[runtime_profile, transport_profile],
        profile_routes=[
            ProfileRoute(
                name="coder-discord",
                platform="discord",
                profile=runtime_profile,
                guild_id="guild-7",
                chat_id="secondary-channel",
            )
        ],
    )
    primary_adapter = Mock()
    primary_adapter.handle_message = AsyncMock()
    primary_adapter._pending_messages = {}
    primary_adapter._session_tasks = {}
    secondary_adapter = Mock()
    secondary_adapter.handle_message = AsyncMock()
    secondary_adapter._pending_messages = {}
    secondary_adapter._session_tasks = {}

    runner = _make_runner()
    runner.config = config
    runner.session_store = SessionStore(sessions_dir, config)
    runner.adapters = {Platform.DISCORD: primary_adapter}
    runner._profile_adapters = {
        runtime_profile: {},
        transport_profile: {Platform.DISCORD: secondary_adapter},
    }
    runner._running_agents = {}
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="secondary-channel",
        chat_type="group",
        user_id="secondary-user",
        scope_id="guild-7",
        profile=runtime_profile,
    )
    source._transport_adapter_ref = weakref.ref(secondary_adapter)
    event = _make_event("/loop 5m /status")
    event.source = source

    with gateway_run._profile_runtime_scope(runtime_home):
        await asyncio.to_thread(goals._get_session_db)
        await GatewayRunner._handle_loop_command(runner, event)
        session_entry = runner.session_store.get_or_create_session(source)
        persisted = loops.load_loop(session_entry.session_id)
        assert persisted.route["adapter_profile"] == transport_profile
        persisted.next_due_at = time.time() - 1
        loops.save_loop(session_entry.session_id, persisted)

        # Legacy rows have no persisted transport owner. While the original
        # source is still live, its adapter provenance remains authoritative.
        legacy_scan_state = loops.LoopState.from_json(persisted.to_json())
        legacy_scan_state.route.pop("adapter_profile")
        await GatewayRunner._loop_wakeup_fire_one(
            runner,
            session_entry.session_id,
            legacy_scan_state,
            time.time(),
            set(),
            set(),
            runtime_profile,
        )

    secondary_adapter.handle_message.assert_awaited_once()
    primary_adapter.handle_message.assert_not_awaited()

    # A restart drops the in-process weakref. The persisted owner profile must
    # recover the same transport independently of the routed runtime profile.
    runner.session_store.close_all_db_handles()
    restarted = _make_runner()
    restarted.config = config
    restarted.session_store = SessionStore(sessions_dir, config)
    restarted.adapters = {Platform.DISCORD: primary_adapter}
    restarted._profile_adapters = runner._profile_adapters
    restarted._running_agents = {}
    with gateway_run._profile_runtime_scope(runtime_home):
        await asyncio.to_thread(goals._get_session_db)
        due = loops.load_loop(session_entry.session_id)
        due.next_due_at = time.time() - 1
        loops.save_loop(session_entry.session_id, due)
        await GatewayRunner._loop_wakeup_fire_one(
            restarted,
            session_entry.session_id,
            due,
            time.time(),
            set(),
            set(),
            runtime_profile,
        )
    restarted.session_store.close_all_db_handles()

    assert secondary_adapter.handle_message.await_count == 2
    primary_adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("injection", ["complete", "abandon"])
async def test_slash_loop_settlement_follows_session_rotation(loop_env, injection):
    config = GatewayConfig(sessions_dir=loop_env / "gateway-sessions")
    runner = _make_runner()
    runner.config = config
    runner.session_store = SessionStore(config.sessions_dir, config)
    runner._running_agents = {}
    source = _make_event("wakeup").source
    session_entry = runner.session_store.get_or_create_session(source)
    session_key = session_entry.session_key
    successors = []

    async def _rotate_during_slash_dispatch(_event):
        successors.append(await runner.async_session_store.reset_session(session_key))
        if injection == "abandon":
            raise RuntimeError("slash dispatch failed after rotation")

    adapter = Mock()
    adapter.handle_message = AsyncMock(side_effect=_rotate_during_slash_dispatch)
    adapter._pending_messages = {}
    adapter._session_tasks = {}
    runner.adapters = {Platform.DISCORD: adapter}

    manager = loops.LoopManager(session_id=session_entry.session_id)
    state = manager.set(
        "/new",
        interval_seconds=300,
        route={
            "platform": "discord",
            "chat_id": source.chat_id,
            "chat_type": source.chat_type,
            "thread_id": source.thread_id,
            "user_id": source.user_id,
        },
    )
    state.next_due_at = time.time() - 1
    loops.save_loop(session_entry.session_id, state)

    await GatewayRunner._loop_wakeup_fire_one(
        runner, session_entry.session_id, state, time.time(), set()
    )

    successor = successors[0]
    old_state = loops.load_loop(session_entry.session_id)
    successor_state = loops.load_loop(successor.session_id)
    assert old_state.migrated_to == successor.session_id
    assert successor_state.awaiting_response is False
    assert successor_state.ticks_fired == (1 if injection == "complete" else 0)


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_reclaims_due_idle_awaiting_tick(
    loop_env, monkeypatch
):
    """A terminal-path bookkeeping miss must not wedge an idle due loop."""
    import hermes_state

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    profile_name = "coder"
    profile_home = loop_env / "profiles" / profile_name
    profile_home.mkdir(parents=True)
    sessions_dir = loop_env / "gateway-sessions"
    config = GatewayConfig(
        sessions_dir=sessions_dir,
        multiplex_profiles=True,
        multiplex_profile_allowlist=[profile_name],
        profile_routes=[
            ProfileRoute(
                name="coder-discord",
                platform="discord",
                profile=profile_name,
                guild_id="guild-7",
                chat_id="secondary-channel",
            )
        ],
    )
    persisted_source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="secondary-channel",
        chat_type="group",
        user_id="secondary-user",
        scope_id="guild-7",
        profile=profile_name,
    )

    with gateway_run._profile_runtime_scope(profile_home):
        await asyncio.to_thread(goals._get_session_db)
        store = SessionStore(sessions_dir, config)
        session_entry = store.get_or_create_session(persisted_source)
        manager = loops.LoopManager(session_id=session_entry.session_id)
        state = manager.set(
            "check the routed deploy",
            interval_seconds=300,
            route={
                "platform": "discord",
                "chat_id": "secondary-channel",
                "chat_type": "group",
                "user_id": "secondary-user",
            },
        )
        state.ticks_fired = 1
        state.awaiting_response = True
        state.last_fired_at = time.time() - 600
        state.next_due_at = time.time() - 300
        loops.save_loop(session_entry.session_id, state)
        store.close_all_db_handles()

    shared_adapter = Mock()
    shared_adapter.handle_message = AsyncMock()
    shared_adapter._pending_messages = {}

    runner = _make_runner()
    runner.config = config
    runner.session_store = SessionStore(sessions_dir, config)
    runner.adapters = {Platform.DISCORD: shared_adapter}
    runner._profile_adapters = {}
    runner._running_agents = {}
    runner._running = True
    runner._warm_goals_session_db = AsyncMock()

    sleep_calls = 0

    async def _finish_after_one_scan(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            runner._running = False

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

    try:
        await GatewayRunner._loop_wakeup_watcher(runner, interval=0)
    finally:
        runner.session_store.close_all_db_handles()

    shared_adapter.handle_message.assert_awaited_once()
    with gateway_run._profile_runtime_scope(profile_home):
        recovered = loops.load_loop(session_entry.session_id)
        assert recovered is not None
        assert recovered.awaiting_response is True
        assert recovered.ticks_fired == 2
        assert recovered.last_fired_at > state.last_fired_at
    assert loops.load_loop(session_entry.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocker",
    ["active-runner", "pending-head", "queued-overflow", "adapter-owner-task"],
)
async def test_loop_wakeup_watcher_does_not_reclaim_busy_or_pending_tick(
    loop_env, monkeypatch, blocker
):
    """A live turn or FIFO input keeps ownership of an awaiting loop claim."""
    manager = loops.LoopManager(session_id="sid-gateway-loop")
    state = manager.set(
        "check the deploy",
        interval_seconds=300,
        route={
            "platform": "discord",
            "chat_id": "chat-loop",
            "chat_type": "channel",
            "user_id": "user-loop",
        },
    )
    state.ticks_fired = 1
    state.awaiting_response = True
    state.last_fired_at = time.time() - 600
    state.next_due_at = time.time() - 300
    loops.save_loop("sid-gateway-loop", state)

    adapter = Mock()
    adapter.handle_message = AsyncMock()
    adapter._pending_messages = {}
    adapter._session_tasks = {}

    runner = _make_runner()
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running_agents = {}
    runner._running = True
    runner._warm_goals_session_db = AsyncMock()
    session_key = runner.session_store._generate_session_key(None)
    pending_event = _make_event("user input wins")
    if blocker == "active-runner":
        runner._running_agents[session_key] = object()
    elif blocker == "pending-head":
        adapter._pending_messages[session_key] = pending_event
    elif blocker == "queued-overflow":
        runner._queued_events[session_key] = [pending_event]
    else:
        owner_task = Mock()
        owner_task.done.return_value = False
        adapter._session_tasks[session_key] = owner_task

    sleep_calls = 0

    async def _finish_after_one_scan(_delay):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            runner._running = False

    monkeypatch.setattr(gateway_run.asyncio, "sleep", _finish_after_one_scan)

    await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

    adapter.handle_message.assert_not_awaited()
    retained = loops.load_loop("sid-gateway-loop")
    assert retained is not None
    assert retained.awaiting_response is True
    assert retained.ticks_fired == 1
    assert retained.last_fired_at == state.last_fired_at


@pytest.mark.asyncio
async def test_post_turn_session_resolution_failure_is_logged(loop_env, caplog):
    runner = _make_runner()
    runner.session_store.get_or_create_session = Mock(side_effect=RuntimeError("store unavailable"))

    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        await GatewayRunner._run_post_turn_hooks(
            runner,
            agent_result={"final_response": ""},
            source=_make_event("wakeup").source,
            is_internal=True,
        )

    assert "post-turn session resolution failed: store unavailable" in caplog.text


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_keeps_event_loop_responsive_under_writer_lock(loop_env):
    """The wakeup scan's SessionDB calls (list_active_loops / fire_tick /
    complete_tick) must run off the loop thread. A slow writer holding the
    SessionDB writer lock used to block the whole gateway event loop for the
    duration of the hold (#92413)."""
    runner = _make_runner()
    runner._running = True
    runner._running_agents = {}
    runner.adapters = {}

    # Persist an active loop that is due now, routed to a platform with no
    # adapter (so the scan exits after list_active_loops(), before fire_tick).
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m poll CI"))
    state = loops.load_loop("sid-gateway-loop")
    state.next_due_at = time.time() - 1
    loops.save_loop("sid-gateway-loop", state)

    db = loops._get_session_db()
    hold_s = 0.6
    released = threading.Event()

    def _hold_writer_lock():
        with db._lock:
            time.sleep(hold_s)
        released.set()

    # Force the read path onto the writer lock (non-WAL degradation) so the
    # test binds the offload regardless of the host SQLite's WAL support.
    db._wal_active = False

    # One scan only: patch asyncio.sleep inside the watcher to stop the loop
    # after the first iteration.
    orig_sleep = asyncio.sleep
    calls = {"n": 0}

    async def _one_pass_sleep(delay):
        calls["n"] += 1
        if calls["n"] >= 2:  # first call is the 5s connect grace, second ends the scan
            runner._running = False
        return await orig_sleep(0)

    holder = threading.Thread(target=_hold_writer_lock)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", _one_pass_sleep)
        holder.start()
        time.sleep(0.05)  # ensure the lock is held before the scan starts
        watcher = asyncio.ensure_future(GatewayRunner._loop_wakeup_watcher(runner, interval=0))

        # Heartbeat coroutine: measures the longest gap between loop turns
        # while the watcher is (supposedly) blocked in the executor.
        gaps = []
        last = time.monotonic()
        while not watcher.done():
            await orig_sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now
        await watcher
    holder.join()

    assert released.is_set()
    # If the DB call ran on the loop thread, one heartbeat gap would be ~hold_s.
    assert max(gaps) < hold_s / 2, f"event loop stalled for {max(gaps):.3f}s"


@pytest.mark.asyncio
async def test_loop_wakeup_watcher_runs_every_sessiondb_call_off_loop_thread(loop_env):
    """Full wakeup path (slash-command loop, adapter present): list_active_loops,
    fire_tick and complete_tick must each execute on an executor thread, never
    on the event-loop thread (#92413)."""
    runner = _make_runner()
    runner._running = True
    runner._running_agents = {}

    class _Adapter:
        handled = []

        async def handle_message(self, event):
            self.handled.append(event.text)

    runner.adapters = {Platform.DISCORD: _Adapter()}
    runner._build_process_event_source = lambda evt: SimpleNamespace(
        platform=Platform.DISCORD, chat_id=evt["chat_id"], chat_type=evt["chat_type"],
        thread_id=evt["thread_id"] or None, user_id=evt["user_id"], user_name=evt["user_name"],
    )
    runner._session_key_for_source = lambda source: "agent:main:discord:channel:chat-loop"

    # A slash-command loop that is due now.
    await GatewayRunner._handle_loop_command(runner, _make_event("/loop 5m /status"))
    state = loops.load_loop("sid-gateway-loop")
    state.next_due_at = time.time() - 1
    loops.save_loop("sid-gateway-loop", state)

    loop_thread = threading.current_thread()
    on_loop_calls = []
    seen_calls = []

    def _record(name):
        # Positive count guards against a future import hoist in run.py that
        # would bypass these wrappers and leave the off-loop assertion vacuous.
        seen_calls.append(name)
        if threading.current_thread() is loop_thread:
            on_loop_calls.append(name)

    real_list = loops.list_active_loops
    real_fire = loops.LoopManager.fire_tick
    real_complete = loops.LoopManager.complete_tick

    def _list_active_loops(*a, **k):
        _record("list_active_loops")
        return real_list(*a, **k)

    def _fire_tick(self):
        _record("fire_tick")
        return real_fire(self)

    def _complete_tick(self, last_response):
        _record("complete_tick")
        return real_complete(self, last_response)

    orig_sleep = asyncio.sleep
    calls = {"n": 0}

    async def _one_pass_sleep(delay):
        calls["n"] += 1
        if calls["n"] >= 2:
            runner._running = False
        return await orig_sleep(0)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(loops, "list_active_loops", _list_active_loops)
        mp.setattr(loops.LoopManager, "fire_tick", _fire_tick)
        mp.setattr(loops.LoopManager, "complete_tick", _complete_tick)
        mp.setattr(asyncio, "sleep", _one_pass_sleep)
        await GatewayRunner._loop_wakeup_watcher(runner, interval=0)

    assert _Adapter.handled == ["/status"], _Adapter.handled
    assert seen_calls == ["list_active_loops", "fire_tick", "complete_tick"], seen_calls
    assert on_loop_calls == [], f"SessionDB calls ran on the event-loop thread: {on_loop_calls}"
    # complete_tick ran (slash-command loops complete immediately).
    assert loops.load_loop("sid-gateway-loop").ticks_fired == 1
