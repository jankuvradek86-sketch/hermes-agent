"""Gateway /loop command tests — dispatch, routing capture, mid-run guard."""

import asyncio
import logging
import time
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
