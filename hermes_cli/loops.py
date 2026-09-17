"""Recurring in-session wakeups — the /loop command (Claude Code parity).

A loop stops when the agent ends a wakeup reply with ``LOOP_COMPLETE`` on its own line, when
``--times N`` ticks have fired, when the ``--until`` judge rules the condition met, or when the
``loops.max_ticks`` backstop pauses it. State lives in SessionDB ``state_meta`` (same contract as
``hermes_cli/goals.py``); CLI, gateway, and TUI all drive it through :class:`LoopManager`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Floor for fixed intervals. Claude Code allows 30s; anything tighter is almost always an
# accident that burns tokens polling unchanged state. Config loops.min_interval_seconds (clamped ≥ 5).
DEFAULT_MIN_INTERVAL_SECONDS = 30
# Backstop tick budget so an unattended loop can't run forever. 0 = unlimited; config loops.max_ticks.
DEFAULT_MAX_TICKS = 100
# Self-paced mode: start at the floor, double while replies are unchanged, cap at the
# ceiling, snap back to the floor on any change.
DEFAULT_SELF_PACED_FLOOR_SECONDS = 60
DEFAULT_SELF_PACED_CEILING_SECONDS = 15 * 60

# Completion sentinel the wakeup prompt teaches the agent to emit.
LOOP_COMPLETE_MARKER = "LOOP_COMPLETE"


class LoopPersistenceError(RuntimeError):
    """A requested loop mutation could not be durably persisted."""


# Marker on its own line, tolerating surrounding whitespace / trailing punctuation.
_LOOP_COMPLETE_RE = re.compile(
    r"(?im)^\s*" + re.escape(LOOP_COMPLETE_MARKER) + r"\s*[.!]?\s*$"
)
# Interval token: 30s / 5m / 2h / 1h30m (compound units allowed, at least one).
_INTERVAL_TOKEN_RE = re.compile(
    r"^(?=\d)(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)


WAKEUP_PROMPT_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration.\n"
    "If the task is now complete, no longer applicable, or the thing you "
    "were watching has finished, say so and end your reply with "
    f"{LOOP_COMPLETE_MARKER} on its own line — that stops the loop."
)

WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "Stop condition: {until}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration, and "
    "show concrete evidence of the stop condition's status.\n"
    "If the stop condition is met, or the task is no longer applicable, say "
    f"so and end your reply with {LOOP_COMPLETE_MARKER} on its own line — "
    "that stops the loop."
)


def parse_interval_token(token: str) -> Optional[int]:
    """Total seconds for ``30s``/``5m``/``2h``/``1h30m``, else None.

    A bare number is NOT an interval (it collides with prompt text like ``/loop 3 things``).
    """
    m = _INTERVAL_TOKEN_RE.match(token.strip()) if token else None
    if not m:
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mnt * 60 + s
    return total if total > 0 else None


def parse_loop_args(text: str) -> Dict[str, Any]:
    """Parse ``/loop [interval] <prompt> [--times N] [--until ...]``.

    Returns ``{"interval_seconds": int|None, "prompt", "times", "until", "error"}``;
    ``interval_seconds`` None means self-paced, ``error`` is set for unusable input.
    """
    raw = (text or "").strip()
    result: Dict[str, Any] = {"interval_seconds": None, "prompt": "", "times": 0, "until": "", "error": None}
    if not raw:
        return {**result, "error": "empty"}

    # Pull trailing flags first so an interval-looking token inside the --until clause can't
    # confuse the front parse. --until consumes to end-of-line (or to a following --times).
    times, until = 0, ""
    m_times = re.search(r"\s--times\s+(\S+)", raw)
    if m_times:
        try:
            times = int(m_times.group(1))
            if times < 1:
                raise ValueError
        except ValueError:
            return {**result, "error": f"--times expects a positive integer, got {m_times.group(1)!r}"}
        raw = (raw[: m_times.start()] + raw[m_times.end():]).strip()

    m_until = re.search(r"\s--until\s+(.+)$", raw, re.DOTALL)
    if m_until:
        until = m_until.group(1).strip()
        raw = raw[: m_until.start()].strip()

    # Leading "every" sugar: /loop every 5m <prompt>
    tokens = raw.split(None, 1)
    if tokens and tokens[0].lower() == "every" and len(tokens) > 1:
        raw = tokens[1]
        tokens = raw.split(None, 1)

    interval = parse_interval_token(tokens[0]) if tokens else None
    if interval is not None:
        raw = tokens[1].strip() if len(tokens) > 1 else ""

    if not raw:
        return {**result, "error": "missing prompt (usage: /loop [interval] <prompt>)"}
    return {**result, "interval_seconds": interval, "prompt": raw, "times": times, "until": until}


def format_interval(seconds: float) -> str:
    """Render seconds as a compact human interval (``90`` → ``1m30s``)."""
    h, rem = divmod(int(max(0, round(seconds))), 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h}h"] if h else []
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


def _loops_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("loops") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _config_int(key: str, default: int, floor: int) -> int:
    """``loops.<key>`` as an int clamped to ``floor``; ``default`` on any bad value."""
    try:
        return max(floor, int(_loops_config().get(key, default)))
    except Exception:
        return default


def min_interval_seconds() -> int:
    return _config_int("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS, 5)


def max_ticks_default() -> int:
    return _config_int("max_ticks", DEFAULT_MAX_TICKS, 0)


def self_paced_floor_seconds() -> int:
    return _config_int("self_paced_floor_seconds", DEFAULT_SELF_PACED_FLOOR_SECONDS, 10)


def self_paced_ceiling_seconds() -> int:
    floor = self_paced_floor_seconds()
    return _config_int("self_paced_ceiling_seconds", max(floor, DEFAULT_SELF_PACED_CEILING_SECONDS), floor)


@dataclass
class LoopState:
    """Serializable /loop state stored per session."""

    prompt: str
    status: str = "active"            # active | paused | done | cleared
    mode: str = "interval"            # interval | self_paced
    interval_seconds: float = 0.0     # fixed cadence (mode == "interval")
    current_delay: float = 0.0        # live cadence (self-paced backoff)
    times: int = 0                    # user cap (--times N); 0 = none
    until: str = ""                   # judged stop condition; "" = none
    max_ticks: int = DEFAULT_MAX_TICKS  # config backstop; 0 = unlimited
    ticks_fired: int = 0
    created_at: float = 0.0
    last_fired_at: float = 0.0
    next_due_at: float = 0.0
    # True between "wakeup injected" and "that turn's response evaluated": stops a tick from
    # double-firing mid-turn and tells the post-turn hook the turn that just ended was ours.
    awaiting_response: bool = False
    last_response_digest: str = ""    # self-paced change detection
    paused_reason: Optional[str] = None
    last_stop_reason: Optional[str] = None
    # Gateway routing (platform / chat_id / chat_type / thread_id) captured at creation so the
    # idle watcher can inject ticks into the right chat. Empty for CLI/TUI (own schedulers).
    route: Dict[str, str] = field(default_factory=dict)
    # Monotonic persisted revision rejects writes from managers that loaded
    # before another process changed or migrated this loop.
    generation: int = 0
    # A migrated predecessor is terminal: its logical loop belongs to this
    # successor and must never be resurrected by a stale manager.
    migrated_to: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        data = json.loads(raw)
        route = data.get("route")
        kwargs: Dict[str, Any] = {
            "prompt": data.get("prompt", ""),
            "status": data.get("status", "active"),
            "mode": data.get("mode", "interval"),
            "paused_reason": data.get("paused_reason"),
            "last_stop_reason": data.get("last_stop_reason"),
            "route": route if isinstance(route, dict) else {},
        }
        # Remaining scalar fields: missing key -> dataclass default; present-but-falsy -> type zero.
        casts = {"str": str, "int": int, "float": float, "bool": bool}
        for f in fields(cls):
            if f.name not in kwargs:
                kwargs[f.name] = casts[f.type](data.get(f.name, f.default) or casts[f.type]())
        return cls(**kwargs)

    def cadence_label(self) -> str:
        if self.mode == "self_paced":
            live = f", currently {format_interval(self.current_delay)}" if self.current_delay else ""
            return f"self-paced{live}"
        return f"every {format_interval(self.interval_seconds)}"

    def remaining_label(self) -> str:
        if self.status != "active":
            return ""
        remaining = self.next_due_at - time.time()
        return "due now" if remaining <= 0 else f"next in {format_interval(remaining)}"


_META_PREFIX = "loop:"
_MIGRATION_CAS_ATTEMPTS = 3
_MIGRATION_CHAIN_MAX_HOPS = 16


def _meta_key(session_id: str) -> str:
    return f"{_META_PREFIX}{session_id}"


def _get_session_db() -> Optional[Any]:
    """The goals module's cached SessionDB, so goals/loops/heartbeats share one connection and
    its off-loop bootstrap (a cold cache on the loop thread never runs ``SessionDB()`` inline).

    The previous copy here did, which froze the loop for the init duration and dropped the first ``loop:*``
    write (the /goal bug class, #88965).
    """
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB bootstrap failed (%s)", exc)
        return None
    return _goals_db()


def _db_op(label: str, fn, default=None):
    """Run one SessionDB call; any error is logged at debug and yields ``default``."""
    try:
        return fn()
    except Exception as exc:
        logger.debug("LoopManager: %s failed: %s", label, exc)
        return default


def _parse_state(raw: str, session_id: str = "") -> Optional[LoopState]:
    """``LoopState`` from stored JSON; None (warning when *session_id* given) on corrupt data."""
    try:
        return LoopState.from_json(raw)
    except Exception as exc:
        if session_id:
            logger.warning("LoopManager: could not parse stored loop for %s: %s", session_id, exc)
        return None


def load_loop(session_id: str) -> Optional[LoopState]:
    """Load the loop for a session, or None if none exists."""
    db = _get_session_db() if session_id else None
    if db is None:
        return None
    raw = _db_op("get_meta", lambda: db.get_meta(_meta_key(session_id)))
    return _parse_state(raw, session_id) if raw else None


def save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB when its loaded generation is current."""
    _save_loop_if_current(session_id, state)


def _state_json_at_generation(state: LoopState, generation: int) -> str:
    data = asdict(state)
    data["generation"] = generation
    return json.dumps(data, ensure_ascii=False)


def _save_loop_if_current(session_id: str, state: LoopState) -> Optional[bool]:
    """CAS-persist state; True on commit, False on conflict, None on I/O failure."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("LoopManager", "loop", session_id)
        return None
    key = _meta_key(session_id)
    try:
        current_raw = db.get_meta(key)
        if current_raw is None:
            if state.generation != 0:
                return False
        else:
            current = LoopState.from_json(current_raw)
            if current.generation != state.generation or current.migrated_to:
                return False

        next_generation = state.generation + 1
        next_raw = _state_json_at_generation(state, next_generation)
        saved = db.compare_and_set_meta_batch({key: current_raw}, {key: next_raw})
        if saved:
            state.generation = next_generation
            return True
        logger.debug(
            "LoopManager: stale loop write rejected for %s at generation %s",
            session_id, state.generation,
        )
        return False
    except Exception as exc:
        logger.debug("LoopManager: set_meta failed: %s", exc)
        return None


def clear_loop(session_id: str) -> None:
    """Mark a loop cleared in the DB (preserved for audit, status=cleared)."""
    state = load_loop(session_id)
    if state is not None:
        state.status = "cleared"
        save_loop(session_id, state)


def list_active_loops() -> List[Tuple[str, LoopState]]:
    """``[(session_id, LoopState), ...]`` for every ACTIVE loop; ``[]`` on any DB error.

    Used by the gateway's idle wakeup watcher, which scans for due loops on a coarse tick.
    """
    db = _get_session_db()
    if db is None:
        return []
    out: List[Tuple[str, LoopState]] = []
    for key, raw in _db_op("list_meta_prefix", lambda: db.list_meta_prefix(_META_PREFIX), []):
        session_id = key[len(_META_PREFIX):]
        state = _parse_state(raw) if session_id and raw else None
        if state is not None and state.status == "active":
            out.append((session_id, state))
    return out


def migrate_loop_to_session(
    old_session_id: str,
    new_session_id: str,
    *,
    reason: str = "",
    db: Optional[Any] = None,
    follow_migrated: bool = False,
) -> bool:
    """Atomically carry a loop from a parent session to its continuation.

    An existing destination wins unchanged. CAS conflicts retry a bounded
    number of times. Gateway-owned rotations may follow durable migration
    tombstones; compression must leave ``follow_migrated`` false so a stale
    compressor cannot pull a loop back from a newer gateway successor.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    session_db = db if db is not None else _get_session_db()
    if session_db is None:
        return False
    try:
        new_key = _meta_key(new_session_id)
        source_session_id = old_session_id
        visited: set[str] = set()
        followed_hops = 0

        while True:
            if source_session_id == new_session_id:
                target_raw = session_db.get_meta(new_key)
                if target_raw is None:
                    return False
                return LoopState.from_json(target_raw).status != "cleared"
            if source_session_id in visited:
                logger.debug(
                    "LoopManager: loop migration forwarding cycle at %s", source_session_id,
                )
                return False
            visited.add(source_session_id)

            source_key = _meta_key(source_session_id)
            forwarded_to = None
            for attempt in range(1, _MIGRATION_CAS_ATTEMPTS + 1):
                source_raw = session_db.get_meta(source_key)
                new_raw = session_db.get_meta(new_key)
                if source_raw is None:
                    return False
                state = LoopState.from_json(source_raw)
                if state.status == "cleared":
                    if not follow_migrated or not state.migrated_to:
                        return False
                    forwarded_to = state.migrated_to
                    break

                next_generation = state.generation + 1
                predecessor_data = asdict(state)
                predecessor_data.update(
                    status="cleared", generation=next_generation, migrated_to=new_session_id,
                )
                predecessor_raw = json.dumps(predecessor_data, ensure_ascii=False)
                updates = {source_key: predecessor_raw}
                if new_raw is None:
                    successor_data = asdict(state)
                    successor_data.update(generation=next_generation, migrated_to="")
                    # Successor first makes transaction rollback observable if
                    # the predecessor update fails after a valid insert.
                    updates = {
                        new_key: json.dumps(successor_data, ensure_ascii=False),
                        source_key: predecessor_raw,
                    }

                migrated = session_db.compare_and_set_meta_batch(
                    {source_key: source_raw, new_key: new_raw}, updates,
                )
                if migrated:
                    logger.debug(
                        "LoopManager: migrated loop %s -> %s (%s%s%s)",
                        source_session_id,
                        new_session_id,
                        reason or "rotation",
                        ", target preserved" if new_raw is not None else "",
                        f", followed from {old_session_id}"
                        if source_session_id != old_session_id else "",
                    )
                    return True
                logger.debug(
                    "LoopManager: loop migration CAS conflict on attempt %s/%s",
                    attempt, _MIGRATION_CAS_ATTEMPTS,
                )

            if forwarded_to is None:
                # A final failed CAS may have lost specifically to another
                # migration. Re-read once so the gateway owner can follow it.
                source_raw = session_db.get_meta(source_key)
                if source_raw is None:
                    return False
                state = LoopState.from_json(source_raw)
                if (
                    not follow_migrated
                    or state.status != "cleared"
                    or not state.migrated_to
                ):
                    return False
                forwarded_to = state.migrated_to
            if followed_hops >= _MIGRATION_CHAIN_MAX_HOPS:
                logger.debug(
                    "LoopManager: loop migration forwarding exceeded %s hops",
                    _MIGRATION_CHAIN_MAX_HOPS,
                )
                return False
            followed_hops += 1
            source_session_id = forwarded_to
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LoopManager: loop migration failed: %s", exc)
        return False


def _ticks_label(n: int) -> str:
    return f"{n} tick{'s' if n != 1 else ''}"


def _dash(reason: Optional[str]) -> str:
    return f" — {reason}" if reason else ""


def response_signals_complete(response: str) -> bool:
    """True when the agent ended its reply with the LOOP_COMPLETE marker."""
    return bool(response) and _LOOP_COMPLETE_RE.search(response) is not None


def _digest_response(response: str) -> str:
    """Digest for self-paced change detection; whitespace-normalized with clock/timestamp/duration
    tokens stripped so 'checked at 14:02:33' doesn't defeat the backoff."""
    text = (response or "").strip().lower()
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


class LoopManager:
    """Per-session /loop state + tick decisions.

    Drivers call ``set``/``pause``/``resume``/``clear`` for user controls, ``is_due()`` (cheap,
    in-memory), ``fire_tick()`` to claim a tick and get the wakeup message, ``complete_tick()`` to
    evaluate the finished turn, and ``status_line()``.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[LoopState] = load_loop(session_id)

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def refresh(self) -> None:
        """Re-read state from the DB (cross-process safety for the gateway)."""
        self._state = load_loop(self.session_id)

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_loop(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def _persist(self) -> bool:
        if self._state is None:
            return False
        saved = _save_loop_if_current(self.session_id, self._state)
        if saved is not True:
            # The persisted generation is authoritative. Migration tombstones
            # in particular must replace stale active in-memory state.
            self.refresh()
            return False
        return True

    def _persistence_failure_decision(self) -> Dict[str, Any]:
        status = self._state.status if self._state is not None else None
        return {
            "status": status,
            "stopped": status in {"done", "paused", "cleared"},
            "reason": "loop state changed or persistence failed before the update was durable",
            "message": "",
            "persisted": False,
        }

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No loop set. Start one with /loop [interval] <prompt>."
        fired = _ticks_label(s.ticks_fired)
        if s.times:
            caps = [f"{s.ticks_fired}/{s.times} runs"]
        elif s.max_ticks:
            caps = [f"{s.ticks_fired}/{s.max_ticks} budget"]
        else:
            caps = [fired]
        if s.until:
            caps.append(f"until: {s.until}")
        meta = f"{s.cadence_label()}, {', '.join(caps)}"
        if s.status == "active":
            remaining = s.remaining_label()
            tail = ", wakeup running" if s.awaiting_response else (f", {remaining}" if remaining else "")
            return f"↻ Loop (active, {meta}{tail}): {s.prompt}"
        if s.status == "paused":
            return f"⏸ Loop (paused, {meta}{_dash(s.paused_reason)}): {s.prompt}"
        if s.status == "done":
            return f"✓ Loop finished ({fired}{_dash(s.last_stop_reason)}): {s.prompt}"
        return f"Loop ({s.status}, {meta}): {s.prompt}"

    def set(
        self,
        prompt: str,
        *,
        interval_seconds: Optional[int] = None,
        times: int = 0,
        until: str = "",
        route: Optional[Dict[str, str]] = None,
    ) -> LoopState:
        """Start a new loop (replaces any existing one for the session)."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")

        prior_state = self._state
        if prior_state is None:
            # clear() drops the local state. Reload the audit row so a new
            # loop advances its current generation rather than bypassing CAS.
            prior_state = load_loop(self.session_id)
            self._state = prior_state
        if prior_state is not None and prior_state.migrated_to:
            raise LoopPersistenceError(
                f"could not persist loop: session migrated to {prior_state.migrated_to}"
            )

        now = time.time()
        self_paced = interval_seconds is None
        interval = 0.0 if self_paced else float(max(int(interval_seconds), min_interval_seconds()))
        state = LoopState(
            prompt=prompt,
            mode="self_paced" if self_paced else "interval",
            interval_seconds=interval,
            current_delay=float(self_paced_floor_seconds()) if self_paced else interval,
            times=max(0, int(times or 0)),
            until=(until or "").strip(),
            max_ticks=max_ticks_default(),
            created_at=now,
            next_due_at=now,
            route=dict(route or {}),
        )
        if prior_state is not None:
            state.generation = prior_state.generation
        self._state = state
        if not self._persist():
            if self._state is not None and self._state.migrated_to:
                raise LoopPersistenceError(
                    f"could not persist loop: session migrated to {self._state.migrated_to}"
                )
            raise LoopPersistenceError("could not persist loop state")
        return state

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        s = self._state
        if not s or s.status not in {"active", "paused"}:
            return None
        s.status, s.paused_reason, s.awaiting_response = "paused", reason, False
        return s if self._persist() else None

    def resume(self) -> Optional[LoopState]:
        s = self._state
        if not s or s.status == "cleared":
            return None
        s.status, s.paused_reason, s.awaiting_response = "active", None, False
        # Re-arm relative to now so a long pause doesn't fire instantly N times.
        delay = s.current_delay or s.interval_seconds or self_paced_floor_seconds()
        s.next_due_at = time.time() + min(delay, 5.0)
        return s if self._persist() else None

    def clear(self) -> bool:
        if self._state is None or self._state.status == "cleared":
            return False
        self._state.status = "cleared"
        if not self._persist():
            return False
        self._state = None
        return True

    def mark_done(self, reason: str) -> bool:
        if not self._state:
            return False
        self._state.status = "done"
        self._state.last_stop_reason = reason
        self._state.awaiting_response = False
        return self._persist()

    def is_due(self, now: Optional[float] = None) -> bool:
        """Cheap check: active, not mid-wakeup, and the clock has passed."""
        s = self._state
        return (
            s is not None and s.status == "active" and not s.awaiting_response
            and (now if now is not None else time.time()) >= s.next_due_at
        )

    def fire_tick(self) -> Optional[str]:
        """Claim a due tick; returns the message to inject, or None.

        The message is the wakeup-framed prompt, or the raw command when the loop's prompt is
        itself a slash command (``/loop 10m /recap``). Marks ``awaiting_response`` so the tick
        can't double-fire; drivers MUST follow up with ``complete_tick`` (or ``abandon_tick``).
        """
        s = self._state
        if s is None or not self.is_due():
            return None
        s.ticks_fired += 1
        s.last_fired_at = time.time()
        s.awaiting_response = True
        # Provisional schedule from NOW: complete_tick reschedules from turn end, but if the
        # process dies mid-turn this keeps the persisted loop from being 'due' in a tight loop.
        s.next_due_at = s.last_fired_at + (s.current_delay or s.interval_seconds or self_paced_floor_seconds())
        if not self._persist():
            return None

        if s.prompt.lstrip().startswith("/"):
            return s.prompt.strip()
        cadence = f", {s.cadence_label()}" if s.mode == "interval" else ", self-paced"
        template = WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE if s.until else WAKEUP_PROMPT_TEMPLATE
        return template.format(tick=s.ticks_fired, cadence=cadence, prompt=s.prompt, until=s.until)

    def abandon_tick(self) -> bool:
        """Roll back a fired tick whose injection failed (nothing ran)."""
        s = self._state
        if s is None or not s.awaiting_response:
            return False
        s.awaiting_response = False
        s.ticks_fired = max(0, s.ticks_fired - 1)
        return self._persist()

    def release_stale_tick_claim(self, now: Optional[float] = None) -> bool:
        """Release an overdue claim after its gateway turn disappeared.

        The caller must first prove there is no active runner or queued input.
        The tick count and cadence remain because the claimed tick did run;
        only its terminal bookkeeping was missed.
        """
        s = self._state
        check_at = now if now is not None else time.time()
        if (
            s is None
            or s.status != "active"
            or not s.awaiting_response
            or check_at < s.next_due_at
        ):
            return False
        s.awaiting_response = False
        if s.times and s.ticks_fired >= s.times:
            s.status = "done"
            s.last_stop_reason = f"completed the requested {s.times} runs"
        elif s.max_ticks and s.ticks_fired >= s.max_ticks:
            s.status = "paused"
            s.paused_reason = f"tick budget exhausted ({s.ticks_fired}/{s.max_ticks})"
        return self._persist()

    def _stop(self, status: str, reason: str, message: str) -> Dict[str, Any]:
        """Persist a terminal (``done``) or recoverable (``paused``) stop and build the result."""
        s = self._state
        s.status = status
        if status == "done":
            s.last_stop_reason = reason
        else:
            s.paused_reason = reason
        if not self._persist():
            return self._persistence_failure_decision()
        return {"status": status, "stopped": True, "reason": reason, "message": message}

    def complete_tick(self, last_response: str) -> Dict[str, Any]:
        """Evaluate the finished wakeup turn and schedule what's next.

        Returns ``{"status": "active|done|paused", "stopped": bool, "reason": str, "message": str}``;
        ``message`` is a user-visible one-liner, "" in the common still-looping case.
        """
        s = self._state
        if s is None or not s.awaiting_response:
            return {"status": s.status if s else None, "stopped": False, "reason": "no tick in flight", "message": ""}
        s.awaiting_response = False
        now = time.time()
        ticks = _ticks_label(s.ticks_fired)

        # 1. Agent self-stop marker.
        if response_signals_complete(last_response):
            return self._stop("done", "agent signaled the task is complete",
                              f"✓ Loop finished after {ticks} — task complete.")

        # 2. Evidence-based --until judge (reuses the /goal judge; fail-open).
        if s.until and (last_response or "").strip():
            try:
                from hermes_cli.goals import judge_goal

                verdict, reason, _pf, _wait, _tf = judge_goal(s.until, last_response)
            except Exception as exc:
                verdict, reason = "continue", f"judge unavailable: {type(exc).__name__}"
            if verdict == "done":
                return self._stop("done", f"stop condition met: {reason}",
                                  f"✓ Loop finished after {ticks} — {reason}")
            if verdict == "blocked":
                # Unachievable stop condition: pause so the user can re-scope, don't spin.
                why = f"stop condition judged unachievable: {reason}"
                return self._stop("paused", why,
                                  f"⏸ Loop paused — {why}. /loop resume to keep going, /loop stop to end it.")

        # 3. --times user cap.
        if s.times and s.ticks_fired >= s.times:
            return self._stop("done", f"completed the requested {s.times} runs",
                              f"✓ Loop finished — ran {s.times}/{s.times} times.")

        # 4. Config backstop budget → pause (recoverable), not done.
        if s.max_ticks and s.ticks_fired >= s.max_ticks:
            return self._stop(
                "paused", f"tick budget exhausted ({s.ticks_fired}/{s.max_ticks})",
                f"⏸ Loop paused — {s.ticks_fired}/{s.max_ticks} ticks used "
                "(loops.max_ticks). /loop resume to keep going, /loop stop to end it.",
            )

        # 5. Still looping — schedule the next tick from turn end.
        if s.mode == "self_paced":
            digest = _digest_response(last_response)
            floor = self_paced_floor_seconds()
            if digest and digest == s.last_response_digest:
                s.current_delay = min(max(s.current_delay, floor) * 2, self_paced_ceiling_seconds())
            else:
                s.current_delay = float(floor)
            s.last_response_digest = digest
        else:
            s.current_delay = s.interval_seconds
        s.next_due_at = now + s.current_delay
        if not self._persist():
            return self._persistence_failure_decision()
        return {"status": "active", "stopped": False, "reason": "loop continues", "message": ""}


def goal_blocks_loop_tick(session_id: str) -> bool:
    """True when an ACTIVE, non-parked /goal should defer this session's /loop tick.

    Both features inject synthetic turns at idle boundaries; interleaving them would burn the
    goal's turn budget. Parked (waiting), paused, or done goals do NOT block the loop.
    """
    try:
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=session_id)
        return mgr.is_active() and not mgr.is_waiting()
    except Exception:
        return False


LOOP_HELP = (
    "Usage: /loop [interval] <prompt> [--times N] [--until <condition>]\n"
    "  /loop 5m check the deploy status      — first run now, then every 5m\n"
    "  /loop every 10m /recap                — loop a slash command\n"
    "  /loop keep fixing tests until green   — self-paced (backs off while output is unchanged)\n"
    "  /loop 2m poll CI --times 30           — stop after 30 runs\n"
    "  /loop 5m watch the queue --until queue is empty\n"
    "Controls: /loop status · /loop pause · /loop resume · /loop stop\n"
    "The loop also stops itself when the agent replies with "
    f"{LOOP_COMPLETE_MARKER}."
)


def _pause_output(mgr: "LoopManager") -> str:
    state = mgr.pause(reason="user-paused")
    return "No loop set." if state is None else f"⏸ Loop paused: {state.prompt}\nUse /loop resume to continue."


def _resume_output(mgr: "LoopManager") -> str:
    state = mgr.resume()
    return "No loop to resume." if state is None else f"▶ Loop resumed ({state.cadence_label()}): {state.prompt}"


# Control words -> handler returning the output text. Anything else is a new loop spec.
_CONTROL_COMMANDS = {
    **dict.fromkeys(("", "status"), lambda mgr: mgr.status_line()),
    "pause": _pause_output,
    "resume": _resume_output,
    **dict.fromkeys(("stop", "clear", "cancel"), lambda mgr: "✓ Loop stopped." if mgr.clear() else "No active loop."),
    **dict.fromkeys(("help", "--help", "-h"), lambda mgr: LOOP_HELP),
}


def dispatch_loop_command(
    mgr: "LoopManager",
    args: str,
    *,
    route: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Surface-agnostic handler for ``/loop <args>`` → ``{"output": str, "created": bool}``.

    ``output`` is printed/sent verbatim by each surface. ``route`` is stored on new loops so the
    gateway's idle watcher can inject wakeups into the right chat; CLI/TUI pass None.
    """
    arg = (args or "").strip()
    control = _CONTROL_COMMANDS.get(arg.lower())
    if control is not None:
        return {"output": control(mgr), "created": False}

    parsed = parse_loop_args(arg)
    if parsed["error"]:
        if parsed["error"] == "empty":
            return {"output": "Usage: /loop [interval] <prompt> — see /loop help.", "created": False}
        return {"output": f"/loop: {parsed['error']}", "created": False}

    replacing = mgr.has_loop()
    try:
        state = mgr.set(
            parsed["prompt"],
            interval_seconds=parsed["interval_seconds"],
            times=parsed["times"],
            until=parsed["until"],
            route=route,
        )
    except (ValueError, LoopPersistenceError) as exc:
        return {"output": f"/loop: {exc}", "created": False}

    lines = [f"↻ Loop set ({state.cadence_label()}): {state.prompt}"]
    if replacing:
        lines.append("(replaced the previous loop for this session)")
    if parsed["interval_seconds"] is not None and parsed["interval_seconds"] < state.interval_seconds:
        lines.append(
            f"(interval raised to the {format_interval(state.interval_seconds)} minimum — "
            "loops.min_interval_seconds)"
        )
    if state.mode == "self_paced":
        lines.append(
            f"Self-paced: first check in {format_interval(state.current_delay)}; "
            f"backs off up to {format_interval(self_paced_ceiling_seconds())} while nothing changes."
        )
    if state.times:
        lines.append(f"Runs {state.times} time{'s' if state.times != 1 else ''}, then stops.")
    if state.until:
        lines.append(f"Stops when: {state.until}")
    if not state.times and state.max_ticks:
        lines.append(f"Backstop budget: {state.max_ticks} ticks (loops.max_ticks; 0 = unlimited).")
    first = "fires now, then on the cadence above" if state.status == "active" else state.remaining_label()
    lines.append(f"First wakeup {first}. Controls: /loop status · pause · resume · stop.")
    return {"output": "\n".join(lines), "created": True}


__all__ = [
    "LoopState", "LoopManager", "LoopPersistenceError", "parse_loop_args", "parse_interval_token", "format_interval",
    "response_signals_complete", "goal_blocks_loop_tick", "load_loop", "save_loop", "clear_loop",
    "list_active_loops", "migrate_loop_to_session", "dispatch_loop_command", "LOOP_COMPLETE_MARKER",
    "WAKEUP_PROMPT_TEMPLATE", "WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE", "DEFAULT_MIN_INTERVAL_SECONDS",
    "DEFAULT_MAX_TICKS",
]
