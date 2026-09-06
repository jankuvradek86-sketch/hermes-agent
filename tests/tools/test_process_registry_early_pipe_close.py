"""Pipe lifetime must not replace the tracked local process lifetime."""

import subprocess
import sys
import threading

import pytest

from tools.process_registry import ProcessRegistry, ProcessSession


@pytest.mark.parametrize("read_error", [False, True], ids=["eof", "read-error"])
def test_reader_waits_for_real_exit_before_notifying(tmp_path, monkeypatch, read_error):
    release = tmp_path / "release"
    child_code = (
        "import pathlib, sys, time; deadline = time.monotonic() + 20; "
        "release = pathlib.Path(sys.argv[1])\n"
        "while not release.exists() and time.monotonic() < deadline: time.sleep(0.05)\n"
        "sys.exit(7)"
    )
    wrapper_code = (
        "import os, subprocess, sys; os.close(1); os.close(2); "
        "child = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); sys.exit(child.wait())"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", wrapper_code, child_code, str(release)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    stdout = proc.stdout
    if read_error:
        class BrokenReader:
            def read(self, size):
                raise OSError("synthetic pipe read failure")
        monkeypatch.setattr(proc, "stdout", BrokenReader())
    waiting = threading.Event()
    real_wait = proc.wait

    def observed_wait(*args, **kwargs):
        waiting.set()
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(proc, "wait", observed_wait)
    registry = ProcessRegistry()
    session = ProcessSession(id="early_pipe", command="bounded wrapper", task_id="test")
    session.process, session.pid = proc, proc.pid
    session.notify_on_complete = True
    registry._running[session.id] = session
    reader = threading.Thread(target=registry._reader_loop, args=(session,), daemon=True)
    reader.start()
    try:
        assert waiting.wait(5), "reader never reached process wait"
        # Cross the old five-second reap timeout while the child awaits release.
        reader.join(timeout=6.5)
        assert proc.poll() is None
        assert not session.exited, "pipe termination prematurely finalized a live wrapper"
        assert registry.completion_queue.empty()
        assert session.id in registry._running
        release.touch()
        reader.join(timeout=5)
        assert not reader.is_alive()
        assert session.exited and session.exit_code == 7
        assert session.id in registry._finished
        event = registry.completion_queue.get_nowait()
        assert event["session_id"] == session.id and event["exit_code"] == 7
        assert registry.completion_queue.empty()
    finally:
        release.touch()
        real_wait(timeout=25)
        reader.join(timeout=5)
        stdout.close()
