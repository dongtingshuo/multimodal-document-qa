"""Regression coverage for process ownership and desktop lifecycle boundaries."""

import os
import signal

import pytest

from scripts import launch


def test_recovery_ignores_reused_pid_and_other_directory(tmp_path, monkeypatch):
    launch.write_json(
        tmp_path / "children.json",
        [
            {"pid": 901, "identity": "old process"},
            {"pid": 902, "identity": "owned process"},
            {"pid": 903, "identity": "other directory"},
        ],
    )
    identities = {901: "reused process", 902: "owned process", 903: "other directory"}
    monkeypatch.setattr(launch, "process_identity", identities.get)
    monkeypatch.setattr(launch, "_process_cwd", lambda pid: str(launch.ROOT) if pid != 903 else "/elsewhere")
    monkeypatch.setattr(launch, "_command_pids", lambda args: set())
    assert launch.recovery_targets(tmp_path) == {902: "owned process"}
    # A different data directory cannot claim this instance's frontend.
    assert launch.recovery_targets(tmp_path / "another") == {}


def test_legacy_recovery_requires_backend_lock_and_command(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "_command_pids", lambda args: {901, 902})
    monkeypatch.setattr(launch, "_process_cwd", lambda pid: str(launch.ROOT))
    monkeypatch.setattr(
        launch,
        "process_identity",
        lambda pid: "date python -m uvicorn docqa.api:app --port 8200" if pid == 901 else "other tool",
    )
    assert set(launch.recovery_targets(tmp_path)) == {901}


def test_identity_rechecked_before_signal(monkeypatch):
    monkeypatch.setattr(launch, "process_identity", lambda pid: "new process")

    def forbidden(*args):
        pytest.fail("A recycled PID must never be signalled")

    monkeypatch.setattr(os, "kill", forbidden)
    launch.signal_identified(901, "old process", signal.SIGTERM)


def test_permission_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "recovery_targets", lambda path: {901: "owned"})
    monkeypatch.setattr(launch, "process_identity", lambda pid: "owned")

    def deny(*args):
        raise PermissionError

    monkeypatch.setattr(os, "kill", deny)
    with pytest.raises(RuntimeError, match="没有权限"):
        launch.cleanup_orphaned_services(tmp_path)


def test_owner_loss_interrupts_startup(monkeypatch):
    monkeypatch.setattr(os, "getppid", lambda: 1)
    with pytest.raises(KeyboardInterrupt):
        launch.check_owner(901)
    launch.check_owner(None)


def test_cleanup_continues_if_first_child_exits_early(monkeypatch):
    calls = []

    class Child:
        def __init__(self, pid):
            self.pid = pid

        def wait(self, timeout):
            calls.append(("wait", self.pid))

    def killpg(pid, sig):
        calls.append(("signal", pid))
        if pid == 901:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)
    launch.stop_children([Child(901), Child(902)])
    assert calls == [("signal", 901), ("signal", 902), ("wait", 901), ("wait", 902)]


def test_close_during_existing_startup_stops_verified_instance(tmp_path, monkeypatch):
    import json

    info = {"pid": 901, "instance_id": "starting-instance", "ready": False}
    path = tmp_path / "server.lock"
    path.write_text(json.dumps(info))
    identity = "date python scripts/launch.py"
    monkeypatch.setattr(launch, "process_identity", lambda pid: identity)
    monkeypatch.setattr(launch, "_process_cwd", lambda pid: str(launch.ROOT))

    def locked(*args):
        raise BlockingIOError

    monkeypatch.setattr(launch.fcntl, "flock", locked)
    monkeypatch.setattr(launch, "check_owner", lambda pid: None)

    def closing(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(launch.time, "sleep", closing)
    calls = []

    def stop(pid, expected, sig):
        calls.append((pid, expected, sig))
        monkeypatch.setattr(launch, "process_identity", lambda pid: None)

    monkeypatch.setattr(launch, "signal_identified", stop)
    with path.open("a+") as lock, pytest.raises(KeyboardInterrupt):
        launch.attach_existing(lock, 777, lambda *args, **kwargs: pytest.fail("Not ready yet"))
    assert calls == [(901, identity, signal.SIGTERM)]
