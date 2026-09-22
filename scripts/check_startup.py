"""Real process checks: occupied ports, duplicate instance, health and owned cleanup."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch import process_identity, read_json, signal_identified


def occupied():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    return sock


def ready(url):
    try:
        with urlopen(url, timeout=1) as response:
            return response.status == 200
    except (URLError, TimeoutError):
        return False


def wait_for(predicate, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if result := predicate():
            return result
        time.sleep(0.2)
    raise AssertionError("等待检查条件超时")


def server_ready(runtime):
    info = read_json(runtime / "server.lock")
    if (
        isinstance(info, dict)
        and info.get("ready")
        and ready(info["api_url"] + "/api/v1/health")
        and ready(info["ui_url"] + "/_stcore/health")
    ):
        return info
    return None


def stopped(info):
    return not ready(info["api_url"] + "/api/v1/health") and not ready(info["ui_url"] + "/_stcore/health")


def extended_checks(runtime, env, checks):
    processes = []
    identities = {}
    with (runtime / "lifecycle.log").open("w") as log:

        def launch(*args):
            process = subprocess.Popen(
                ["/bin/bash", "scripts/start.sh", *args], cwd=ROOT, env=env, stdout=log, stderr=log
            )
            processes.append(process)
            return process

        def remember(expected_pid=None):
            def expected_ready():
                info = server_ready(runtime)
                return info if info and (expected_pid is None or info["pid"] == expected_pid) else None

            info = wait_for(expected_ready)
            for record in read_json(runtime / "children.json"):
                assert record["identity"], "process identity must be available for orphan recovery"
                identities[record["pid"]] = record["identity"]
            return info

        try:
            server = launch()
            info = remember(server.pid)
            server.send_signal(signal.SIGHUP)
            assert server.wait(timeout=40) == 0
            wait_for(lambda: stopped(info), seconds=5)
            checks["terminal_close_stops_both_services"] = True
            print("通过：关闭终端后两项服务退出", flush=True)

            server = launch()
            info = remember(server.pid)
            status = runtime / "desktop-status.json"
            relay = launch("--owner-pid", str(os.getpid()), "--status-file", str(status))
            wait_for(lambda: (read_json(status) or {}).get("state") == "ready")
            assert server_ready(runtime)["pid"] == server.pid
            relay.terminate()
            assert relay.wait(timeout=40) == 0
            assert server.wait(timeout=40) == 0
            wait_for(lambda: stopped(info), seconds=5)
            checks["desktop_adopts_and_closes_verified_instance"] = True
            print("通过：重复打开使用已有实例，退出后一起关闭", flush=True)

            server = launch()
            info = remember(server.pid)
            previous = read_json(runtime / "children.json")
            server.kill()  # Deliberately simulate a killed supervisor, using test data only.
            server.wait(timeout=10)
            assert ready(info["api_url"] + "/api/v1/health")
            replacement = launch()
            info = remember(replacement.pid)
            assert all(process_identity(r["pid"]) != r["identity"] for r in previous)
            replacement.terminate()
            assert replacement.wait(timeout=40) == 0
            wait_for(lambda: stopped(info), seconds=5)
            checks["killed_supervisor_orphans_recovered"] = True
            print("通过：启动器被强制结束后，重新打开能回收遗留服务", flush=True)

            server = launch()
            info = remember(server.pid)
            record = read_json(runtime / "children.json")[0]
            signal_identified(record["pid"], record["identity"], signal.SIGTERM)
            assert server.wait(timeout=40) == 1
            wait_for(lambda: stopped(info), seconds=5)
            checks["backend_failure_stops_frontend"] = True
            print("通过：后端异常退出时前端跟随退出", flush=True)

            # This small owner process stands in for the native application's
            # parent-child relationship. Its disappearance must stop the service.
            owner_code = (
                "import os,subprocess,time; "
                "subprocess.Popen(['/bin/bash','scripts/start.sh','--owner-pid',str(os.getpid())]); "
                "time.sleep(120)"
            )
            owner = subprocess.Popen(
                [sys.executable, "-c", owner_code], cwd=ROOT, env=env, stdout=log, stderr=log
            )
            processes.append(owner)
            info = remember()
            identities[info["pid"]] = process_identity(info["pid"])
            owner.kill()
            owner.wait(timeout=10)
            wait_for(lambda: stopped(info), seconds=40)
            wait_for(lambda: not (runtime / "server.lock").read_text(), seconds=5)
            checks["desktop_owner_disappearance_stops_services"] = True
            print("通过：App 被强制退出后，服务自动关闭", flush=True)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=40)
            for pid, identity in identities.items():
                if identity:
                    signal_identified(pid, identity, signal.SIGKILL)


def main():
    checks = {}
    with (
        tempfile.TemporaryDirectory(prefix="docqa-startup-") as tmp,
        occupied() as backend,
        occupied() as frontend,
    ):
        runtime = Path(tmp)
        env = dict(
            os.environ,
            DOCQA_DATA_DIR=str(runtime),
            DOCQA_API_KEY="",
            DOCQA_BACKEND_PORT=str(backend.getsockname()[1]),
            DOCQA_FRONTEND_PORT=str(frontend.getsockname()[1]),
            DOCQA_PARSER="pymupdf",
            DOCQA_ENABLE_OCR="false",
            LC_ALL="en_US.UTF-8",
            DOCQA_OPEN_BROWSER="false",
        )
        with (runtime / "server.log").open("w+") as log:
            server = subprocess.Popen(
                ["/bin/bash", "scripts/start.sh"], cwd=ROOT, env=env, stdout=log, stderr=log
            )
            info = None
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        log.seek(0)
                        raise AssertionError(log.read())
                    try:
                        info = json.loads((runtime / "server.lock").read_text())
                    except (FileNotFoundError, ValueError):
                        time.sleep(0.2)
                        continue
                    if ready(info["api_url"] + "/api/v1/health") and ready(
                        info["ui_url"] + "/_stcore/health"
                    ):
                        break
                    time.sleep(0.2)
                else:
                    raise AssertionError("服务启动超时")
                checks["health"] = True
                assert info["api_url"] != f"http://127.0.0.1:{backend.getsockname()[1]}"
                assert info["ui_url"] != f"http://127.0.0.1:{frontend.getsockname()[1]}"
                checks["occupied_ports_selected_alternatives"] = True
                duplicate = subprocess.run(
                    ["/bin/bash", "scripts/start.sh", "--check"],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                assert duplicate.returncode == 0 and "已有实例" in duplicate.stdout, duplicate
                assert server.poll() is None and ready(info["api_url"] + "/api/v1/health")
                checks["duplicate_keeps_existing_server"] = True
                for listener in (backend, frontend):
                    with socket.create_connection(listener.getsockname(), timeout=1):
                        accepted, _ = listener.accept()
                        accepted.close()
                checks["unrelated_listeners_preserved"] = True
            finally:
                if server.poll() is None:
                    server.send_signal(signal.SIGTERM)
                server.wait(timeout=40)
            assert server.returncode == 0
            assert info and not ready(info["api_url"] + "/api/v1/health")
            assert not ready(info["ui_url"] + "/_stcore/health")
            assert not (runtime / "server.lock").read_text()
            checks["owned_processes_stopped_and_lock_released"] = True
        normal = subprocess.run(
            ["/bin/bash", "scripts/start.sh", "--check"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert normal.returncode == 0 and "启动成功" in normal.stdout, normal.stdout + normal.stderr
        checks["restart_check_mode"] = True
        print("通过：基础启动、端口占用、重复启动及清理 6 项", flush=True)
        extended_checks(runtime, env, checks)
    report = ROOT / "data" / "startup-check" / "report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
