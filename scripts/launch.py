"""Single-instance local supervisor with port selection and readiness checks."""

import argparse
import errno
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docqa.config import Settings


def _command_pids(args):
    """Return numeric PIDs reported by a local process-inspection command."""
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    pids = set()
    for value in result.stdout.split():
        try:
            pids.add(int(value))
        except ValueError:
            continue
    return pids


def _process_cwd(pid):
    """Read a process working directory without relying on platform-specific APIs."""
    lsof = "/usr/sbin/lsof"
    try:
        result = subprocess.run(
            [lsof, "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def process_identity(pid):
    """Include start time and argv so a recycled PID never authorizes cleanup."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def recovery_targets(data_dir):
    saved = read_json(data_dir / "children.json")
    records = saved if isinstance(saved, list) else []
    targets = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        pid, identity = record.get("pid"), record.get("identity")
        if (
            type(pid) is int
            and pid > 1
            and pid != os.getpid()
            and identity
            and process_identity(pid) == identity
            and _process_cwd(pid) == str(ROOT)
        ):
            targets[pid] = identity
    # Migrate old backends only when they actually hold this data directory's
    # lock and run this application's backend. Never sweep a range of ports.
    for pid in _command_pids(["/usr/sbin/lsof", "-t", str(data_dir / "backend.lock")]):
        identity = process_identity(pid)
        if (
            pid > 1
            and pid != os.getpid()
            and identity
            and "-m uvicorn docqa.api:app" in identity
            and _process_cwd(pid) == str(ROOT)
        ):
            targets[pid] = identity
    return targets


def signal_identified(pid, identity, sig):
    if process_identity(pid) != identity:
        return
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
    except PermissionError as exc:
        raise RuntimeError("没有权限关闭上次遗留的项目服务，请退出旧的文档问答程序。") from exc


def cleanup_orphaned_services(data_dir):
    targets = recovery_targets(data_dir)
    if not targets:
        return
    print("正在关闭上次遗留的项目服务…", flush=True)
    for pid, identity in targets.items():
        signal_identified(pid, identity, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not any(process_identity(pid) == identity for pid, identity in targets.items()):
            break
        time.sleep(0.1)
    for pid, identity in targets.items():
        signal_identified(pid, identity, signal.SIGKILL)
    # Confirm listeners/locks are released before reserving new ports.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not any(process_identity(pid) == identity for pid, identity in targets.items()):
            return
        time.sleep(0.1)
    raise RuntimeError("旧服务尚未完全退出，请稍后重新启动。")


def check_owner(owner_pid):
    if owner_pid and os.getppid() != owner_pid:
        raise KeyboardInterrupt


def stop_children(children):
    # Ignore repeated close signals while releasing both process groups.
    previous = {
        sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def attach_existing(lock, owner_pid, publish):
    """A desktop relay owns no new servers, but can close the verified instance."""
    original = None
    identity = None
    announced = False
    try:
        deadline = time.monotonic() + 60
        while True:
            check_owner(owner_pid)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                raise RuntimeError("已有服务已退出，请点击重新启动。")
            lock.seek(0)
            try:
                current = json.loads(lock.read())
            except ValueError:
                current = {}
            if not isinstance(current, dict):
                current = {}
            pid = current.get("pid")
            if original is None and current.get("instance_id") and type(pid) is int and pid > 1:
                candidate = process_identity(pid)
                if candidate and "scripts/launch.py" in candidate and _process_cwd(pid) == str(ROOT):
                    original, identity = current, candidate
            if original is not None and current.get("instance_id") != original["instance_id"]:
                raise RuntimeError("服务实例已变更，请重新启动。")
            if original and current.get("ready") and not announced:
                publish("ready", "服务已就绪，关闭此窗口会停止服务。", **current)
                open_browser(current["ui_url"])
                announced = True
            if not announced and time.monotonic() > deadline:
                raise RuntimeError("已有实例尚未就绪或版本过旧，请退出旧启动器后重试。")
            time.sleep(0.3)
    finally:
        # Verify the instance token and the OS identity immediately before stop.
        lock.seek(0)
        try:
            latest = json.loads(lock.read())
        except ValueError:
            latest = {}
        if (
            original
            and identity
            and isinstance(latest, dict)
            and latest.get("instance_id") == original["instance_id"]
        ):
            signal_identified(original["pid"], identity, signal.SIGTERM)
            deadline = time.monotonic() + 20
            while process_identity(original["pid"]) == identity and time.monotonic() < deadline:
                time.sleep(0.1)


def open_browser(url):
    if os.getenv("DOCQA_OPEN_BROWSER", "").lower() in {"1", "true", "yes"}:
        try:
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            print("浏览器自动打开失败，请使用启动窗口中的打开页面按钮。", flush=True)


def reserve_port(start, excluded=()):
    if not 1024 <= start <= 65535:
        raise ValueError("端口须在 1024–65535 之间")
    for port in range(start, min(start + 100, 65536)):
        if port in excluded:
            continue
        sock = socket.socket()
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            return sock, port
        except OSError as exc:
            sock.close()
            if exc.errno != errno.EADDRINUSE:
                raise RuntimeError(f"无法绑定本地端口：{exc.strerror}。这不是端口占用。") from exc
    raise RuntimeError(f"从 {start} 起的端口均已占用")


def wait_ready(process, url, timeout=45, owner_pid=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        check_owner(owner_pid)
        if process.poll() is not None:
            raise RuntimeError(f"服务启动失败，退出码 {process.returncode}；请查看上方日志")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"等待服务就绪超时：{url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="启动并检查服务，随后停止本次创建的进程")
    parser.add_argument("--owner-pid", type=int, help="桌面 App 的父进程")
    parser.add_argument("--status-file", type=Path, help="桌面状态文件")
    args = parser.parse_args()

    def publish(state, message, **extra):
        if args.status_file:
            write_json(args.status_file, {**extra, "state": state, "message": message})

    publish("starting", "正在启动文档问答服务…")
    os.chdir(ROOT)
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = settings.data_dir / "server.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            info = lock.read().strip()
            print("同一数据目录已有实例运行：" + (info or "正在启动，请稍后查看终端"), flush=True)
            if args.owner_pid:
                try:
                    attach_existing(lock, args.owner_pid, publish)
                except KeyboardInterrupt:
                    publish("stopped", "服务已关闭")
                    return 0
                except (OSError, RuntimeError) as exc:
                    publish("error", str(exc))
                    return 1
            else:
                try:
                    existing = json.loads(info)
                    if isinstance(existing, dict) and existing.get("ready"):
                        open_browser(existing["ui_url"])
                except (ValueError, KeyError):
                    pass
            return 0
        backend_socket = frontend_socket = None
        children = []
        try:
            # A previous Terminal/app may have been closed before its child
            # processes received the supervisor cleanup signal. Once this
            # supervisor lock is acquired, no managed instance is active, so
            # reclaim only recorded children or this data directory's backend.
            check_owner(args.owner_pid)
            cleanup_orphaned_services(settings.data_dir)
            requested_backend = int(os.getenv("DOCQA_BACKEND_PORT", "8000"))
            requested_frontend = int(os.getenv("DOCQA_FRONTEND_PORT", "8501"))
            backend_socket, backend_port = reserve_port(requested_backend)
            frontend_socket, frontend_port = reserve_port(requested_frontend, {backend_port})
            for label, requested, actual in [
                ("后端", requested_backend, backend_port),
                ("前端", requested_frontend, frontend_port),
            ]:
                if actual != requested:
                    print(f"{label}端口 {requested} 已占用，改用 {actual}。", flush=True)
            api_url = f"http://127.0.0.1:{backend_port}"
            ui_url = f"http://127.0.0.1:{frontend_port}"
            env = dict(os.environ, DOCQA_API_URL=api_url, DOCQA_DATA_DIR=str(settings.data_dir))
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            lock.seek(0)
            lock.truncate()
            instance = {
                "pid": os.getpid(),
                "instance_id": uuid.uuid4().hex,
                "api_url": api_url,
                "ui_url": ui_url,
                "ready": False,
            }
            lock.write(json.dumps(instance))
            lock.flush()
            backend_socket.close()
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "docqa.api:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(backend_port),
                    ],
                    env=env,
                    start_new_session=True,
                )
            )
            records = [{"pid": child.pid, "identity": process_identity(child.pid)} for child in children]
            write_json(settings.data_dir / "children.json", records)
            wait_ready(children[-1], api_url + "/api/v1/health", owner_pid=args.owner_pid)
            frontend_socket.close()
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "ui/app.py",
                        "--server.address",
                        "127.0.0.1",
                        "--server.port",
                        str(frontend_port),
                        "--server.headless",
                        "true",
                        "--browser.gatherUsageStats",
                        "false",
                    ],
                    env=env,
                    start_new_session=True,
                )
            )
            records = [{"pid": child.pid, "identity": process_identity(child.pid)} for child in children]
            write_json(settings.data_dir / "children.json", records)
            wait_ready(children[-1], ui_url + "/_stcore/health", owner_pid=args.owner_pid)
            instance["ready"] = True
            lock.seek(0)
            lock.truncate()
            lock.write(json.dumps(instance))
            lock.flush()
            publish("ready", "服务已就绪，关闭此窗口会停止服务。", **instance)
            print(f"\n启动成功\n前端：{ui_url}\n后端：{api_url}\n按 Ctrl+C 停止本次启动的服务。", flush=True)
            if not args.check:
                open_browser(ui_url)
            if args.check:
                return 0
            while all(child.poll() is None for child in children):
                check_owner(args.owner_pid)
                time.sleep(0.3)
            raise RuntimeError("一个服务已经退出，正在停止另一个服务")
        except KeyboardInterrupt:
            publish("stopped", "服务已关闭")
            return 0
        except (ValueError, OSError, RuntimeError) as exc:
            publish("error", str(exc))
            print(f"启动失败：{exc}", file=sys.stderr)
            return 1
        finally:
            for sock in (backend_socket, frontend_socket):
                if sock is not None:
                    sock.close()
            stop_children(children)
            if children:
                (settings.data_dir / "children.json").unlink(missing_ok=True)
            lock.seek(0)
            lock.truncate()


if __name__ == "__main__":

    def stop_on_signal(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGHUP, stop_on_signal)
    raise SystemExit(main())
