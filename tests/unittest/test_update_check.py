"""验证后台更新检查的版本语义、弱网退出、缓存及命令行边界。"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from filelock import FileLock
from loguru import logger
from typer.testing import CliRunner

from mineru.config import Config
from mineru.utils import update_check as updates

config_module = importlib.import_module("mineru.config")
FETCH_INDEX = updates._fetch_index


@pytest.fixture(autouse=True)
def isolated_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """隔离配置与用户缓存，所有测试默认禁止真实外网请求。"""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    monkeypatch.setenv("MINERU_LANG", "en")
    monkeypatch.delenv(updates._OWNER_ENV, raising=False)
    settings = Config(update_check={"enabled": True})
    monkeypatch.setattr(config_module, "get_config", lambda: settings)
    from mineru.utils.i18n import reset_language_cache

    reset_language_cache()
    monkeypatch.setattr(updates, "_fetch_index", lambda *args: None)
    yield
    reset_language_cache()


def _file(version: str, *, yanked: bool | str = False, python: str = ">=3.11") -> dict[str, object]:
    """构造标准 wheel 文件记录，用于验证索引筛选语义。"""
    return {"filename": f"mineru-{version}-py3-none-any.whl", "yanked": yanked, "requires-python": python}


def _snapshot(version: str = "99.0", source: updates.Source = "pypi") -> updates.Snapshot:
    """创建确定的新版本快照，不依赖仓库当前版本号。"""
    return updates.Snapshot(source=source, release=updates.Release(version=version))


def _cache_path(tmp_path: Path) -> Path:
    """返回测试缓存路径，并准备后台写入所需父目录。"""
    path = tmp_path / "cache" / "update-check.json"
    path.parent.mkdir(exist_ok=True)
    return path


def test_index_selects_stable_post_release_and_excludes_yanked() -> None:
    """正式 post 版参与排序，预发布、开发版与空理由撤回文件都不参与。"""
    files = [_file(version) for version in ("4.9", "4.10", "4.10.post1", "5.0rc1", "5.0.dev1", "5.0+local")]
    files += [_file("6.0", yanked=True), _file("7.0", yanked=""), _file("8.0", yanked="broken")]
    result = updates._parse_index({"name": "mineru", "files": files}, "pypi")
    assert result.release is not None
    assert result.release.version == "4.10.post1"


def test_index_accepts_sdist_and_retains_python_requirements() -> None:
    """同版文件的 Python 约束共同保留，源码包也能识别。"""
    files = [_file("5.0", python=">=3.14"), {"filename": "mineru-5.0.tar.gz", "requires-python": ">=3.13"}]
    result = updates._parse_index({"name": "mineru", "files": files}, "tuna")
    assert result.release == updates.Release(version="5.0", python_requires=(">=3.13", ">=3.14"))


@pytest.mark.parametrize("files", [[], [_file("5.0", yanked=True)], [_file("5.0rc1")]])
def test_valid_index_without_stable_files_is_success(files: list[dict[str, object]]) -> None:
    """全部撤回或仅有预发布版本是有效快照，不应误判为官方请求失败。"""
    assert updates._parse_index({"name": "mineru", "files": files}, "pypi").release is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"name": "other", "files": []},
        {"name": "mineru", "files": [{}]},
        {"name": "mineru", "files": [{"filename": "other-9.0.tar.gz"}]},
        {"name": "mineru", "files": [_file("5.0", python="not-a-specifier")]},
    ],
)
def test_invalid_index_is_rejected(payload: object) -> None:
    """项目、文件或约束异常时丢弃该来源，允许另一来源接替。"""
    with pytest.raises(ValueError):
        updates._parse_index(payload, "pypi")


@pytest.mark.parametrize("official", [_snapshot("100.0"), _snapshot("98.0"), updates.Snapshot(source="pypi")])
def test_official_snapshot_wins_even_when_mirror_returns_first(
    official: updates.Snapshot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """镜像先返回不能抢占官方结果，包括官方较低版本和全部撤回情形。"""
    mirror_finished = threading.Event()

    def fetch(source: updates.Source, deadline: float, stopped: threading.Event) -> updates.Snapshot:
        """用事件确定双源返回顺序，避免依赖机器调度速度。"""
        if source == "tuna":
            mirror_finished.set()
            return _snapshot("99.0", "tuna")
        assert mirror_finished.wait(1)
        return official

    monkeypatch.setattr(updates, "_fetch_index", fetch)
    assert updates._query_sources("auto", threading.Event()) == official


def test_stalled_official_uses_mirror_and_drops_late_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """DNS 等不可及时中断的请求不会延长本轮结果窗口。"""
    release = threading.Event()
    finished = threading.Event()
    monkeypatch.setattr(updates, "CHECK_DEADLINE", 0.05)

    def fetch(source: updates.Source, deadline: float, stopped: threading.Event) -> updates.Snapshot:
        """官方请求停在事件上，镜像立即返回有效快照。"""
        if source == "pypi":
            release.wait(2)
            finished.set()
            return _snapshot("100.0")
        return _snapshot("99.0", "tuna")

    monkeypatch.setattr(updates, "_fetch_index", fetch)
    try:
        start = time.monotonic()
        assert updates._query_sources("auto", threading.Event()) == _snapshot("99.0", "tuna")
        assert time.monotonic() - start < 1
    finally:
        release.set()
        assert finished.wait(1)


@pytest.mark.parametrize("mode, expected", [("auto", {"pypi", "tuna"}), ("pypi", {"pypi"}), ("tuna", {"tuna"})])
def test_failed_sources_are_silent_and_respect_selection(
    mode: updates.SourceMode, expected: set[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """指定来源只访问该来源，全部失败返回未知而非已经最新。"""
    calls: set[str] = set()

    def fetch(source: updates.Source, deadline: float, stopped: threading.Event) -> updates.Snapshot:
        """模拟连接失败，核对实际发出的来源集合。"""
        calls.add(source)
        raise OSError("offline")

    monkeypatch.setattr(updates, "_fetch_index", fetch)
    assert updates._query_sources(mode, threading.Event()) is None
    assert calls == expected


def test_fresh_cache_avoids_network_and_deduplicates_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """新版本每天只提示一次，并且缓存有效期间不发网络请求。"""
    path = _cache_path(tmp_path)
    now = time.time()
    cache = updates.Cache(states={"auto": updates.CheckState(snapshot=_snapshot(), checked_at=now, next_check_at=now + 86400)})
    updates._write_cache(path, cache)
    calls: list[str] = []
    monkeypatch.setattr(updates, "_query_sources", lambda *args: calls.append("network"))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING", format="{message}")
    try:
        updates._check("auto", threading.Event())
        updates._check("auto", threading.Event())
    finally:
        logger.remove(sink)
    assert calls == []
    assert len(messages) == 1
    assert "99.0" in messages[0]
    assert len(updates._read_cache(path).notices) == 1


def test_failed_refresh_preserves_success_and_backs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """刷新失败保留过期快照但不提示，后续失败按 1、2、4、6 小时退避。"""
    path = _cache_path(tmp_path)
    now = 1_000_000.0
    state = updates.CheckState(snapshot=_snapshot(), checked_at=now - 90000)
    updates._write_cache(path, updates.Cache(states={"auto": state}))
    calls: list[str] = []
    monkeypatch.setattr(updates, "_query_sources", lambda *args: calls.append("network"))
    monkeypatch.setattr(updates, "_notify", lambda *args: pytest.fail("stale notification"))
    for hours in (1, 2, 4, 6, 6):
        monkeypatch.setattr(updates.time, "time", lambda: now)
        updates._check("auto", threading.Event())
        saved = updates._read_cache(path).states["auto"]
        assert saved.snapshot == state.snapshot
        assert saved.checked_at == state.checked_at
        assert saved.next_check_at == now + hours * 3600
        before = len(calls)
        updates._check("auto", threading.Event())
        assert len(calls) == before
        now = saved.next_check_at


def test_successful_refresh_replaces_mirror_snapshot_and_resets_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """官方确认撤回后替换旧镜像结果，不能把历史最高版本永久保留。"""
    path = _cache_path(tmp_path)
    updates._write_cache(
        path, updates.Cache(states={"auto": updates.CheckState(snapshot=_snapshot(source="tuna"), failures=3)})
    )
    monkeypatch.setattr(updates, "_query_sources", lambda *args: updates.Snapshot(source="pypi"))
    updates._check("auto", threading.Event())
    saved = updates._read_cache(path).states["auto"]
    assert saved.snapshot == updates.Snapshot(source="pypi")
    assert saved.failures == 0
    assert saved.next_check_at - saved.checked_at == 86400


def test_corrupt_cache_can_be_rebuilt_and_source_modes_are_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """损坏缓存可重建，切换来源不复用另一来源的检查时间。"""
    path = _cache_path(tmp_path)
    path.write_text("{broken", encoding="utf-8")
    calls: list[str] = []

    def query(mode: updates.SourceMode, stopped: threading.Event) -> updates.Snapshot:
        """按请求策略返回无稳定版快照，记录检查次数。"""
        calls.append(mode)
        return updates.Snapshot(source="tuna" if mode == "tuna" else "pypi")

    monkeypatch.setattr(updates, "_query_sources", query)
    updates._check("auto", threading.Event())
    updates._check("tuna", threading.Event())
    updates._check("auto", threading.Event())
    assert calls == ["auto", "tuna"]
    assert set(updates._read_cache(path).states) == {"auto", "tuna"}


def test_cache_write_failure_preserves_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原子替换失败时原缓存保持完整，临时文件得到清理。"""
    path = _cache_path(tmp_path)
    path.write_text("original", encoding="utf-8")

    def denied(source: str, destination: Path) -> None:
        """模拟只读文件系统拒绝原子替换。"""
        raise PermissionError("read only")

    monkeypatch.setattr(updates.os, "replace", denied)
    with pytest.raises(PermissionError):
        updates._write_cache(path, updates.Cache())
    assert path.read_text() == "original"
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["locked", "unwritable"])
def test_unavailable_cache_does_not_fail_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """锁竞争和目录不可写均静默跳过，不启动网络请求。"""
    monkeypatch.setattr(updates, "_query_sources", lambda *args: pytest.fail("unexpected request"))
    if failure == "unwritable":
        (tmp_path / "cache").write_text("occupied", encoding="utf-8")
        updates._Session().run(True, False)
    else:
        path = _cache_path(tmp_path)
        with FileLock(str(path.with_suffix(".lock"))):
            updates._Session().run(True, False)
        assert not path.exists()


@pytest.mark.parametrize(
    "enabled, interactive, ci, expected",
    [
        (False, True, False, False),
        ("auto", False, False, False),
        ("auto", True, True, False),
        ("auto", True, False, True),
        (True, False, True, True),
    ],
)
def test_enable_policy(
    enabled: bool | str, interactive: bool, ci: bool, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """auto 尊重终端和 CI，显式开启允许脚本使用，关闭则完全跳过。"""
    settings = Config(update_check={"enabled": enabled})
    monkeypatch.setattr(config_module, "get_config", lambda: settings)
    calls: list[str] = []
    monkeypatch.setattr(updates, "_check", lambda *args: calls.append("checked"))
    updates._Session().run(interactive, ci)
    assert bool(calls) is expected


def test_notification_language_python_requirement_and_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """新版不兼容 Python 时说明要求，中文提示和全局 error 级别均生效。"""
    from mineru.utils.i18n import reset_language_cache

    monkeypatch.setenv("MINERU_LANG", "zh")
    reset_language_cache()
    snapshot = updates.Snapshot(source="pypi", release=updates.Release(version="99.0", python_requires=(">=99",)))
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        cache = updates.Cache()
        assert updates._notify(snapshot, cache, time.time(), threading.Event())
        assert "需要 Python >=99" in messages[0]
        assert "包管理器升级" not in messages[0]
        config_module.get_config().log.level = "error"
        assert not updates._notify(snapshot, updates.Cache(), time.time(), threading.Event())
        assert len(messages) == 1
    finally:
        logger.remove(sink)


def test_stopped_or_current_version_never_notifies() -> None:
    """命令关闭后及版本没有更新时都不提示。"""
    from mineru.version import __version__

    assert not updates._notify(_snapshot(__version__), updates.Cache(), time.time(), threading.Event())
    stopped = threading.Event()
    stopped.set()
    assert not updates._notify(_snapshot(), updates.Cache(), time.time(), stopped)


def test_each_target_version_keeps_its_own_notice_history() -> None:
    """切换来源看到不同目标版本后，旧目标仍保留当天已提示记录。"""
    cache = updates.Cache()
    now = time.time()
    stopped = threading.Event()
    assert updates._notify(_snapshot("99.0"), cache, now, stopped)
    assert updates._notify(_snapshot("100.0"), cache, now + 1, stopped)
    assert not updates._notify(_snapshot("99.0"), cache, now + 2, stopped)
    assert updates._notify(_snapshot("99.0"), cache, now + 86400, stopped)


def test_real_cli_cached_warning_preserves_stdout(tmp_path: Path) -> None:
    """真实控制台通过默认 stderr sink 提示更新，stdout 保持完整业务 JSON。"""
    now = time.time()
    updates._write_cache(
        _cache_path(tmp_path),
        updates.Cache(states={"auto": updates.CheckState(snapshot=_snapshot(), checked_at=now, next_check_at=now + 86400)}),
    )
    script = r'''
import json, threading
import typer
from loguru import logger
from mineru.cli import main as entry
from mineru.cli.runtime import run_cli
from mineru.cli.contracts import CliContext

notified = threading.Event()
logger.add(lambda message: notified.set(), level="WARNING")
def business():
    """确认缓存提示已经发出，再完成命令。"""
    assert notified.wait(3), "cached warning not emitted"
    return {"business": "ok"}
app = typer.Typer()
@app.command()
def parse():
    """使用真实 CLI runtime 保持标准输出协议。"""
    run_cli(CliContext(json_mode=False), business, render=json.dumps)
def invoke():
    """仅替换业务命令，保留真实入口和日志配置。"""
    app(args=[], prog_name="mineru")
entry.app = invoke
entry.main()
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("MINERU_") and key != updates._OWNER_ENV}
    env.update(MINERU_HOME=str(tmp_path), MINERU_UPDATE_CHECK_ENABLED="true", MINERU_LANG="en")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"business": "ok"}
    assert result.stderr.count("A new MinerU stable release is available") == 1
    assert "WARNING" in result.stderr and "99.0" in result.stderr


@pytest.mark.parametrize(
    "application, arguments",
    [
        ("mineru.cli.main", ["--help"]),
        ("mineru.cli.main", ["parse", "--help"]),
        ("mineru.cli.main", ["--version"]),
        ("mineru.cli.main", ["version"]),
        ("mineru.cli.main", ["version", "--json"]),
        ("mineru.cli.main", ["parse"]),
        ("mineru.kit.main", ["--help"]),
        ("mineru.kit.main", ["hardware", "info", "--help"]),
        ("mineru.kit.main", ["version"]),
        ("mineru.kit.main", ["parse"]),
    ],
)
def test_help_version_and_invalid_arguments_do_not_start_worker(
    application: str, arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """实际命令树中的帮助、版本和参数错误均不创建检查线程。"""
    calls: list[str] = []
    monkeypatch.setattr(updates._Session, "run", lambda *args: calls.append("started"))
    with updates.cli_update_check():
        CliRunner().invoke(importlib.import_module(application).app, arguments)
        assert not updates._session.get().started
    assert calls == []


def test_kit_json_and_embedded_invocations_do_not_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """硬件命令的 JSON 模式和直接嵌入调用都不触发联网。"""
    from mineru.kit.commands import hardware
    from mineru.kit.main import app

    monkeypatch.setattr(hardware, "_info_data", lambda **kwargs: {"ok": True})
    with updates.cli_update_check():
        result = CliRunner().invoke(app, ["hardware", "info", "--json"])
        assert not updates._session.get().started
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"ok": True}
    result = CliRunner().invoke(app, ["hardware", "info"])
    assert result.exit_code == 0
    assert updates._session.get() is None


def test_scope_starts_once_and_suppresses_internal_children(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一控制台仅启动一次，退出清理上下文且内部进程不检查。"""
    entered = threading.Event()
    monkeypatch.setattr(updates._Session, "run", lambda *args: entered.set())
    with updates.cli_update_check():
        session = updates._session.get()
        updates.start_update_check(command_name="parse")
        updates.start_update_check(command_name="parse")
        assert session.started and entered.wait(1)
        assert os.environ[updates._OWNER_ENV] == str(os.getpid())
    assert session.stopped.is_set()
    assert updates._OWNER_ENV not in os.environ
    monkeypatch.setenv(updates._OWNER_ENV, "parent")
    with updates.cli_update_check():
        assert updates._session.get() is None


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_cli_exits_while_dns_is_blocked(tmp_path: Path, exit_code: int) -> None:
    """真实子进程在两个 DNS 请求卡住时仍及时退出，保留 stdout 和业务退出码。"""
    script = r'''
import json, os, socket, sys, threading, time
from pathlib import Path
import typer
from mineru.cli import main as entry
from mineru.cli.runtime import run_cli
from mineru.cli.contracts import CliContext

entered = threading.Event()
blocked = threading.Event()
def blocked_dns(*args, **kwargs):
    """模拟无法及时取消的系统 DNS 调用。"""
    entered.set()
    blocked.wait(60)
    raise OSError("offline")
socket.getaddrinfo = blocked_dns

def business():
    """确认后台已经进入 DNS，再返回业务输出。"""
    assert entered.wait(8), "network worker never entered DNS"
    (Path(os.environ["MINERU_HOME"]) / "business-finished").write_text(str(time.monotonic()))
    return {"business": "ok"}

app = typer.Typer()
@app.command()
def parse():
    """通过真实 CLI runtime 输出结果和指定退出码。"""
    run_cli(CliContext(json_mode=False), business, render=json.dumps, exit_code=int(sys.argv[1]))

def invoke():
    """保留真实 main 生命周期，仅替换业务命令。"""
    app(args=[], prog_name="mineru")
entry.app = invoke
entry.main()
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith("MINERU_") and key != updates._OWNER_ENV}
    env.update(MINERU_HOME=str(tmp_path), MINERU_UPDATE_CHECK_ENABLED="true", MINERU_LANG="en", PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(exit_code)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        stdout, stderr = process.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(f"CLI waited for DNS during exit: {stdout} {stderr}")
    assert process.returncode == exit_code, stderr
    assert time.monotonic() - float((tmp_path / "business-finished").read_text()) < 2
    assert json.loads(stdout) == {"business": "ok"}
    assert "WARNING" not in stderr and "Traceback" not in stderr


def test_http_index_limits_and_real_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """通过 HTTPX 模拟传输验证真实请求头、响应解析和大小限制。"""
    client_type = httpx.Client
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """返回标准索引，不访问外网。"""
        seen.append(request)
        return httpx.Response(200, json={"name": "mineru", "files": [_file("99.0")]})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=transport, **kwargs))
    result = FETCH_INDEX("pypi", time.monotonic() + 3, threading.Event())
    assert result.release.version == "99.0"
    assert seen[0].headers["Accept"] == "application/vnd.pypi.simple.v1+json"
    monkeypatch.setattr(updates, "MAX_RESPONSE_BYTES", 8)
    with pytest.raises(ValueError, match="too large"):
        FETCH_INDEX("pypi", time.monotonic() + 3, threading.Event())
