"""仅由控制台入口启用的稳定版检查；网络和缓存操作均不阻塞命令退出。"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version
from pydantic import BaseModel, Field, field_validator

Source = Literal["pypi", "tuna"]
SourceMode = Literal["auto", "pypi", "tuna"]
SOURCES: dict[Source, str] = {
    "pypi": "https://pypi.org/simple/mineru/",
    "tuna": "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/mineru/",
}
CHECK_INTERVAL = 24 * 60 * 60
FAILURE_INTERVAL = 60 * 60
CHECK_DEADLINE = 3.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_CACHE_BYTES = 64 * 1024
_OWNER_ENV = "_MINERU_UPDATE_CHECK_OWNER_PID"
_CI_ENV = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "JENKINS_URL")


class Release(BaseModel):
    """缓存可用稳定版及各发布文件声明的 Python 约束。"""

    version: str
    python_requires: tuple[str, ...] = ("",)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        """拒绝缓存中的无效版本、开发版、预发布版及本地版本。"""
        version = Version(value)
        if version.is_prerelease or version.is_devrelease or version.local is not None:
            raise ValueError("Not a public stable version")
        return str(version)

    @field_validator("python_requires")
    @classmethod
    def validate_python_requires(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """校验每个 Python 约束，避免损坏缓存形成错误升级提示。"""
        if not values:
            raise ValueError("Missing Python requirements")
        for value in values:
            SpecifierSet(value)
        return values


class Snapshot(BaseModel):
    """一次成功读取的索引快照；无稳定版与请求失败分别表达。"""

    source: Source
    release: Release | None = None


class CheckState(BaseModel):
    """分别保存成功快照和失败退避，失败不会覆盖成功结果。"""

    snapshot: Snapshot | None = None
    checked_at: float = Field(default=0, ge=0, allow_inf_nan=False)
    next_check_at: float = Field(default=0, ge=0, allow_inf_nan=False)
    failures: int = Field(default=0, ge=0, le=4)


class Notice(BaseModel):
    """同一安装环境、当前版本与目标版本的最近提示记录。"""

    version: str
    shown_at: float = Field(ge=0, allow_inf_nan=False)


class Cache(BaseModel):
    """按来源策略隔离检查状态，并跨进程共享提示频率。"""

    schema_version: Literal[1] = 1
    states: dict[SourceMode, CheckState] = Field(default_factory=dict)
    notices: dict[str, Notice] = Field(default_factory=dict)


def _read_cache(path: Path) -> Cache:
    """将缺失、过大或损坏的缓存视为空缓存。"""
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_CACHE_BYTES + 1)
        if len(data) <= MAX_CACHE_BYTES:
            return Cache.model_validate_json(data)
    except (OSError, ValueError):
        pass
    return Cache()


def _write_cache(path: Path, cache: Cache) -> None:
    """在同目录原子替换缓存，进程提前退出也不会留下半个 JSON。"""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(cache.model_dump_json())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                os.unlink(temporary)


def _parse_index(payload: object, source: Source) -> Snapshot:
    """按标准文件名识别 MinerU 版本，过滤撤回文件和非稳定版。"""
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("name"), str)
        or canonicalize_name(payload["name"]) != "mineru"
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("Invalid project index")
    releases: dict[Version, set[str]] = {}
    for item in payload["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
            raise ValueError("Invalid distribution metadata")
        filename = item["filename"]
        if filename.endswith(".whl"):
            name, version, _, _ = parse_wheel_filename(filename)
        else:
            name, version = parse_sdist_filename(filename)
        if canonicalize_name(name) != "mineru":
            raise ValueError("Unexpected distribution name")
        # JSON 索引的 yanked 也可以是空字符串；只有 false 或缺省才表示未撤回。
        if item.get("yanked", False) is not False or version.is_prerelease or version.is_devrelease or version.local:
            continue
        requires_python = item.get("requires-python")
        if requires_python is None:
            requires_python = ""
        if not isinstance(requires_python, str):
            raise ValueError("Invalid Python requirement")
        SpecifierSet(requires_python)
        releases.setdefault(version, set()).add(requires_python)
    latest = max(releases) if releases else None
    release = Release(version=str(latest), python_requires=tuple(sorted(releases[latest]))) if latest is not None else None
    return Snapshot(source=source, release=release)


def _fetch_index(source: Source, deadline: float, stopped: threading.Event) -> Snapshot:
    """限时、限响应大小读取索引；不下载发行文件或访问文档内容。"""
    import httpx

    timeout = httpx.Timeout(2.0, connect=1.0, pool=1.0)
    headers = {"Accept": "application/vnd.pypi.simple.v1+json", "User-Agent": "MinerU-update-check"}
    with httpx.Client(timeout=timeout, headers=headers) as client:
        with client.stream("GET", SOURCES[source]) as response:
            response.raise_for_status()
            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=16384):
                if stopped.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("Update check expired")
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES:
                    raise ValueError("Project index too large")
    return _parse_index(json.loads(data), source)


def _query_sources(mode: SourceMode, stopped: threading.Event) -> Snapshot | None:
    """并发查询双源，官方成功即采用；镜像只在官方失败或截止后采用。"""
    sources: tuple[Source, ...] = ("pypi", "tuna") if mode == "auto" else (mode,)
    results: queue.Queue[Snapshot | None] = queue.Queue()
    deadline = time.monotonic() + CHECK_DEADLINE

    def fetch(source: Source) -> None:
        """隔离请求异常，迟到线程只写队列，不写缓存或日志。"""
        try:
            result = _fetch_index(source, deadline, stopped)
            if time.monotonic() >= deadline or stopped.is_set():
                result = None
        except Exception:
            result = None
        results.put(result)

    for source in sources:
        threading.Thread(target=fetch, args=(source,), daemon=True, name=f"mineru-update-{source}").start()
    fallback: Snapshot | None = None
    remaining = len(sources)
    while remaining and not stopped.is_set():
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            break
        try:
            result = results.get(timeout=min(timeout, 0.05))
        except queue.Empty:
            continue
        remaining -= 1
        if result is not None:
            if result.source == "pypi" or mode != "auto":
                return result
            fallback = result
    return None if stopped.is_set() else fallback


def _notify(snapshot: Snapshot, cache: Cache, now: float, stopped: threading.Event) -> bool:
    """仅向现有 warning 日志输出新稳定版，尊重级别并按安装环境去重。"""
    from loguru import logger

    from ..config import get_config
    from ..version import __version__
    from .i18n import t

    release = snapshot.release
    if release is None or Version(release.version) <= Version(__version__):
        return False
    identity = hashlib.sha256(f"{sys.prefix}\0{__version__}\0{release.version}".encode()).hexdigest()[:24]
    previous = cache.notices.get(identity)
    if previous is not None and previous.version == release.version and 0 <= now - previous.shown_at < CHECK_INTERVAL:
        return False
    if get_config().log.level in {"error", "critical"} or stopped.is_set() or sys.is_finalizing():
        return False
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    compatible = any(SpecifierSet(requirement).contains(python_version) for requirement in release.python_requires)
    if compatible:
        message = t(
            "A new MinerU stable release is available: {current} -> {latest}. "
            "Upgrade mineru using the package manager for this environment.",
            current=__version__,
            latest=release.version,
        )
    else:
        message = t(
            "A new MinerU stable release is available: {current} -> {latest}. "
            "It requires Python {requires}; this environment uses Python {python}.",
            current=__version__,
            latest=release.version,
            requires=" / ".join(release.python_requires),
            python=python_version,
        )
    logger.warning(message)
    # 清理旧记录并限制数量，避免长期共用 MINERU_HOME 时缓存无限增长。
    cache.notices = {key: value for key, value in cache.notices.items() if 0 <= now - value.shown_at < CHECK_INTERVAL}
    cache.notices = dict(sorted(cache.notices.items(), key=lambda item: item[1].shown_at)[-31:])
    cache.notices[identity] = Notice(version=release.version, shown_at=now)
    return True


def _check(mode: SourceMode, stopped: threading.Event) -> None:
    """在后台持有非阻塞跨进程锁，执行缓存读取、退避、刷新和提示。"""
    from filelock import FileLock

    home = Path(os.environ.get("MINERU_HOME") or "~/.mineru").expanduser()
    path = home / "cache" / "update-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock")), timeout=0):
        if stopped.is_set():
            return
        cache = _read_cache(path)
        state = cache.states.setdefault(mode, CheckState())
        now = time.time()
        fresh = state.snapshot is not None and 0 <= now - state.checked_at < CHECK_INTERVAL
        notified = fresh and _notify(state.snapshot, cache, now, stopped)
        if notified:
            _write_cache(path, cache)
        # 对时钟回拨或异常未来时间不无限信任，正常成功/失败退避均不超过一天。
        if 0 < state.next_check_at - now <= CHECK_INTERVAL:
            return
        state.failures = min(state.failures + 1, 4)
        state.next_check_at = now + min(2 ** (state.failures - 1), 6) * FAILURE_INTERVAL
        # 先持久化尝试时间，短命令退出或进程被终止也不会导致请求风暴。
        _write_cache(path, cache)
        snapshot = _query_sources(mode, stopped)
        if stopped.is_set() or snapshot is None:
            return
        now = time.time()
        state.snapshot = snapshot
        state.checked_at = now
        state.next_check_at = now + CHECK_INTERVAL
        state.failures = 0
        # 已由缓存提示时，本次刷新只保存结果，保持每次命令最多提示一次。
        if not notified:
            _notify(snapshot, cache, now, stopped)
        _write_cache(path, cache)


@dataclass
class _Session:
    """记录一次控制台调用；最多启动一组 daemon 线程且退出时不等待。"""

    stopped: threading.Event = field(default_factory=threading.Event)
    started: bool = False
    owner_pid: int = field(default_factory=os.getpid)

    def run(self, interactive: bool, ci: bool) -> None:
        """在后台装载配置并隔离所有更新检查异常。"""
        try:
            from ..config import get_config

            config = get_config().update_check
            if config.enabled is False or (config.enabled == "auto" and (not interactive or ci)):
                return
            if not self.stopped.is_set():
                _check(config.source, self.stopped)
        except Exception:
            # 更新检查属于尽力提示，不打印网络、配置或缓存失败日志。
            return


_session: ContextVar[_Session | None] = ContextVar("mineru_update_check", default=None)


@contextmanager
def cli_update_check() -> Iterator[None]:
    """显式限定控制台生命周期，并让内部子进程继承跳过标记。"""
    if _session.get() is not None or os.environ.get(_OWNER_ENV):
        yield
        return
    session = _Session()
    token = _session.set(session)
    os.environ[_OWNER_ENV] = str(os.getpid())
    try:
        yield
    finally:
        session.stopped.set()
        _session.reset(token)
        os.environ.pop(_OWNER_ENV, None)


def start_update_check(*, command_name: str | None = None, json_mode: bool = False) -> None:
    """仅在控制台业务命令解析完成后启动，SDK、版本和 JSON 调用直接跳过。"""
    session = _session.get()
    if session is None or session.started or session.owner_pid != os.getpid() or json_mode or command_name == "version":
        return
    session.started = True
    try:
        interactive = sys.stdin.isatty() and sys.stderr.isatty()
        ci = any(os.environ.get(name, "").lower() not in {"", "0", "false", "no"} for name in _CI_ENV)
        threading.Thread(target=session.run, args=(interactive, ci), daemon=True, name="mineru-update-check").start()
    except Exception:
        return


__all__ = ["cli_update_check", "start_update_check"]
