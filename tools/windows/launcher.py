"""Project Sekai Windows 版启动器：检查更新 → 启动本机游戏服务器 → 打开浏览器。

目录结构（解压后）：
    Sekai/
      启动 Sekai.cmd      双击运行（调用下面的 python\\python.exe launcher.py）
      launcher.py         本文件
      python/             Windows 嵌入式 Python 和依赖（RUNTIME 文件记着运行环境版本）
      app/                游戏代码（VERSION 文件记着游戏版本；自动更新只替换这个目录）

自动更新：
- 每次启动先读更新清单（SEKAI_UPDATE_URL，默认官网 /downloads/windows/latest.json）。
- 有新版本、运行环境没变：下载游戏代码包，校验 SHA-256，解压到 app.new，再整个换掉 app/。
- 运行环境也变了（Python 或依赖升级）：打开下载页，让玩家下载新的完整包。
- 游戏开着时每两小时再查一次（server/app.py 的 /api/client/update），网页顶部提示“重启即可更新”。
连不上更新服务器时照常启动，不影响离线游玩。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
UPDATE_URL = os.environ.get(
    "SEKAI_UPDATE_URL", "https://sekai.yuxianglin.xyz/downloads/windows/latest.json"
)
DEFAULT_PORT = 8095


def version_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.strip().split("."))
    except ValueError:
        return (0,)


def read(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return default


def local_version() -> str:
    return read(APP / "VERSION", "0")


def runtime_id() -> str:
    return read(ROOT / "python" / "RUNTIME", "")


def fetch_manifest(timeout: float = 8) -> dict:
    request = urllib.request.Request(UPDATE_URL, headers={"User-Agent": "SekaiLauncher"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, target: Path, sha256: str) -> None:
    """下载到 target，边下边算 SHA-256，校验不过就删掉并报错。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "SekaiLauncher"})
    with urllib.request.urlopen(request, timeout=30) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1 << 16):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total and time.monotonic() - last > 0.5:
                last = time.monotonic()
                print(
                    f"\r  下载中 {done * 100 // total:3d}%  ({done >> 20} / {total >> 20} MB)",
                    end="",
                )
    print()
    if digest.hexdigest() != sha256:
        target.unlink(missing_ok=True)
        raise ValueError("下载的文件校验不通过（可能传输出错），这次不更新")


def apply_app_update(manifest: dict) -> None:
    """下载新的游戏代码包，解压到 app.new，确认完整后整个换掉 app/（失败时原来的 app/ 不动）。"""
    work = ROOT / "update"
    package = work / f"app-{manifest['version']}.zip"
    download(manifest["app_url"], package, manifest["app_sha256"])
    staging = ROOT / "app.new"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(package) as z:
        z.extractall(staging)
    if (
        not (staging / "server" / "app.py").exists()
        or read(staging / "VERSION") != manifest["version"]
    ):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("更新包内容不完整，这次不更新")
    old = ROOT / "app.old"
    shutil.rmtree(old, ignore_errors=True)
    if APP.exists():
        APP.rename(old)
    staging.rename(APP)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    # 启动器本身也跟着游戏代码更新（下次启动生效）
    fresh = APP / "launcher.py"
    if fresh.exists():
        shutil.copyfile(fresh, ROOT / "launcher.py")


def check_and_update() -> None:
    print(f"Project Sekai  当前版本 {local_version()}")
    print("检查更新…")
    try:
        manifest = fetch_manifest()
    except Exception as e:  # noqa: BLE001 —— 连不上就照常启动
        print(f"  连不上更新服务器（{type(e).__name__}），先用当前版本")
        return
    latest = manifest.get("version", "0")
    if version_tuple(latest) <= version_tuple(local_version()):
        print("  已是最新版本")
        return
    if manifest.get("runtime") != runtime_id():
        print(
            f"  发现新版本 {latest}，它需要新的运行环境：请下载完整安装包（已在浏览器打开下载页）"
        )
        webbrowser.open(manifest.get("page_url") or manifest.get("full_url") or UPDATE_URL)
        return
    print(f"  发现新版本 {latest}，正在更新…")
    try:
        apply_app_update(manifest)
        print(f"  已更新到 {latest}")
    except Exception as e:  # noqa: BLE001
        print(f"  更新失败：{e}；先用当前版本")


def free_port(preferred: int) -> int:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


def main() -> None:
    os.chdir(ROOT)
    if "--no-update" not in sys.argv:
        check_and_update()
    sys.path.insert(0, str(APP))
    os.environ["SEKAI_DESKTOP"] = "1"
    os.environ["SEKAI_VERSION"] = local_version()
    os.environ["SEKAI_UPDATE_URL"] = UPDATE_URL
    os.environ["SEKAI_RUNTIME"] = runtime_id()
    port = free_port(DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/"
    print()
    print(f"游戏服务器启动中：{url}")
    print("第一次启动要编译地图生成程序，大约一两分钟；之后就快了。")
    print("关闭这个窗口即可退出游戏。")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    import uvicorn

    uvicorn.run("server.app:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
