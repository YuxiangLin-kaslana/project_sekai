"""打 Windows 下载包（在 Linux 上就能打，不需要 Windows 机器）。

    uv run python tools/windows/build.py              # 版本号自动按日期递增
    uv run python tools/windows/build.py --version 2026.10.10.3 --base-url https://example.com

做的事：
1. 下载 Windows 官方嵌入式 Python（PY_VERSION），解压到 Sekai/python/，配好 python3xx._pth。
2. 用 pip 下载各依赖的 win_amd64 安装包（版本和本机一致，REQUIREMENTS），解压进 python/Lib/site-packages。
3. 复制游戏代码到 Sekai/app/（engine、controllers、server、rules、地球地图数据……），写 VERSION。
4. 打两个包到 downloads/windows/（服务器 /downloads/windows/ 直接提供下载）：
   - Sekai-<版本>-win64.zip：完整包（解压即玩）；
   - sekai-app-<版本>.zip：只有游戏代码，给启动器自动更新用；
   并写更新清单 latest.json（版本、运行环境、两个包的地址和 SHA-256）和下载页 index.html。
运行环境（Python 版本 + 依赖版本）没变时，玩家的启动器只下载游戏代码包自动更新；变了就提示重新下完整包。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "downloads" / "windows"
WORK = ROOT / "out" / "windows-build"
CACHE = ROOT / "out" / "windows-cache"

PY_VERSION = "3.14.5"
REQUIREMENTS = [  # 和本机 uv 环境一致的版本（uvicorn 不带 [standard]：uvloop 没有 Windows 版，本机游玩用不上）
    "fastapi==0.142.4",
    "starlette==1.7.0",
    "pydantic==2.13.5",
    "pydantic-core==2.46.5",
    "numba==0.68.0",
    "llvmlite==0.50.0",
    "numpy==2.5.3",
    "pillow==12.3.0",
    "pyyaml==6.0.3",
    "uvicorn==0.54.0",
    "click==8.5.0",
    "colorama",  # click 在 Windows 上要用
    "h11==0.16.0",
    "anyio==4.15.1",
    "idna==3.20",
    "annotated-types==0.8.0",
    "typing-extensions==4.16.0",
    "typing-inspection==0.4.4",
]
APP_PARTS = ["engine", "controllers", "server", "rules", "data/earth", "游戏规则.md", "README.md"]
KEEP_VERSIONS = 3  # downloads/windows/ 里保留最近几个版本的包


def runtime_id() -> str:
    """运行环境的指纹：Python 版本 + 依赖版本。变了就要重新下完整包。"""
    material = "\n".join([PY_VERSION, *REQUIREMENTS])
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def next_version() -> str:
    today = datetime.date.today().strftime("%Y.%m.%d")
    latest = OUT / "latest.json"
    n = 1
    if latest.exists():
        old = json.loads(latest.read_text(encoding="utf-8")).get("version", "")
        if old.startswith(today + "."):
            n = int(old.rsplit(".", 1)[1]) + 1
    return f"{today}.{n}"


def fetch(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"下载 {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
        tmp.rename(target)
    return target


def build_python(dest: Path) -> None:
    """嵌入式 Python + 依赖。"""
    tag = "".join(PY_VERSION.split(".")[:2])  # 3.14.5 → 314
    embed = fetch(
        f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip",
        CACHE / f"python-{PY_VERSION}-embed-amd64.zip",
    )
    with zipfile.ZipFile(embed) as z:
        z.extractall(dest)
    # 路径：标准库 zip、python 目录、site-packages、游戏代码；import site 让 .pth 生效
    (dest / f"python{tag}._pth").write_text(
        f"python{tag}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="utf-8"
    )
    wheels = CACHE / f"wheels-{runtime_id()}"
    if not wheels.exists():
        print("下载 Windows 依赖包")
        subprocess.run(
            [
                "uvx", "pip", "download", "--only-binary=:all:", "--platform", "win_amd64",
                "--python-version", PY_VERSION.rsplit(".", 1)[0], "--implementation", "cp",
                "--abi", f"cp{tag}", "-d", str(wheels), *REQUIREMENTS,
            ],
            check=True,
        )  # fmt: skip
    site = dest / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as z:
            z.extractall(site)
    # 少数 wheel 把文件放在 <名字>.data/purelib|platlib 里：挪到 site-packages 根
    for data in site.glob("*.data"):
        for part in ("purelib", "platlib"):
            src = data / part
            if src.exists():
                shutil.copytree(src, site, dirs_exist_ok=True)
        shutil.rmtree(data)
    (dest / "RUNTIME").write_text(runtime_id(), encoding="utf-8")


def build_app(dest: Path, version: str) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.nbi", "*.nbc")
    for part in APP_PARTS:
        src = ROOT / part
        target = dest / part
        if src.is_dir():
            shutil.copytree(src, target, ignore=ignore)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    shutil.copyfile(HERE / "launcher.py", dest / "launcher.py")
    (dest / "VERSION").write_text(version, encoding="utf-8")


def zip_dir(src: Path, target: Path, prefix: str = "") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(src.rglob("*")):
            if path.is_file():
                z.write(path, prefix + path.relative_to(src).as_posix())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def publish_github(version: str, files: list[Path]) -> None:
    """在 GitHub 发 Release v<版本>，附完整包、更新包和更新清单。"""
    import os

    notes = (
        "Windows 64 位完整包：解压后双击“启动游戏.cmd”，游戏在浏览器里打开、本机运行。"
        "之后每次启动会自动检查更新。\n\n"
        f"- {files[0].name}：完整包（嵌入式 Python + 依赖 + 游戏）\n"
        f"- {files[1].name}：只含游戏代码，供自动更新\n- latest.json：更新清单"
    )
    subprocess.run(
        [
            os.environ.get("GH", "gh"), "release", "create", f"v{version}",
            "--repo", os.environ.get("SEKAI_REPO", "YuxiangLin-kaslana/project_sekai"),
            "--target", "main", "--title", f"Project Sekai {version}（Windows）",
            "--notes", notes, *map(str, files),
        ],
        check=True,
    )  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=None)
    parser.add_argument("--base-url", default="https://sekai.yuxianglin.xyz")
    parser.add_argument(
        "--github",
        action="store_true",
        help="打完包同时在 GitHub 发 Release v<版本>（需要 gh 命令行已登录；GH 环境变量可指定 gh 路径）",
    )
    args = parser.parse_args()
    version = args.version or next_version()
    base = args.base_url.rstrip("/") + "/downloads/windows/"

    shutil.rmtree(WORK, ignore_errors=True)
    game = WORK / "Sekai"
    build_python(game / "python")
    build_app(game / "app", version)
    shutil.copyfile(HERE / "launcher.py", game / "launcher.py")
    cmd = (
        '@echo off\r\nchcp 65001 > nul\r\ncd /d "%~dp0"\r\nset PYTHONIOENCODING=utf-8\r\n'
        '"python\\python.exe" launcher.py %*\r\nif errorlevel 1 pause\r\n'
    )
    (game / "启动游戏.cmd").write_bytes(cmd.encode("utf-8"))
    (game / "说明.txt").write_text(
        "Project Sekai Windows 版\r\n\r\n"
        "双击“启动游戏.cmd”：会先检查更新，然后在浏览器里打开游戏（本机运行，不需要联网也能玩）。\r\n"
        "第一次启动要编译地图生成程序，大约一两分钟。关闭黑色窗口即可退出。\r\n"
        "有新版本时会自动下载更新；需要换运行环境时会打开下载页，下载新的完整包解压覆盖即可。\r\n",
        encoding="utf-8",
    )

    full = OUT / f"Sekai-{version}-win64.zip"
    app_zip = OUT / f"sekai-app-{version}.zip"
    zip_dir(game, full, prefix="Sekai/")
    zip_dir(game / "app", app_zip)
    manifest = {
        "version": version,
        "runtime": runtime_id(),
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "app_url": base + app_zip.name,
        "app_sha256": sha256(app_zip),
        "app_size": app_zip.stat().st_size,
        "full_url": base + full.name,
        "full_sha256": sha256(full),
        "full_size": full.stat().st_size,
        "page_url": base,
    }
    (OUT / "latest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # 只留最近几个版本
    for pattern in ("Sekai-*-win64.zip", "sekai-app-*.zip"):
        old = sorted(OUT.glob(pattern), key=lambda p: p.stat().st_mtime)[:-KEEP_VERSIONS]
        for p in old:
            p.unlink()
    size = full.stat().st_size / 1e6
    (OUT / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project Sekai Windows 版</title>
<style>body{{font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:640px;margin:40px auto;padding:0 16px;background:#14110c;color:#efe4cc}}
a.btn{{display:inline-block;padding:10px 18px;border-radius:8px;background:#d8b469;color:#1a140a;font-weight:600;text-decoration:none}}
small{{color:#a39b86}}</style>
<h1>Project Sekai · Windows 版</h1>
<p>版本 {html.escape(version)} · 64 位 Windows 10 / 11 · {size:.0f} MB</p>
<p><a class="btn" href="{html.escape(full.name)}">下载完整包</a></p>
<p>解压后双击“启动游戏.cmd”，游戏在浏览器里打开，本机运行。之后每次启动会自动检查更新。</p>
<p><small>SHA-256：{manifest["full_sha256"]}</small></p>
""",
        encoding="utf-8",
    )
    if args.github:
        publish_github(version, [full, app_zip, OUT / "latest.json"])
    print(
        f"完成：{full.name}（{size:.0f} MB）、{app_zip.name}，版本 {version}，运行环境 {runtime_id()}"
    )


if __name__ == "__main__":
    main()
