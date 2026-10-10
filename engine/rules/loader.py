"""从 YAML 目录加载规则集。

这是引擎里少数做文件 I/O 的地方，只在启动时调用；结算（step）本身不读文件。

目录结构：
    rules/<name>/ruleset.yaml    名称、版本、继承、世界生成 / 城市 / 堆叠 / 开局参数
    rules/<name>/terrain.yaml
    rules/<name>/resources.yaml
    rules/<name>/units.yaml
    rules/<name>/buildings.yaml   市中心建筑 + 城外设施
    rules/<name>/nations.yaml     国家模板（名字、颜色、城市名、东方 / 西方系列）
    rules/<name>/names.yaml       名字库：每个系列的国名、城市名、人名

继承：ruleset.yaml 写 `extends: <父规则集名>`，子规则集里的同名条目整条覆盖父规则集，
值写 null 表示删除该条目；ruleset.yaml 里的配置段（worldgen、cities 等）按字段合并。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from engine.rules.schema import Ruleset

RULES_ROOT = Path(__file__).resolve().parents[2] / "rules"
_ENTRY_FILES = ("terrain", "resources", "units", "buildings", "nations")
_CONFIG_SECTIONS = (
    "worldgen",
    "cities",
    "stacking",
    "encirclement",
    "tactics",
    "morale",
    "healing",
    "stability",
    "pillage",
    "start",
)  # 继承时按字段合并


class RulesetError(Exception):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise RulesetError(f"{path}: YAML 格式错误：{e}") from e
    if not isinstance(data, dict):
        raise RulesetError(f"{path}: 顶层必须是映射")
    return data


def _load_raw(directory: Path, root: Path, chain: tuple[str, ...]) -> dict[str, Any]:
    if not (directory / "ruleset.yaml").exists():
        raise RulesetError(f"{directory}: 找不到 ruleset.yaml")
    meta = _read_yaml(directory / "ruleset.yaml")
    raw: dict[str, Any] = {name: {} for name in _ENTRY_FILES}

    parent = meta.pop("extends", None)
    if parent is not None:
        if parent in chain:
            raise RulesetError(f"规则集循环继承：{' -> '.join((*chain, parent))}")
        base = _load_raw(root / parent, root, (*chain, parent))
        raw.update({k: v for k, v in base.items() if k not in _ENTRY_FILES})
        for name in _ENTRY_FILES:
            raw[name] = dict(base[name])

    for key, value in meta.items():
        if key in _CONFIG_SECTIONS and isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value

    for name in _ENTRY_FILES:
        for key, entry in _read_yaml(directory / f"{name}.yaml").items():
            if entry is None:
                raw[name].pop(key, None)
            else:
                raw[name][key] = entry
    # 名字库（东方 / 西方……）：子规则集的同名系列整个覆盖
    names = _read_yaml(directory / "names.yaml")
    if names:
        raw["names"] = {**(raw.get("names") or {}), **names}
    return raw


def load_ruleset(name_or_path: str | Path, root: Path = RULES_ROOT) -> Ruleset:
    """按名称（rules/ 下的子目录）或路径加载规则集。"""
    directory = Path(name_or_path)
    if not directory.is_dir():
        directory = root / str(name_or_path)
    raw = _load_raw(directory, directory.parent, (directory.name,))

    for name in _ENTRY_FILES:
        for key, entry in raw[name].items():
            if not isinstance(entry, dict):
                raise RulesetError(f"{name}.{key}: 条目必须是映射")
            entry["id"] = key

    canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    raw["id"] = f"{raw.get('name')}@{raw.get('version')}#{digest}"

    try:
        return Ruleset.model_validate(raw)
    except ValidationError as e:
        raise RulesetError(f"{directory}: 规则集校验失败\n{e}") from e


def list_scenarios(root: Path = RULES_ROOT) -> list[str]:
    """可以开局的剧本：rules/ 下有 ruleset.yaml、名字不以下划线开头的目录（_world 等是共用的基底）。"""
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "ruleset.yaml").exists()
    )
