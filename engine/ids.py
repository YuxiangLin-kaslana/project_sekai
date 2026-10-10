"""实体 ID。

内部用整数（来自 GameState.next_id，只增不减，所有实体共用一个计数器）；
对外（指令、事件、前端）用字符串 `<kind>:<n>`，如 `unit:12`、`city:3`、`nation:1`。
"""

from __future__ import annotations

KINDS = ("nation", "city", "unit")


def ref(kind: str, n: int) -> str:
    if kind not in KINDS:
        raise ValueError(f"未知的实体类型 {kind!r}")
    return f"{kind}:{n}"


def parse_ref(text: str, kind: str) -> int:
    """`unit:12` → 12；类型不符或格式不对时抛出 ValueError。"""
    prefix, sep, num = text.partition(":")
    if sep != ":" or prefix != kind or not num.isdigit():
        raise ValueError(f"{text!r} 不是合法的 {kind} ID（应为 {kind}:<数字>）")
    return int(num)
