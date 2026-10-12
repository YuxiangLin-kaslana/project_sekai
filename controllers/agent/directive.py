"""agent 方针（Nation.directive）的软权重：规则 AI 在打分里加上这些，方针明显生效，但不越过合法性和常识。"""

from __future__ import annotations

import math


def _d(state, nid) -> dict:
    n = state.nations.get(nid)
    if n is None or not (n.agent or {}).get("enabled"):
        return {}
    return n.directive or {}


def aggression(state, nid) -> float:
    """进攻性 −1（防守）… 1（进攻）。"""
    return float(_d(state, nid).get("aggression", 0.0))


def nation_weight(state, nid, other) -> float:
    """对某国：正 = 优先打它，负 = 避免和它打（−1 … 1）。"""
    return float(_d(state, nid).get("nations", {}).get(str(other), 0.0))


def city_weight(state, nid, cid) -> float:
    """重点攻取的城市（0 … 1）。"""
    return float(_d(state, nid).get("cities", {}).get(str(cid), 0.0))


def unit_weight(state, nid, unit_type: str) -> float:
    """兵种偏好：正 = 多造，负 = 少造（−1 … 1）。"""
    return float(_d(state, nid).get("units", {}).get(unit_type, 0.0))


def attack_need(state, nid, enemy) -> float:
    """战线进攻要的兵力优势乘这个系数：越激进、越想打这个国家，要的优势越小（平滑，0.45 … 2.2）。"""
    return math.exp(-0.6 * aggression(state, nid) - 0.4 * nation_weight(state, nid, enemy))
