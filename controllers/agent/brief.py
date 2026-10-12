"""给大模型看的精简局势简报（几千 token）。全部从当前对局和剧本规则集生成：兵种、科技、方针、国家
都按规则集里有什么就列什么，换剧本不用改这里。"""

from __future__ import annotations

from collections import Counter

from engine.ids import ref
from engine.systems import diplomacy
from engine.systems.cities import nation_resources
from engine.systems.combat import city_defense
from engine.systems.techs import available as tech_available
from engine.systems.techs import era as tech_era


def _name(s, nid) -> str:
    n = s.nations.get(nid)
    return n.name if n else f"nation:{nid}"


def build(state, rules, nid: int, memory: dict | None = None) -> dict:
    """nid 国家的局势简报。memory：服务器记下的上次决策以来的事（丢城、得城、被宣战……）。"""
    s, g = state, state.grid
    me = s.nations[nid]
    memory = memory or {}
    cities = s.nation_cities(nid)
    pw = diplomacy.power(s, rules)
    alive = [n for n in sorted(s.nations) if s.nation_cities(n)]
    total_cities = sum(len(s.nation_cities(n)) for n in alive) or 1
    borders = diplomacy._border_pairs(s)
    mine = [u for u in s.units.values() if u.nation == nid]
    army = Counter(
        rules.units[u.type].name for u in mine if rules.units[u.type].category == "military"
    )

    # 各国：国力、关系、是否接壤、在和谁打
    nations = []
    for other in alive:
        if other == nid:
            continue
        rel = diplomacy.relation(s, nid, other)
        nations.append(
            {
                "id": ref("nation", other),
                "name": _name(s, other),
                "cities": len(s.nation_cities(other)),
                "power_vs_me": round(pw.get(other, 0) / max(1.0, pw.get(nid, 0)), 2),
                "status": rel.get("status"),
                "relation": round(diplomacy.value(s, nid, other)),
                "truce_until": rel.get("truce") if (rel.get("truce") or -1) > s.turn else None,
                "border": (min(nid, other), max(nid, other)) in borders,
                "at_war_with": [_name(s, e) for e in diplomacy.enemies(s, other)],
                "allies": [_name(s, a) for a in diplomacy.allies(s, other)],
            }
        )
    nations.sort(key=lambda x: -x["cities"])

    # 本国城市：大城和受威胁的城
    threats = {}
    for c in cities:
        threats[c.id] = sum(
            1
            for t in g.within(c.tile, 4)
            for u in s.units_at(t, "surface")
            if u.nation != nid and diplomacy.at_war(s, u.nation, nid)
        )
    shown = sorted(cities, key=lambda c: (-threats[c.id], -c.population))[:12]
    city_rows = [
        {
            "id": ref("city", c.id),
            "name": c.name,
            "population": c.population,
            "stability": c.stability,
            "defense": city_defense(s, rules, c),
            "enemy_units_near": threats[c.id],
            "capital": c.capital,
        }
        for c in shown
    ]

    # 战线（规则 AI 的执行情况）
    fronts = []
    for f in (me.strategy or {}).get("fronts", [])[:8]:
        target = s.cities.get(f.get("target_city") or f.get("objective") or -1)
        fronts.append(
            {
                "enemy": _name(
                    s,
                    f.get("enemy")
                    if f.get("enemy") is not None
                    else (target.nation if target else -1),
                ),
                "target_city": target.name if target else None,
                "mode": f.get("stance") or f.get("mode"),
                "our_strength": f.get("strength"),
                "enemy_strength": f.get("enemy_strength"),
                "note": (f.get("reason") or "")[:80],
            }
        )

    # 敌城（候选的进攻目标）：交战国离本国最近的城
    targets = []
    if cities:
        for c in s.cities.values():
            if c.nation == nid or not diplomacy.at_war(s, c.nation, nid):
                continue
            d = min(g.distance(c.tile, x.tile) for x in cities)
            targets.append((d, c))
        targets.sort(key=lambda x: (x[0], x[1].id))
    enemy_cities = [
        {
            "id": ref("city", c.id),
            "name": c.name,
            "owner": _name(s, c.nation),
            "distance": d,
            "defense": city_defense(s, rules, c),
            "capital": c.capital,
        }
        for d, c in targets[:10]
    ]

    offers = [
        {
            "offer": o["id"],
            "from": ref("nation", o["from"]),
            "from_name": _name(s, o["from"]),
            "kind": o["kind"],
        }
        for o in s.offers
        if o["to"] == nid
    ]
    since = (me.agent or {}).get("last_turn", -1000)
    news = [
        f"第{e['turn']}回合：{e['text']}"
        for e in s.chronicle
        if e["turn"] > since
        and (
            ref("nation", nid) in e.get("nations", [])
            or e["kind"] in ("coalition", "eliminated", "civil_war")
        )
    ][-12:]

    techs = []
    if me.techs is not None:
        for tid in tech_available(s, rules, nid)[:12]:
            t = rules.techs[tid]
            techs.append({"id": tid, "name": t.name, "cost": t.cost, "effect": t.description[:60]})
    from controllers.city_manager import POLICIES

    units = []
    for u in rules.units.values():
        if u.category != "military" or not u.buildable:
            continue
        units.append(
            {
                "id": u.id,
                "name": u.name,
                "domain": u.domain,
                "strength": max(u.strength, u.ranged_strength or 0),
                "cost": u.cost,
                "needs": u.requires_resources,
            }
        )

    return {
        "scenario": rules.title,
        "turn": s.turn,
        "me": {
            "id": ref("nation", nid),
            "name": me.name,
            "cities": len(cities),
            "share_of_world_cities": round(len(cities) / total_cities, 2),
            "population": sum(c.population for c in cities),
            "gold": me.gold,
            "power": round(pw.get(nid, 0)),
            "rank_by_power": 1 + sum(1 for n in alive if pw.get(n, 0) > pw.get(nid, 0)),
            "era": tech_era(s, rules, nid),
            "researching": me.research,
            "resources": sorted(nation_resources(s, rules, nid)),
            "army": dict(army.most_common()),
            "policy": me.policy,
            "at_war_with": [_name(s, e) for e in diplomacy.enemies(s, nid)],
            "allies": [_name(s, a) for a in diplomacy.allies(s, nid)],
        },
        "previous_plan": {k: me.directive.get(k) for k in ("summary", "aggression", "turn")}
        if me.directive
        else None,
        "since_last_decision": memory.get("events", [])[-15:] + news,
        "nations": nations,
        "my_cities": city_rows,
        "fronts": fronts,
        "enemy_cities_near_me": enemy_cities,
        "pending_offers": offers,
        "options": {
            "research": techs,
            "policies": {k: v["name"] for k, v in POLICIES.items()},
            "unit_types": units,
        },
    }
