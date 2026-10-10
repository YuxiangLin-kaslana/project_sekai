"""确定性战斗。射程预览和结算用同一套检查。

- 每国在一格里最多 2 个军事单位（堆叠上限），只要不在行军、不在编队里，每个单位都可以各自攻击。
- 近战只在同一格里交战：近战攻击 = 开进相邻的敌军格，在格子里交战；打完不论胜负都留在那格。
  双方部队同处一格（交战格）时，以后每回合在格子里自动接着打，直到一方被消灭或撤出。
  城防还在的敌国城市不能开进去，近战只能从相邻格强攻城墙；城防打空后才能开进城与守军交战，
  守军打光即占领。
- 带 area_damage 标签的远程单位（炮兵、战列舰）炮击整格：目标格里的每个敌方军事单位
  （和敌国城市）都按各自的防御力受到伤害。
- 被包围的单位、城市战斗力打折（见 engine/systems/encirclement.py）。
- 战术修正（attack_modifiers，数值在规则集 tactics 和地形 attack 里）：从丘陵出击居高临下加成、
  从沼泽雨林雪地出击减成；近战侧击——目标格四周每多一面有本方陆军 +10%（最多三面），城市也算；
  隔着河近战进攻 −25%（走桥 −10%）。防守方仍按地形 defense、驻守、步兵防守等加成。
- 炮击压制（suppress 标签：炮兵、战列舰、轰炸机）：打中的部队（炮击波及的整格都算）失去驻守加成；
  远程攻击先于近战结算，同一回合的“炮火准备、步兵冲锋”打的就是没有驻守加成的守军。
- 士气（morale_penalty）：交战双方各比本国 / 对方的城市数和军队数，取较低的那项比例，
  低于规则集 morale.start% 开始掉战斗力，低到 morale.full% 时掉满 max_penalty%（城市防御也算）。
- 飞机出击 = 起飞 → 攻击 → 返航降落：只有停在本国机场（城市、机场、航母）上、本回合还没转场的飞机
  才能出击，打击作战半径 range 内的目标，打完飞回原基地；一次出击只攻击一次，用掉整个回合。
- 防空单位（anti_air > 0，高射炮）只打飞机：能主动射击射程内的敌方航空单位；射程内的格子挨敌方空袭时，
  覆盖这一格的最强一门高射炮先对来袭飞机开火，飞机带伤投弹（伤害随之下降），血打光就被击落。
"""

from __future__ import annotations

from collections import Counter

from engine.actions import CityAttack
from engine.ids import parse_ref, ref
from engine.state import MOVE_STEPS, City, Unit
from engine.systems import encirclement
from engine.systems.cities import city_stat, nation_supply
from engine.systems.context import StepContext
from engine.systems.progression import gain, level, unit_strength

FLAK_FACTOR = 30  # 一门高射炮对来袭飞机的火力：FLAK_FACTOR × 防空火力 ÷ 飞机战斗力（6–40）
FLAK_GUNS = 3  # 最多几门一起开火
FLAK_CAP = 70  # 一次空袭最多被高射火力打掉这么多生命


def attack_range(rules, actor: Unit | City) -> int:
    if isinstance(actor, City):
        return 2
    ut = rules.units[actor.type]
    return ut.range if ut.ranged_strength or ut.domain == "air" or ut.anti_air or ut.torpedo else 1


def attack_kind(rules, actor) -> str:
    if isinstance(actor, City):
        return "ranged"
    ut = rules.units[actor.type]
    if ut.anti_air:
        return "anti_air"
    if ut.torpedo:
        return "torpedo"
    return "air" if ut.domain == "air" else "ranged" if ut.ranged_strength else "melee"


def lock_reason(state, rules, actor) -> str | None:
    if actor.attacked_turn == state.turn:
        return "本回合已经攻击"
    if isinstance(actor, City):
        return "城防已被摧毁" if actor.hp <= 0 else None
    ut = rules.units[actor.type]
    if ut.category != "military":
        return "只有军事单位可以攻击"
    if actor.layer == "cargo":
        return "在船上，先下船"
    if "carrier" in ut.tags:
        return "航空母舰没有攻击手段，靠舰载机作战"
    if "unarmed" in ut.tags:
        return f"{ut.name}没有攻击手段"
    if actor.formation is not None:
        return "编队不能攻击，请先解散编队"
    if actor.destination is not None:
        return "行军中不能交战，先停下（驻守）"
    if ut.domain == "air":
        # 出击 = 起飞 → 攻击 → 返航降落：要从本国机场起飞，本回合转过场的不能再出击
        from engine.systems.movement import is_air_base

        if actor.moves_left < ut.moves:
            return "本回合已经转场，下一回合再出击"
        if not is_air_base(state, rules, actor.nation, actor.tile):
            return "不在本国机场或航母上，不能起飞出击"
    if actor.moves_left * MOVE_STEPS + actor.road_credit <= 0:
        return "移动力已用完，下一回合恢复"
    return None


def line_tiles(grid, start, end) -> list[int]:
    """Cube interpolation, including the shortest image across the map seam."""
    aq, ar = int(grid.axial_q[start]), int(grid.axial_r[start])
    bq, br = int(grid.axial_q[end]), int(grid.axial_r[end])
    if grid.wrap_x:
        bq = min(
            (bq - grid.width, bq, bq + grid.width),
            key=lambda q: (abs(q - aq) + abs(br - ar) + abs(q + br - aq - ar), q),
        )
    distance = grid.distance(start, end)
    tiles = []
    for i in range(1, distance):
        t = i / distance
        x, z = aq + (bq - aq) * t + 1e-6, ar + (br - ar) * t + 1e-6
        y = -x - z
        rx, ry, rz = round(x), round(y), round(z)
        dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dz > dy:
            rz = -rx - ry
        col = rx + (rz - (rz & 1)) // 2
        try:
            tile = grid.idx(col, rz)
        except IndexError:
            continue
        tiles.append(tile)
    return tiles


def _note(notes, label, pct):
    if notes is not None and pct:
        notes.append((label, pct))


def line_blocked(state, rules, actor, source, tile) -> bool:
    """从 source 打 tile 的射线有没有被山脉 / 树林挡住。间接射击（indirect 标签，炮兵）不需要视线。"""
    if isinstance(actor, Unit) and "indirect" in rules.units[actor.type].tags:
        return False
    return any(
        rules.terrain_at(int(state.grid.terrain[t])).blocks_sight
        for t in line_tiles(state.grid, source, tile)
    )


def morale_penalty(state, rules, nation, against) -> tuple[int, int]:
    """nation 和 against 交战时的士气：返回（战斗力 −%，兵力比例 %）。兵力比例 = 本国 / 对方的
    城市数、军队数里较低的那项；各国城市数、军队数每回合统计一次。"""
    cache = state._power_cache
    if cache.get("turn") != state.turn:
        cache.clear()
        cache["turn"] = state.turn
        cache["cities"] = Counter(c.nation for c in state.cities.values())
        cache["army"] = Counter(
            u.nation for u in state.units.values() if rules.units[u.type].category == "military"
        )
    ratios = [
        100 * counts[nation] // counts[against]
        for counts in (cache["cities"], cache["army"])
        if counts[against] > 0
    ]
    if not ratios:
        return 0, 100
    ratio, cfg = min(ratios), rules.morale
    if ratio >= cfg.start:
        return 0, ratio
    penalty = cfg.max_penalty * (cfg.start - ratio) // max(1, cfg.start - cfg.full)
    return min(cfg.max_penalty, penalty), ratio


def _morale(state, rules, actor, against, value, notes):
    if against is None or against == actor.nation:
        return value
    penalty, ratio = morale_penalty(state, rules, actor.nation, against)
    if not penalty:
        return value
    _note(notes, f"士气低落（兵力只有对方 {ratio}%）", -penalty)
    return value * (100 - penalty) // 100


def strength(
    state, rules, actor, *, defending=False, ranged=False, hp=None, notes=None, against=None
) -> int:
    """战斗力。ranged：远程 / 防空火力；hp：按这个生命值算（预估带伤后的战斗力）；
    notes：传入列表时把各项修正（说明, 百分比）记进去，给攻击预览显示；
    against：对手国家，算兵力悬殊的士气。"""
    ring = encirclement.status(state, rules)
    if isinstance(actor, City):
        defense = city_stat(rules, actor, "defense", 100)
        _note(notes, "城防建筑", defense - 100)
        value = (12 + actor.population * 2) * defense // 100
        bonus = 8 * (level(actor.xp, "city") - 1)
        _note(notes, f"城市 Lv.{level(actor.xp, 'city')}", bonus)
        value = max(1, value * (100 + bonus) // 100)
        if actor.id in ring.cities:
            _note(notes, "被包围", -rules.encirclement.city_penalty)
            value = value * (100 - rules.encirclement.city_penalty) // 100
        return max(1, _morale(state, rules, actor, against, value, notes))
    ut = rules.units[actor.type]
    value = ut.strength
    if ranged:
        value = ut.ranged_strength or ut.anti_air or ut.torpedo or ut.strength
    terrain = rules.terrain_at(int(state.grid.terrain[actor.tile]))
    mods = [(ut.name, m) for m in ut.modifiers]
    city = state.cities.get(int(state.grid.city_of[actor.tile]))
    if city is not None and city.nation == actor.nation:
        mods += [
            (rules.buildings[b].name, m)
            for b in city.buildings
            for m in rules.buildings[b].modifiers
        ]
    for source, m in mods:
        w = m.when
        if m.target != "unit" or m.stat != "strength":
            continue
        if w.defending is not None and w.defending != defending:
            continue
        if w.terrain_in is not None and terrain.id not in w.terrain_in:
            continue
        if w.unit_tag is not None and w.unit_tag not in ut.tags:
            continue
        if m.op != "add":
            _note(notes, source + ("防守" if w.defending else ""), m.value - 100)
        value = value + m.value if m.op == "add" else value * m.value // 100
    if ut.category == "military":
        # 战略资源加成：燃料、钨芯弹、铝制机体……（规则在 resources.yaml 的 combat）
        for rid in sorted(nation_supply(state, rules, actor.nation)):
            res = rules.resources[rid]
            for bonus in res.combat:
                if actor.type in bonus.units or ut.domain in bonus.domains:
                    _note(notes, f"{res.name}：{bonus.label}", bonus.strength)
                    value = value * (100 + bonus.strength) // 100
    if actor.corps is not None:
        from engine.systems.generals import bonus, commander

        general = commander(state, rules, actor)
        if general is not None:
            extra = bonus(rules, general.rank)
            _note(notes, f"{general.title}·{general.name} {'★' * general.rank}", extra)
            value = value * (100 + extra) // 100
    lv = level(actor.xp)
    _note(notes, f"经验 Lv.{lv}", lv * 25 / 2)
    value = unit_strength(value, actor.xp)
    # 带伤按生命百分比算（步兵、坦克生命上限更高）
    health = (actor.hp if hp is None else hp) * 100 // ut.hp
    _note(notes, "带伤", (50 + health // 2) - 100)
    value = value * (50 + health // 2) // 100
    if defending:
        _note(notes, f"地形：{terrain.name}", terrain.defense)
        _note(notes, "驻守", 25 if actor.fortified else 0)
        value = value * max(25, 100 + terrain.defense + (25 if actor.fortified else 0)) // 100
    if actor.id in ring.units:
        _note(notes, "被包围", -rules.encirclement.penalty)
        value = value * (100 - rules.encirclement.penalty) // 100
    return max(1, _morale(state, rules, actor, against, value, notes))


def attack_modifiers(state, rules, actor, tile, kind) -> list[tuple[str, int]]:
    """进攻方的战术修正 [(说明, 百分比)]：出击地形（所有地面 / 舰船进攻），近战另算侧击和跨河。
    tile 是目标格；同格交战时 actor 就在 tile 里（四周六面都算侧击，没有跨河）。"""
    if not isinstance(actor, Unit) or actor.layer != "surface" or kind in ("air", "anti_air"):
        return []
    g, cfg = state.grid, rules.tactics
    out = []
    terrain = rules.terrain_at(int(g.terrain[actor.tile]))
    if terrain.attack:
        out.append((f"从{terrain.name}出击", terrain.attack))
    if kind != "melee":
        return out
    sides = sum(
        1
        for n in g.neighbors_of(tile)
        if n != actor.tile
        and any(
            u.nation == actor.nation
            and u.id != actor.id
            and rules.units[u.type].category == "military"
            and rules.units[u.type].domain == "land"
            for u in state.units_at(n, "surface")
        )
    )
    flank = min(cfg.flank_max, sides)
    if flank:
        out.append((f"侧击（{flank} 面）", flank * cfg.flank))
    if actor.tile != tile:
        from engine.systems.roads import crosses_river

        if crosses_river(g, actor.tile, tile):
            bridge = (min(actor.tile, tile), max(actor.tile, tile)) in state.roads
            out.append(("跨河进攻（走桥）", -cfg.bridge) if bridge else ("跨河进攻", -cfg.river))
    return out


def _damage_scale(rules, actor, target) -> int:
    """伤害倍率（%）：兵种对部队 / 对城市的倍率 × 军种克制（规则集 domain_damage：舰炮打岸上部队强、
    陆军打舰船弱、飞机打舰船强……）。城市的防御射击按陆军算。"""
    if isinstance(target, City):
        return rules.units[actor.type].damage_vs_city if isinstance(actor, Unit) else 100
    attacker = rules.units[actor.type].domain if isinstance(actor, Unit) else "land"
    vs = rules.domain_damage.get(attacker, {}).get(rules.units[target.type].domain, 100)
    base = rules.units[actor.type].damage_vs_units if isinstance(actor, Unit) else 100
    return base * vs // 100


def _apply(value, mods) -> int:
    return max(1, value * (100 + sum(v for _, v in mods)) // 100) if mods else value


def target_at(state, rules, actor, tile):
    city = state.city_at(tile)
    if (
        isinstance(actor, Unit)
        and attack_kind(rules, actor) == "melee"
        and city is not None
        and city.nation != actor.nation
        and city.hp > 0
    ):
        return city  # 城防还在：近战只能强攻城墙，进不了城和守军交战
    enemies = [
        u
        for u in state.units_at(tile)
        if u.nation != actor.nation and rules.units[u.type].category == "military"
    ]
    if isinstance(actor, Unit) and rules.units[actor.type].torpedo:
        # 鱼雷只打舰船
        ships = [u for u in enemies if u.layer == "surface" and rules.units[u.type].domain == "sea"]
        return max(
            ships, key=lambda u: (strength(state, rules, u, defending=True), -u.id), default=None
        )
    if isinstance(actor, Unit) and rules.units[actor.type].anti_air:
        # 高射炮只打飞机
        air = [u for u in enemies if u.layer == "air"]
        return max(
            air, key=lambda u: (strength(state, rules, u, defending=True), -u.id), default=None
        )
    if isinstance(actor, Unit):
        ut = rules.units[actor.type]
        if ut.domain != "air":
            enemies = [u for u in enemies if u.layer == "surface"]
            if attack_kind(rules, actor) == "melee":
                enemies = [u for u in enemies if rules.units[u.type].domain == ut.domain]
    if enemies:
        return max(enemies, key=lambda u: (strength(state, rules, u, defending=True), -u.id))
    city = state.city_at(tile)
    if city is not None and city.nation != actor.nation:
        return city
    return None


def preview(state, rules, actor, tile) -> tuple[dict | None, str | None]:
    problem = lock_reason(state, rules, actor)
    if problem:
        return None, problem
    if tile == actor.tile or state.grid.distance(actor.tile, tile) > attack_range(rules, actor):
        return None, "超出攻击范围"
    kind = attack_kind(rules, actor)
    if kind == "ranged" and line_blocked(state, rules, actor, actor.tile, tile):
        return None, "山脉或树林遮挡射线"
    target = target_at(state, rules, actor, tile)
    if target is None:
        return None, "这里没有可攻击的敌方军事单位或城市"
    if isinstance(target, City) and target.hp <= 0 and kind != "melee":
        return None, "城防已摧毁，需要陆军近战占领"
    if kind == "melee" and isinstance(target, City) and rules.units[actor.type].domain != "land":
        return None, "舰船不能登陆攻城"
    if kind == "melee" and isinstance(target, Unit):
        # 近战要开进目标格交战：本方在那一格不能超过堆叠上限
        from engine.systems.stacking import has_room

        ut = rules.units[actor.type]
        if not has_room(state, rules, tile, ut.stack_class, actor.nation):
            return None, "目标格本方部队已满，不能再开进去交战"
    notes_a, notes_d = [], []
    a = strength(state, rules, actor, ranged=kind != "melee", notes=notes_a, against=target.nation)
    d = strength(state, rules, target, defending=True, notes=notes_d, against=actor.nation)
    flak, flak_from = 0, None
    if kind == "air":
        # 空袭进了敌方高射炮的覆盖范围：先挨高射火力，带伤投弹
        guns = air_defense(state, rules, actor, tile)
        if guns:
            # 覆盖这一格的高射炮都开火（最强的那门记功），总共最多打掉 FLAK_CAP 点
            gun = guns[0]
            own = strength(state, rules, actor, against=gun.nation)
            total = 0
            for g in guns[:FLAK_GUNS]:
                power = strength(state, rules, g, ranged=True, against=actor.nation)
                total += max(6, min(40, FLAK_FACTOR * power // own))
            flak = min(actor.hp, total, FLAK_CAP)
            flak_from = ref("unit", gun.id)
            a = strength(
                state, rules, actor, ranged=True, hp=actor.hp - flak, against=target.nation
            )
    tactical = attack_modifiers(state, rules, actor, tile, kind)
    a = _apply(a, tactical)
    damage = max(8, min(65, 30 * a // d)) if actor.hp > flak else 0
    # 兵种对部队 / 城市的伤害倍率（炮兵：打部队弱、削城防强），只改伤害，不改战斗力和反击
    scale = _damage_scale(rules, actor, target)
    if scale != 100:
        damage = damage * scale // 100
        tactical = [
            *tactical,
            ("攻城火力" if isinstance(target, City) else "对部队火力", scale - 100),
        ]
    # 炮击整格：其余敌军（和城市）各按自己的防御力受伤
    area = []
    if (
        kind != "melee"
        and isinstance(actor, Unit)
        and "area_damage" in rules.units[actor.type].tags
    ):
        others = [
            u
            for u in state.units_at(tile, target.layer if isinstance(target, Unit) else "surface")
            if u.nation != actor.nation
            and rules.units[u.type].category == "military"
            and u is not target
        ]
        city = state.city_at(tile)
        if city is not None and city.nation != actor.nation and city is not target:
            others.append(city)
        for o in sorted(others, key=lambda o: (isinstance(o, City), o.id)):
            hit = max(
                8,
                min(65, 30 * a // strength(state, rules, o, defending=True, against=actor.nation)),
            )
            hit = hit * _damage_scale(rules, actor, o) // 100
            area.append(
                {
                    "target": ref("city" if isinstance(o, City) else "unit", o.id),
                    "damage": min(o.hp, hit),
                }
            )
    retaliation = max(6, min(60, 25 * d // a)) if kind == "melee" else 0
    if target.hp <= 0:
        retaliation = 0
    if kind == "air" and isinstance(target, Unit):
        dt = rules.units[target.type]
        retaliation = (
            max(5, min(35, 15 * d // a)) if dt.layer == "air" or "air_superiority" in dt.tags else 0
        )
    if not damage:
        retaliation, area = 0, []
    return {
        "tile": tile,
        "target": ref("city" if isinstance(target, City) else "unit", target.id),
        "kind": kind,
        "damage": min(target.hp, damage),
        "retaliation": min(actor.hp - flak, retaliation),
        "flak": flak,  # 来袭飞机先挨的高射火力（击落时 damage 为 0）
        # 炮击压制：这次攻击会让打中的部队失去驻守加成
        "suppress": isinstance(actor, Unit)
        and "suppress" in rules.units[actor.type].tags
        and kind != "melee",
        "flak_from": flak_from,
        "attacker_strength": a,
        "defender_strength": d,
        "modifiers": [
            {"side": "attacker", "label": label, "value": value}
            for label, value in [*notes_a, *tactical]
        ]
        + [{"side": "defender", "label": label, "value": value} for label, value in notes_d],
        "area": area,
    }, None


def air_defense(state, rules, actor, tile) -> list[Unit]:
    """覆盖 tile 的敌方高射炮（相对 actor），火力最强的在前。"""
    guns = [
        u
        for t in state.grid.within(tile, max_anti_air_range(rules))
        for u in state.units_at(int(t), "surface")
        if u.nation != actor.nation
        and rules.units[u.type].anti_air
        and state.grid.distance(u.tile, tile) <= rules.units[u.type].range
    ]
    return sorted(guns, key=lambda u: (-strength(state, rules, u, ranged=True), u.id))


def max_anti_air_range(rules) -> int:
    return max((u.range for u in rules.units.values() if u.anti_air), default=0)


def tactics(state, rules, actor) -> dict:
    reach = attack_range(rules, actor)
    tiles = state.grid.within(actor.tile, reach)
    tiles = sorted(t for t in tiles if t != actor.tile)
    reason = lock_reason(state, rules, actor)
    targets, obstructed = [], []
    for t in tiles:
        info, problem = preview(state, rules, actor, t)
        if info:
            targets.append(info)
        elif problem == "山脉或树林遮挡射线":
            obstructed.append(t)
    return {
        "attack_range": reach,
        "attack_tiles": tiles,
        "targets": targets,
        "attack_blocked": reason,
        "obstructed": obstructed,
    }


def _capture(ctx, city, unit):
    s, rules = ctx.state, ctx.rules
    previous = city.nation
    city.nation, city.hp, city.capital = unit.nation, 100, False
    city.stability = min(city.stability, rules.stability.captured)  # 新占领地区不稳
    city.queue, city.production, city.attacked_turn = [], 0, s.turn
    city.managed = True
    city.management = {"status": "pending", "reason": "城市已易主，按新国家方针重新安排"}
    city.production_reason = ""
    s._network_cache = None
    s._settlement_usage = None
    s._flow_cache.clear()
    s._edge_cache.clear()
    s._worked_cache.clear()
    s._yield_cache.clear()
    s.grid.owner[s.grid.city_of == city.id] = unit.nation
    # Aircraft on a captured base and noncombatants surrender with the city.
    for parked in list(s.units_at(city.tile)):
        if parked.nation == previous:
            ctx.log.add(
                "unit_destroyed",
                unit=ref("unit", parked.id),
                tile=parked.tile,
                unit_type=parked.type,
                nation=ref("nation", parked.nation),
                reason="投降",
            )
            _lost_aboard(ctx, parked)
            s.remove_unit(parked.id)
    if s.nations[previous].capital == city.id:
        remaining = s.nation_cities(previous)
        s.nations[previous].capital = remaining[0].id if remaining else None
        if remaining:
            remaining[0].capital = True
    if s.nations[unit.nation].capital is None:
        s.nations[unit.nation].capital, city.capital = city.id, True
    ctx.log.add(
        "city_captured",
        city=ref("city", city.id),
        tile=city.tile,
        nation=ref("nation", unit.nation),
        previous=ref("nation", previous),
    )


def _lost_aboard(ctx, ship) -> None:
    """船 / 航母被击沉或投降前：船上的部队、舰载机记为随船损失（state.remove_unit 会一起移除）。"""
    for rider in ctx.state.aboard(ship):
        ctx.log.add(
            "unit_destroyed",
            unit=ref("unit", rider.id),
            tile=rider.tile,
            unit_type=rider.type,
            nation=ref("nation", rider.nation),
            reason="随船沉没",
        )


def _destroy(ctx, unit, tile, reason=None):
    extra = {"reason": reason} if reason else {}
    ctx.log.add(
        "unit_destroyed",
        unit=ref("unit", unit.id),
        tile=tile,
        unit_type=unit.type,
        nation=ref("nation", unit.nation),
        **extra,
    )
    _lost_aboard(ctx, unit)
    ctx.state.remove_unit(unit.id)


def resolve(ctx: StepContext) -> None:
    s, rules = ctx.state, ctx.rules
    # 整个战斗阶段用同一份包围状态（按移动后的位置）
    s._encircled = None
    encirclement.status(s, rules)
    s._encircled_lock = True
    try:
        _resolve(ctx)
    finally:
        s._encircled_lock = False
        s._encircled = None


def _military(rules, u):
    return rules.units[u.type].category == "military"


def _contested(state, rules):
    """有两个以上国家的地表军事单位同处的格子（交战格），按格子编号排序。"""
    out = []
    for (tile, layer), ids in sorted(state._at.items()):
        if layer != "surface":
            continue
        nations = {state.units[i].nation for i in ids if _military(rules, state.units[i])}
        if len(nations) > 1:
            out.append(tile)
    return out


def _skirmish(ctx: StepContext) -> None:
    """交战格里的部队自动接着打：每支（按国家、ID 顺序）打一次格子里最强的敌军，双方都受伤。"""
    s, rules = ctx.state, ctx.rules
    for tile in _contested(s, rules):
        for uid in sorted(
            (u.id for u in s.units_at(tile, "surface") if _military(rules, u)),
            key=lambda i: (s.units[i].nation, i),
        ):
            u = s.units.get(uid)
            if u is None or u.attacked_turn == s.turn or u.formation is not None:
                continue
            foes = [
                v
                for v in s.units_at(tile, "surface")
                if v.nation != u.nation and _military(rules, v)
            ]
            if not foes:
                break
            target = max(foes, key=lambda v: (strength(s, rules, v, defending=True), -v.id))
            a = _apply(
                strength(s, rules, u, against=target.nation),
                attack_modifiers(s, rules, u, tile, "melee"),
            )
            d = strength(s, rules, target, defending=True, against=u.nation)
            damage = min(target.hp, max(8, min(65, 30 * a // d)))
            retaliation = min(u.hp, max(6, min(60, 25 * d // a)))
            target.hp -= damage
            u.hp -= retaliation
            u.attacked_turn = u.damaged_turn = target.damaged_turn = s.turn
            u.fortified = False
            ctx.log.add(
                "combat",
                attacker=ref("unit", u.id),
                defender=ref("unit", target.id),
                source=tile,
                tile=tile,
                kind="melee",
                damage=damage,
                retaliation=retaliation,
                attacker_strength=a,
                defender_strength=d,
                area=[],
                attacker_hp=u.hp,
                defender_hp=target.hp,
                reason="同格交战",
            )
            gain(ctx, u, 4 + (8 if target.hp <= 0 else 0), "unit")
            gain(ctx, target, 3, "unit")
            if target.hp <= 0:
                _destroy(ctx, target, tile)
            if u.hp <= 0:
                _destroy(ctx, u, tile)
        _settle(ctx, tile)


def _settle(ctx: StepContext, tile: int) -> None:
    """交战格打完：只剩一个国家的军队时，城防已空的敌国城市被占领，留在格里的敌方平民投降。"""
    s, rules = ctx.state, ctx.rules
    soldiers = [u for u in s.units_at(tile, "surface") if _military(rules, u)]
    nations = {u.nation for u in soldiers}
    if len(nations) != 1:
        return
    winner = min(soldiers, key=lambda u: u.id)
    city = s.city_at(tile)
    if city is not None and city.nation != winner.nation and city.hp <= 0:
        _capture(ctx, city, winner)
    for stranded in list(s.units_at(tile, "surface")):
        if stranded.nation != winner.nation and not _military(rules, stranded):
            _destroy(ctx, stranded, tile, "投降")


def _resolve(ctx: StepContext) -> None:
    s, rules = ctx.state, ctx.rules
    _skirmish(ctx)
    for order in ctx.attacks:
        is_city = isinstance(order, CityAttack)
        collection, key = (s.cities, "city") if is_city else (s.units, "unit")
        automatic_order = (order.type, getattr(order, key), order.to) in ctx.automatic_attacks
        actor = collection.get(parse_ref(getattr(order, key), key))
        if actor is None:
            if automatic_order:
                continue
            ctx.log.reject(order.model_dump(mode="json"), "not_found", "攻击者已不存在")
            continue
        if actor.nation != parse_ref(order.issuer, "nation"):
            if automatic_order:
                continue
            ctx.log.reject(order.model_dump(mode="json"), "not_owner", "攻击者已不属于发令国")
            continue
        tile = s.grid.idx(*order.to)
        info, problem = preview(s, rules, actor, tile)
        if problem is None and not is_city and actor.id in ctx.moving:
            problem = "本回合在行军，不能交战"
        if problem:
            if automatic_order:
                ctx.log.add(
                    "directive_report",
                    nation=order.issuer,
                    goal="unit_ai",
                    status="blocked",
                    reason=problem,
                )
                continue
            ctx.log.reject(order.model_dump(mode="json"), "cannot_attack", problem)
            continue
        target = target_at(s, rules, actor, tile)
        start = actor.tile
        if info["flak"]:
            gun = s.units[parse_ref(info["flak_from"], "unit")]
            actor.hp -= info["flak"]
            actor.damaged_turn = s.turn
            ctx.log.add(
                "anti_air_fire",
                unit=info["flak_from"],
                target=ref("unit", actor.id),
                tile=start,
                damage=info["flak"],
                target_hp=actor.hp,
            )
            gain(ctx, gun, 3 + (6 if actor.hp <= 0 else 0), "unit")
            if actor.hp <= 0:
                actor.attacked_turn = s.turn
                _destroy(ctx, actor, start, "被防空火力击落")
                continue
        target.hp = max(0, target.hp - info["damage"])
        splashed = []
        for hit in info["area"]:
            kind_, _, oid = hit["target"].partition(":")
            other = s.cities.get(int(oid)) if kind_ == "city" else s.units.get(int(oid))
            if other is None:
                continue
            other.hp = max(0, other.hp - hit["damage"])
            other.damaged_turn = s.turn
            splashed.append(other)
        # 炮击压制：打中的部队（炮击波及的整格都算）失去驻守加成
        suppressed = []
        if info.get("suppress"):
            for victim in [target, *splashed]:
                if isinstance(victim, Unit) and victim.fortified:
                    victim.fortified = False
                    suppressed.append(ref("unit", victim.id))
        actor.hp = max(0, actor.hp - info["retaliation"])
        actor.attacked_turn = actor.damaged_turn = target.damaged_turn = s.turn
        if isinstance(actor, Unit):
            actor.moves_left, actor.road_credit, actor.destination, actor.fortified = (
                0,
                0,
                None,
                False,
            )
        ctx.log.add(
            "combat",
            attacker=ref(key, actor.id),
            defender=info["target"],
            source=start,
            **info,
            suppressed=suppressed,
            attacker_hp=actor.hp,
            defender_hp=target.hp,
        )
        gain(ctx, actor, 4 + (8 if target.hp <= 0 else 0), key)
        gain(ctx, target, 3, "city" if isinstance(target, City) else "unit")
        for other in splashed:
            if isinstance(other, Unit) and other.hp <= 0 and other.id in s.units:
                _destroy(ctx, other, other.tile, "炮击")
        dead_target = target.hp <= 0
        if dead_target and isinstance(target, Unit):
            ctx.log.add(
                "unit_destroyed",
                unit=ref("unit", target.id),
                tile=target.tile,
                unit_type=target.type,
                nation=ref("nation", target.nation),
            )
            _lost_aboard(ctx, target)
            s.remove_unit(target.id)
        if actor.hp <= 0 and isinstance(actor, Unit):
            ctx.log.add(
                "unit_destroyed",
                unit=ref("unit", actor.id),
                tile=start,
                unit_type=actor.type,
                nation=ref("nation", actor.nation),
            )
            _lost_aboard(ctx, actor)
            s.remove_unit(actor.id)
        elif info["kind"] == "melee" and isinstance(target, Unit):
            # 近战 = 开进目标格交战：不论胜负都留在那一格（敌军还在就成了交战格）
            if s.units.get(actor.id) is actor and actor.tile != tile:
                s.move_unit(actor, tile)
                ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])
            _settle(ctx, tile)
        elif info["kind"] == "melee" and dead_target:
            enemies = [
                u
                for u in s.units_at(tile, actor.layer)
                if u.nation != actor.nation and rules.units[u.type].category == "military"
            ]
            city = s.city_at(tile)
            if not enemies and (city is None or city.nation == actor.nation or city.hp <= 0):
                if city is not None and city.nation != actor.nation:
                    _capture(ctx, city, actor)
                elif city is None:
                    for stranded in list(s.units_at(tile, actor.layer)):
                        if stranded.nation != actor.nation:
                            ctx.log.add(
                                "unit_destroyed",
                                unit=ref("unit", stranded.id),
                                tile=tile,
                                unit_type=stranded.type,
                                nation=ref("nation", stranded.nation),
                                reason="投降",
                            )
                            s.remove_unit(stranded.id)
                s.move_unit(actor, tile)
                ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])
