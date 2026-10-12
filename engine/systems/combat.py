"""确定性战斗。射程预览和结算用同一套检查。

- 一个格子最多放 3 个地表单位（stacking.total；城里按 cities.max_units），只要不在行军、不在编队里，
  每个单位都可以各自攻击。
- 敌我单位不能同格：近战只打相邻的格子，攻击方留在自己的格子里；目标格的敌方军事单位打光了才开进去
  （剩下的敌方平民被消灭，_advance）。城防还在的敌国城市只能从相邻格强攻城墙；城防打空后近战开进去占领。
  （旧存档里如果还有敌我同格的交战格，每回合在格里接着打完，_skirmish。）
- 带 area_damage 标签的远程单位（炮兵、战列舰）炮击整格：目标格里的每个敌方军事单位
  （和敌国城市）都按各自的防御力受到伤害。
- 被包围的单位、城市战斗力打折（见 engine/systems/encirclement.py）。
- 下了海的陆军（engine/systems/naval.py）不能攻击；挨打时按舰船算（鱼雷打得着，陆军近战打不着），
  防御力只剩 embark.defense%、受到的伤害 ×embark.damage_taken%，不反击。同格有本方战舰时敌人先打战舰
  （护航，_screened）；炮击、地毯式轰炸波及整格，照样打得到它。
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

import numpy as np

from engine.actions import CityAttack
from engine.ids import parse_ref, ref
from engine.state import MOVE_STEPS, City, Unit
from engine.systems import encirclement
from engine.systems.cities import city_stat, nation_supply
from engine.systems.context import StepContext
from engine.systems.naval import afloat
from engine.systems.progression import gain, level, unit_strength

FUEL_SHORTAGE = 30  # 没有石油：坦克、飞机（fuel 单位）战斗力 −30%
DECAY_STRENGTH = 25  # 大一统之后的腐败：军纪废弛，战斗力最多 −25%（腐败超过起点后每 4 点 −1%）
FLAK_FACTOR = 30  # 一门高射炮对来袭飞机的火力：FLAK_FACTOR × 防空火力 ÷ 飞机战斗力（6–40）
FLAK_CAP = 70  # 飞一段航线最多被防空火力打掉这么多生命
FIGHTER_VS_BOMBER = 160  # 战斗机打轰炸机的伤害（%）


# ---------------------------------------------------------------- 城防（城市和守城部队共用一个血条）


def city_max_hp(rules, city) -> int:
    """城市本体的城防上限：基础值 + 城防建筑（城防工事、首都府）。"""
    return rules.cities.defense_base + city_stat(rules, city, "defense", 0)


def garrison(state, rules, city) -> list:
    """守城的近战部队：城里本国的陆上近战军事单位。它们的生命按一半计入城防值，不能出城攻击。"""
    return [
        u
        for u in state.units_at(city.tile, "surface")
        if u.nation == city.nation
        and rules.units[u.type].category == "military"
        and rules.units[u.type].domain == "land"
        and attack_kind(rules, u) == "melee"
    ]


def city_defense(state, rules, city) -> int:
    """城防值：城市本体 + 守城近战部队的生命（各按一半）。打到 0 才能占领。"""
    return max(0, city.hp) + sum(u.hp // 2 for u in garrison(state, rules, city))


def damage_city(ctx, city, amount: int, by=None) -> None:
    """城防挨打：先打城市本体，打空了再打守城部队（城防值 1 点 = 部队 2 点生命，生命多的先挨）。
    by：攻击方（打死守城部队记进它的战绩）。"""
    s, rules = ctx.state, ctx.rules
    take = min(city.hp, amount)
    city.hp -= take
    left = amount - take
    for u in sorted(garrison(s, rules, city), key=lambda u: (-u.hp, u.id)):
        if left <= 0:
            break
        hit = min(u.hp, left * 2)
        u.hp -= hit
        u.damaged_turn = s.turn
        left -= (hit + 1) // 2
        if u.hp <= 0:
            _destroy(ctx, u, u.tile, "守城阵亡", by=by)


def _war(state, a, b) -> bool:
    """a、b 两国在交战（外交，engine/systems/diplomacy.py）：只能打交战国。"""
    from engine.systems.diplomacy import at_war

    return at_war(state, a, b)


def attack_range(rules, actor: Unit | City, grid=None) -> int:
    """射程。飞机是作战半径（传 grid 时按开局设置 air_range 缩放；从起飞机场量）。"""
    if isinstance(actor, City):
        return 2
    ut = rules.units[actor.type]
    if ut.domain == "air" and grid is not None:
        from engine.systems.movement import air_radius

        return air_radius(grid, ut)
    if not (ut.ranged_strength or ut.domain == "air" or ut.anti_air or ut.torpedo):
        return 1
    if grid is not None and not ut.anti_air:
        # 开局设置：海军射程（舰炮、鱼雷）、炮兵射程（陆上远程）各自的百分比
        key = "naval_range" if ut.domain == "sea" else "artillery_range"
        pct = grid.tuning.get(key, 100)
        if pct != 100:
            return max(1, round(ut.range * pct / 100))
    return ut.range


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
    if ut.domain == "land" and afloat(state, rules, actor):
        return "在海上没有攻击力，先上岸"
    if "carrier" in ut.tags and not ut.ranged_strength:
        return "航空母舰没有攻击手段，靠舰载机作战"
    if "unarmed" in ut.tags:
        return f"{ut.name}没有攻击手段"
    if actor.formation is not None:
        return "编队不能攻击，请先解散编队"
    home = state.city_at(actor.tile)
    if (
        home is not None
        and home.nation == actor.nation
        and ut.domain == "land"
        and attack_kind(rules, actor) == "melee"
    ):
        return "守城的近战部队不能出击（城里只有远程部队能攻击；要进攻先出城）"
    if actor.destination is not None:
        return "行军途中（还没到目的地）不能交战，先停下或走到位"
    if ut.domain == "air":
        # 出击三回合：起飞飞到目标上空 → 攻击 → 返航降落（engine/systems/air.py）
        from engine.systems.movement import is_air_base

        sortie = actor.sortie
        if sortie is not None:
            if sortie["stage"] == "patrol":
                from engine.systems.air import fighter_ready

                if not fighter_ready(state, sortie):
                    return "正在返航降落"
                return None  # 战斗机在空中：还能再攻击一次
            if sortie["stage"] == "outbound" and not sortie.get("strikes"):
                return None  # 轰炸机刚飞到目标上空：这回合投弹
            return "出击任务中（返航），降落前不能下别的命令"
        if actor.moves_left < ut.moves:
            return "本回合已经转场，下一回合再出击"
        if not is_air_base(state, rules, actor.nation, actor.tile):
            return "不在本国机场或航母上，不能起飞出击"
    # 文明 6 式：可以先移动再攻击，但要留着攻击花的移动力（tactics.attack_cost），攻击用掉剩下的全部移动力
    from engine.systems.movement import attack_steps

    left = actor.moves_left * MOVE_STEPS + actor.road_credit
    if left <= 0:
        return "移动力已用完，下一回合恢复"
    if left < attack_steps(rules, actor):
        return f"剩下的移动力不够攻击（攻击要 {rules.tactics.attack_cost} 点移动力）"
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
    """从 source 打 tile 的射线有没有被山脉 / 树林挡住。间接射击（indirect 标签，炮兵）不需要视线；
    舰炮从海上打岸上（弹道高、有观测），也不受岸边的山林遮挡。"""
    if isinstance(actor, Unit) and (
        "indirect" in rules.units[actor.type].tags or rules.units[actor.type].domain == "sea"
    ):
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
    state,
    rules,
    actor,
    *,
    defending=False,
    ranged=False,
    hp=None,
    notes=None,
    against=None,
    flak=False,
    deck_gun=False,
) -> int:
    """战斗力。ranged：远程 / 防空火力；hp：按这个生命值算（预估带伤后的战斗力）；
    notes：传入列表时把各项修正（说明, 百分比）记进去，给攻击预览显示；
    against：对手国家，算兵力悬殊的士气。"""
    ring = encirclement.status(state, rules)
    if isinstance(actor, City):
        # 城市的战斗力随城防值（城市本体 + 守城部队）升高
        value = 8 + city_defense(state, rules, actor) // 5
        bonus = 8 * (level(actor.xp, "city") - 1)
        _note(notes, f"城市 Lv.{level(actor.xp, 'city')}", bonus)
        value = max(1, value * (100 + bonus) // 100)
        if actor.id in ring.cities and not defending:
            _note(notes, "被包围", -rules.encirclement.city_penalty)
            value = value * (100 - rules.encirclement.city_penalty) // 100
        return max(1, _morale(state, rules, actor, against, value, notes))
    ut = rules.units[actor.type]
    value = ut.strength
    if flak:
        value = ut.anti_air or ut.flak or 1  # 防空火力（高射炮、舰艇自带防空）
    elif deck_gun:
        value = ut.deck_gun or ut.strength  # 驱逐舰的甲板炮（打岸上目标）
    elif ranged:
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
        supply = nation_supply(state, rules, actor.nation)
        if ut.fuel and "oil" not in supply:
            # 坦克、飞机靠燃料：本国没有石油供应，战斗力 −FUEL_SHORTAGE%
            _note(notes, "缺石油燃料", -FUEL_SHORTAGE)
            value = value * (100 - FUEL_SHORTAGE) // 100
        # 大一统之后的腐败：军纪废弛，战斗力最多 −DECAY_STRENGTH%（分裂出去的新政权没有这个减益）
        n = state.nations.get(actor.nation)
        cfg = rules.stability
        if n is not None and cfg.decay and n.decay > cfg.decay_start:
            pct = min(DECAY_STRENGTH, (n.decay - cfg.decay_start) // 4)
            if pct:
                _note(notes, "军纪废弛", -pct)
                value = value * (100 - pct) // 100
        # 科技加成（techs.yaml 的 strength）
        from engine.systems.techs import strength_bonuses

        for label, pct in strength_bonuses(state, rules, actor, ut):
            _note(notes, label, pct)
            value = value * (100 + pct) // 100
        # 战略资源加成：钨芯弹、雷达……（规则在 resources.yaml 的 combat）
        for rid in sorted(supply):
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
    if defending and ut.domain == "land" and afloat(state, rules, actor):
        # 下了海的陆军：没有地形、驻守加成，防御力只剩 embark.defense%
        _note(notes, "在海上", rules.embark.defense - 100)
        value = value * rules.embark.defense // 100
    elif defending:
        _note(notes, f"地形：{terrain.name}", terrain.defense)
        _note(notes, "驻守", 25 if actor.fortified else 0)
        site = 0
        fac = rules.facility_at(int(state.grid.facility[actor.tile]))
        if (
            fac is not None
            and int(state.grid.owner[actor.tile]) == actor.nation
            and actor.tile not in state.damaged
        ):
            # 进驻本国的军事基地等设施：防守加成（设施的 tile defense 修正器）
            site = sum(
                m.value
                for m in fac.modifiers
                if m.target == "tile" and m.stat == "defense" and m.op == "add"
            )
            _note(notes, fac.name, site)
        value = (
            value * max(25, 100 + terrain.defense + site + (25 if actor.fortified else 0)) // 100
        )
    if actor.id in ring.units and not defending:
        # 被包围：攻击力只剩一半（挨打时受到的伤害翻倍，见 preview）
        _note(notes, "被包围", -rules.encirclement.penalty)
        value = value * (100 - rules.encirclement.penalty) // 100
    return max(1, _morale(state, rules, actor, against, value, notes))


def attack_modifiers(state, rules, actor, tile, kind, target=None) -> list[tuple[str, int]]:
    """进攻方的战术修正 [(说明, 百分比)]：出击地形；打部队时看朝向（侧击 +30%、背击 +80%）；
    隔着河进攻 −50%。飞机不受这些影响。"""
    if not isinstance(actor, Unit) or actor.layer != "surface" or kind in ("air", "anti_air"):
        return []
    g, cfg = state.grid, rules.tactics
    out = []
    terrain = rules.terrain_at(int(g.terrain[actor.tile]))
    if terrain.attack:
        out.append((f"从{terrain.name}出击", terrain.attack))
    if isinstance(target, Unit) and target.layer == "surface" and actor.tile != target.tile:
        from engine.systems.zoc import bearing, facing_modifier

        mod = facing_modifier(rules, target.facing, bearing(g, target.tile, actor.tile))
        if mod is not None:
            out.append(mod)
    if actor.tile != tile and tile in g.neighbors_of(actor.tile):
        from engine.systems.roads import crosses_river

        if crosses_river(g, actor.tile, tile):
            out.append(("跨河进攻", -cfg.river))
    return out


def domain_of(state, rules, unit) -> str:
    """打起来按什么军种算：下了海的陆军按舰船算（鱼雷打得着，陆军近战打不着）。"""
    ut = rules.units[unit.type]
    if ut.domain == "land" and afloat(state, rules, unit):
        return "sea"
    return ut.domain


def _screened(state, rules, units):
    """海上护航：同格有本方战舰时，敌人打不着下了海的陆军（先打战舰）。"""
    guards = [u for u in units if not afloat(state, rules, u)]
    return guards or units


def _damage_scale(rules, actor, target, state=None) -> int:
    """伤害倍率（%）：兵种对部队 / 对城市的倍率 × 军种克制（规则集 domain_damage：舰炮打岸上部队强、
    陆军打舰船弱、飞机打舰船强……）。城市的防御射击按陆军算；下了海的陆军按舰船算（给了 state 时）。"""
    if isinstance(target, City):
        return rules.units[actor.type].damage_vs_city if isinstance(actor, Unit) else 100
    attacker = rules.units[actor.type].domain if isinstance(actor, Unit) else "land"
    victim = rules.units[target.type].domain if state is None else domain_of(state, rules, target)
    vs = rules.domain_damage.get(attacker, {}).get(victim, 100)
    base = rules.units[actor.type].damage_vs_units if isinstance(actor, Unit) else 100
    if (
        isinstance(actor, Unit)
        and "air_superiority" in rules.units[actor.type].tags
        and rules.units[target.type].domain == "air"
        and "air_superiority" not in rules.units[target.type].tags
    ):
        base = base * FIGHTER_VS_BOMBER // 100  # 战斗机打轰炸机
    return base * vs // 100


def _apply(value, mods) -> int:
    return max(1, value * (100 + sum(v for _, v in mods)) // 100) if mods else value


def bombable(state, rules, actor, tile) -> bool:
    """轰炸机能不能炸这一格的设施：外国领土上、还没被炸坏的城外设施。"""
    if not isinstance(actor, Unit):
        return False
    ut = rules.units[actor.type]
    if ut.domain != "air" or "air_superiority" in ut.tags or not ut.ranged_strength:
        return False
    g = state.grid
    owner = int(g.owner[tile])
    return (
        bool(g.facility[tile])
        and owner >= 0
        and _war(state, owner, actor.nation)
        and tile not in state.damaged
        and state.city_at(tile) is None
    )


def bomb_facility(ctx, actor, tile) -> None:
    """炸坏 tile 上的设施（轰炸机投弹命中设施格）。"""
    s, rules = ctx.state, ctx.rules
    b = rules.facility_at(int(s.grid.facility[tile]))
    s.damaged[tile] = s.turn
    s._worked_cache.clear()
    s._yield_cache.clear()
    ctx.log.add(
        "facility_bombed",
        unit=ref("unit", actor.id),
        nation=ref("nation", actor.nation),
        victim=ref("nation", int(s.grid.owner[tile])),
        facility=b.id,
        tile=tile,
    )


def target_at(state, rules, actor, tile):
    city = state.city_at(tile)
    if city is not None and _war(state, city.nation, actor.nation):
        # 城市和守城部队共用一个血条：打城市格只打城防值（高射炮照样打城上空的飞机，
        # 鱼雷打不了城市）
        ut = rules.units[actor.type] if isinstance(actor, Unit) else None
        if ut is None or not (ut.anti_air or (ut.torpedo and not ut.deck_gun)):
            return city
    enemies = [
        u
        for u in state.units_at(tile)
        if _war(state, u.nation, actor.nation) and rules.units[u.type].category == "military"
    ]
    if isinstance(actor, Unit) and rules.units[actor.type].torpedo:
        # 鱼雷只打舰船（下了海的陆军也算，有护航先打护航的战舰）
        ships = _screened(
            state,
            rules,
            [u for u in enemies if u.layer == "surface" and domain_of(state, rules, u) == "sea"],
        )
        if ships or not rules.units[actor.type].deck_gun:
            return max(
                ships,
                key=lambda u: (strength(state, rules, u, defending=True), -u.id),
                default=None,
            )
        # 没有敌舰：用甲板炮打岸上的部队 / 城市（下面按远程的规则挑）
    if isinstance(actor, Unit) and "air_superiority" in rules.units[actor.type].tags:
        # 战斗机：格子上空有敌机就打敌机（空战 / 拦截），没有才扫射地面
        planes = [u for u in enemies if u.layer == "air"]
        if planes:
            return max(
                planes,
                key=lambda u: (
                    "air_superiority" not in rules.units[u.type].tags,  # 先打轰炸机
                    strength(state, rules, u, defending=True),
                    -u.id,
                ),
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
                enemies = [u for u in enemies if domain_of(state, rules, u) == ut.domain]
    enemies = _screened(state, rules, enemies)
    if enemies:
        return max(enemies, key=lambda u: (strength(state, rules, u, defending=True), -u.id))
    city = state.city_at(tile)
    if city is not None and _war(state, city.nation, actor.nation):
        return city
    if isinstance(actor, Unit):
        # 格子里只剩敌方平民（开拓者、工兵、将领、海军将领）：所有单位都能攻击，打中就消灭（不俘虏）；
        # 近战只能打同一军种的（陆军打陆上的、舰船打海上的）
        ut = rules.units[actor.type]
        civilians = [
            u
            for u in state.units_at(tile, "surface")
            if _war(state, u.nation, actor.nation)
            and (attack_kind(rules, actor) != "melee" or domain_of(state, rules, u) == ut.domain)
        ]
        if civilians:
            return min(civilians, key=lambda u: u.id)
    return None


def preview(state, rules, actor, tile) -> tuple[dict | None, str | None]:
    problem = lock_reason(state, rules, actor)
    if problem:
        return None, problem
    overhead = isinstance(actor, Unit) and actor.sortie is not None  # 出击的飞机在目标正上空攻击
    # 飞机的作战半径从起飞的机场量（战斗机在空中也不能飞出机场的作战半径）
    origin = actor.sortie["base"] if overhead else actor.tile
    if overhead and actor.sortie["stage"] == "outbound" and tile != actor.tile:
        return None, "轰炸机只能轰炸飞到的这一格"
    if (tile == actor.tile and not overhead) or state.grid.distance(origin, tile) > attack_range(
        rules, actor, state.grid
    ):
        return None, "超出攻击范围" if not overhead else "超出机场的作战半径"
    kind = attack_kind(rules, actor)
    if kind == "ranged" and line_blocked(state, rules, actor, actor.tile, tile):
        return None, "山脉或树林遮挡射线"
    target = target_at(state, rules, actor, tile)
    if target is None and kind == "air" and bombable(state, rules, actor, tile):
        # 轰炸机炸设施：炸坏后不产出、机场不能起降，要对方城市花工期修
        from engine.systems.air import flight_flak

        shots = flight_flak(state, rules, actor, actor.tile, tile) if tile != actor.tile else []
        flak = sum(d for _, d, _ in shots)
        b = rules.facility_at(int(state.grid.facility[tile]))
        return {
            "tile": tile,
            "target": None,
            "kind": "air",
            "facility_strike": True,
            "facility": b.id,
            "damage": 0,
            "retaliation": 0,
            "flak": flak,
            "suppress": False,
            "flak_from": ref("unit", shots[0][0].id) if shots else None,
            "attacker_strength": strength(state, rules, actor, ranged=True),
            "defender_strength": 0,
            "modifiers": [],
            "area": [],
        }, None
    if target is None:
        return None, "这里没有可攻击的敌方军事单位或城市"
    if isinstance(target, Unit) and rules.units[target.type].category != "military":
        # 平民（含将领）没有战斗力：打中就消灭
        return {
            "tile": tile,
            "target": ref("unit", target.id),
            "kind": kind,
            "kill": True,
            "damage": target.hp,
            "retaliation": 0,
            "flak": 0,
            "suppress": False,
            "flak_from": None,
            "attacker_strength": strength(state, rules, actor),
            "defender_strength": 0,
            "modifiers": [],
            "area": [],
        }, None
    if (
        isinstance(target, City)
        and city_defense(state, rules, target) <= 0
        and isinstance(actor, Unit)
        and rules.units[actor.type].domain == "sea"
        and not {"unarmed", "carrier"} & set(rules.units[actor.type].tags)
    ):
        # 舰队占领：城防打空、城里没有守军的沿海城市，贴在城边的战舰直接开进港口占领
        if state.grid.distance(actor.tile, tile) > 1:
            return None, "城防已摧毁：战舰要贴到城边（相邻）才能开进港口占领"
        return {
            "tile": tile,
            "target": ref("city", target.id),
            "kind": "capture",
            "seize_city": True,
            "capture": True,
            "damage": 0,
            "retaliation": 0,
            "flak": 0,
            "suppress": False,
            "flak_from": None,
            "attacker_strength": strength(state, rules, actor),
            "defender_strength": 0,
            "modifiers": [],
            "area": [],
        }, None
    if kind == "melee" and isinstance(target, City) and rules.units[actor.type].domain != "land":
        return None, "舰船不能登陆攻城"
    if isinstance(target, City) and city_defense(state, rules, target) <= 0:
        if kind != "melee":
            return None, "城防已打空，需要陆军近战（或贴城的战舰）占领"
        from engine.systems.stacking import has_room

        if not has_room(state, rules, tile, rules.units[actor.type].stack_class, actor.nation):
            return None, "那一格本方已满，开不进去"
        return {
            "tile": tile,
            "target": ref("city", target.id),
            "kind": "melee",
            "storm_city": True,  # 城防已空：近战开进去占领，城里剩下的敌方单位全部消灭
            "capture": True,
            "damage": 0,
            "retaliation": 0,
            "flak": 0,
            "suppress": False,
            "flak_from": None,
            "attacker_strength": strength(state, rules, actor),
            "defender_strength": 0,
            "modifiers": [],
            "area": [],
        }, None
    gun = kind == "torpedo" and not (
        isinstance(target, Unit) and domain_of(state, rules, target) == "sea"
    )
    if gun:
        kind = "ranged"  # 驱逐舰没有敌舰可打：甲板炮打岸上目标
    notes_a, notes_d = [], []
    a = strength(
        state,
        rules,
        actor,
        ranged=kind != "melee",
        deck_gun=gun,
        notes=notes_a,
        against=target.nation,
    )
    d = strength(state, rules, target, defending=True, notes=notes_d, against=actor.nation)
    flak, flak_from = 0, None
    if kind == "air" and tile != actor.tile:
        # 飞过去的航线进 / 出敌方防空圈：先挨防空火力，带伤攻击（engine/systems/air.py）
        from engine.systems.air import flight_flak

        shots = flight_flak(state, rules, actor, actor.tile, tile)
        if shots:
            flak = sum(d for _, d, _ in shots)
            flak_from = ref("unit", shots[0][0].id)
            a = strength(
                state, rules, actor, ranged=True, hp=actor.hp - flak, against=target.nation
            )
    tactical = attack_modifiers(state, rules, actor, tile, kind, target)
    a = _apply(a, tactical)
    damage = max(8, min(65, 30 * a // d)) if actor.hp > flak else 0
    # 兵种对部队 / 城市的伤害倍率（炮兵：打部队弱、削城防强），只改伤害，不改战斗力和反击
    scale = _damage_scale(rules, actor, target, state)
    if scale != 100:
        damage = damage * scale // 100
        tactical = [
            *tactical,
            ("攻城火力" if isinstance(target, City) else "对部队火力", scale - 100),
        ]
    if isinstance(target, City) and kind == "melee":
        # 城防挡住近战：近战攻城伤害 −70%（炮兵、飞机、舰船不受影响）
        cut = rules.cities.melee_reduction
        damage = max(1, damage * (100 - cut) // 100)
        tactical = [*tactical, ("城防挡住近战", -cut)]
    if isinstance(target, Unit) and rules.units[target.type].domain == "sea":
        fac = rules.facility_at(int(state.grid.facility[target.tile]))
        if (
            fac is not None
            and fac.shelter
            and int(state.grid.owner[target.tile]) == target.nation
            and target.tile not in state.damaged
        ):
            # 停在本国港口里的舰船：受到的伤害 −80%
            damage = max(1, damage * (100 - fac.shelter) // 100)
            tactical = [*tactical, (f"{fac.name}掩护", -fac.shelter)]
    ring_now = encirclement.status(state, rules)
    if (isinstance(target, City) and target.id in ring_now.cities) or (
        isinstance(target, Unit) and target.id in ring_now.units
    ):
        # 被包围的部队 / 城市受到的伤害翻倍
        mult = rules.encirclement.damage_taken
        damage = damage * mult // 100
        tactical = [*tactical, ("目标被包围", mult - 100)]
    at_sea = isinstance(target, Unit) and rules.units[target.type].domain == "land"
    at_sea = at_sea and afloat(state, rules, target)
    if at_sea:
        # 下了海的陆军血量极低：受到的伤害 ×embark.damage_taken%
        mult = rules.embark.damage_taken
        damage = damage * mult // 100
        tactical = [*tactical, ("目标在海上（血量极低）", mult - 100)]
    # 炮击整格：其余敌军（和城市）各按自己的防御力受伤；地毯式轰炸（area_tiles = 4）再覆盖目标格往前的
    # 菱形三格，那三格里的敌军、敌城按 AREA_OUTER% 伤害
    area = []
    area_tiles = [tile]
    if (
        kind != "melee"
        and isinstance(actor, Unit)
        and "area_damage" in rules.units[actor.type].tags
    ):
        ut_a = rules.units[actor.type]
        if ut_a.area_tiles > 1:
            area_tiles = carpet(state.grid, origin, tile)[: ut_a.area_tiles]
        layer = target.layer if isinstance(target, Unit) else "surface"
        for t in area_tiles:
            others = [
                u
                for u in state.units_at(t, layer if t == tile else "surface")
                if _war(state, u.nation, actor.nation)
                and rules.units[u.type].category == "military"
                and u is not target
            ]
            city = state.city_at(t)
            if city is not None and _war(state, city.nation, actor.nation) and city is not target:
                others.append(city)
            for o in sorted(others, key=lambda o: (isinstance(o, City), o.id)):
                hit = max(
                    8,
                    min(
                        65,
                        30 * a // strength(state, rules, o, defending=True, against=actor.nation),
                    ),
                )
                hit = hit * _damage_scale(rules, actor, o, state) // 100
                if isinstance(o, Unit) and afloat(state, rules, o):
                    hit = hit * rules.embark.damage_taken // 100
                if t != tile:
                    hit = hit * AREA_OUTER // 100
                area.append(
                    {
                        "target": ref("city" if isinstance(o, City) else "unit", o.id),
                        "tile": t,
                        "damage": min(
                            city_defense(state, rules, o) if isinstance(o, City) else o.hp, hit
                        ),
                    }
                )
    retaliation = max(6, min(60, 25 * d // a)) if kind == "melee" and not at_sea else 0
    if (city_defense(state, rules, target) if isinstance(target, City) else target.hp) <= 0:
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
        "damage": min(
            city_defense(state, rules, target) if isinstance(target, City) else target.hp, damage
        ),
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
        "area_tiles": area_tiles,
    }, None


AREA_OUTER = 70  # 地毯式轰炸：目标格之外那三格的伤害（%）


def carpet(grid, origin: int, tile: int) -> list[int]:
    """地毯式轰炸的菱形：目标格、沿进攻方向往前一格、以及这两格共同的两个邻格（共 4 格）。"""
    from engine.systems.zoc import bearing

    d = bearing(grid, origin, tile) if origin != tile else 0
    out = [tile]
    for k in (d, (d + 5) % 6, (d + 1) % 6):
        n = int(grid.neighbors[tile, k])
        if n >= 0:
            out.append(n)
    return out


def tactics(state, rules, actor) -> dict:
    reach = attack_range(rules, actor, state.grid)
    sortie = getattr(actor, "sortie", None)
    origin = sortie["base"] if sortie is not None else actor.tile
    tiles = state.grid.within(origin, reach)
    tiles = sorted(t for t in tiles if t != actor.tile or sortie is not None)
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


def preview_from(state, rules, actor, spot: int, left: int, tile: int):
    """actor 先走到 spot（剩 left 的移动力，单位 1/MOVE_STEPS 点）再打 tile 的预览（文明 6 式先走后打）。
    临时把单位挪过去算完再挪回来，状态不变。"""
    old = (
        actor.tile,
        actor.moves_left,
        actor.road_credit,
        actor.facing,
        actor.fortified,
        actor.destination,
    )
    try:
        if spot != actor.tile:
            state.move_unit(actor, spot)
            actor.fortified = False
        actor.moves_left, actor.road_credit = divmod(left, MOVE_STEPS)
        actor.destination = None  # 走到了：这回合已经下了去 spot 的移动指令也按到达算
        return preview(state, rules, actor, tile)
    finally:
        if actor.tile != old[0]:
            state.move_unit(actor, old[0])
        (actor.moves_left, actor.road_credit, actor.facing, actor.fortified, actor.destination) = (
            old[1:]
        )


def approach_targets(state, rules, actor, limit: int = 24) -> list[dict]:
    """先移动、再攻击能打到的目标（不含站着就能打的）：[{from, left, ...预览}]。
    每个目标挑剩余移动力最多的出发格；只算地面和海上部队（飞机按出击规则另算）。"""
    if not isinstance(actor, Unit) or rules.units[actor.type].domain == "air":
        return []
    if lock_reason(state, rules, actor):
        return []
    from engine.systems.diplomacy import at_war
    from engine.systems.movement import attack_steps, has_room, movement_map

    best, _ = movement_map(state, rules, actor)
    need = attack_steps(rules, actor)
    spots = sorted(
        (t for t, left in best.items() if t != actor.tile and left >= need),
        key=lambda t: (-best[t], t),
    )
    spots = [t for t in spots if has_room(state, rules, actor, t)]
    reach = attack_range(rules, actor, state.grid)
    here = {t for t in state.grid.within(actor.tile, reach)}

    def hostile(t: int) -> bool:
        city = state.city_at(t)
        if (
            city is not None
            and city.nation != actor.nation
            and at_war(state, city.nation, actor.nation)
        ):
            return True
        return any(
            u.nation != actor.nation
            and rules.units[u.type].category == "military"
            and at_war(state, u.nation, actor.nation)
            for u in state.units_at(t)
        )

    out, seen = [], set()
    for spot in spots:
        for t in state.grid.within(spot, reach):
            if t in seen or t == spot or t in here or not hostile(t):
                continue
            info, _ = preview_from(state, rules, actor, spot, best[spot], t)
            if info is None:
                continue
            seen.add(t)
            out.append({**info, "from": spot, "left": best[spot]})
            if len(out) >= limit:
                return out
    return out


def _capture(ctx, city, unit):
    s, rules = ctx.state, ctx.rules
    previous = city.nation
    from engine.systems.diplomacy import note_capture

    note_capture(s, previous, unit.nation)  # 求和时看僵持
    s._closed_cache.clear()
    was_capital = city.capital
    city.nation, city.hp, city.capital = unit.nation, city_max_hp(rules, city) // 2, False
    city.drafted = 0  # 原国家征召走的人口不再占新主人的住房
    city.captured_turn, city.captured_from = s.turn, previous
    city.stability = min(city.stability, rules.stability.captured)  # 新占领地区不稳
    city.queue, city.production, city.attacked_turn = [], 0, s.turn
    city.managed = True
    city.management = {"status": "pending", "reason": "城市已易主，按新国家方针重新安排"}
    city.production_reason = ""
    s._network_cache = None
    s._settlement_usage = None
    s.invalidate_paths()
    s._worked_cache.clear()
    s._yield_cache.clear()
    g = s.grid
    g.owner[g.city_of == city.id] = unit.nation
    # 原来属于这座城、被占领划走的土地：在攻占方手里的还给这座城
    for t, home in list(s.land_home.items()):
        if home == city.id and int(g.owner[t]) == unit.nation and s.city_at(t) is None:
            g.city_of[t] = city.id
            del s.land_home[t]
    s._territory_cache = None
    # Aircraft on a captured base and noncombatants surrender with the city.
    for parked in list(s.units_at(city.tile)):
        if parked.nation == previous:
            credit(unit, parked, s)  # 城里投降的飞机、平民记进占城部队的战绩
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
        capital=was_capital,
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


def credit(unit, victim, state) -> None:
    """战绩：unit 击败了 victim（连同 victim 搭载的舰载机一起算）。城市、设施的攻击不记。"""
    if isinstance(unit, Unit) and unit.id in state.units and victim is not unit:
        unit.kills += 1 + len(state.aboard(victim))


def _destroy(ctx, unit, tile, reason=None, by=None):
    """消灭 unit（by：击败它的单位，记进 by 的战绩）。"""
    credit(by, unit, ctx.state)
    extra = {"reason": reason} if reason else {}
    if unit.name:
        extra["name"] = unit.name  # 舰名、将领姓名
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
        _base_fire(ctx)
        from engine.systems import air

        air.after_combat(ctx)
    finally:
        s._encircled_lock = False
        s._encircled = None


def _base_fire(ctx) -> None:
    """自带炮击的设施（军事基地）：每回合战斗阶段末，自动打射程内生命最少的交战国地表部队。
    炸坏的、没被本国控制的不开火。"""
    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    kinds = [b for b in rules.buildings.values() if b.bombard]
    if not kinds:
        return
    index = {rules.facility_index(b.id): b for b in kinds}
    for tile in np.flatnonzero(np.isin(g.facility, list(index))).tolist():
        owner = int(g.owner[tile])
        if owner < 0 or tile in s.damaged:
            continue
        b = index[int(g.facility[tile])]
        targets = [
            u
            for t in g.within(tile, b.bombard_range)
            for u in s.units_at(t, "surface")
            if _war(s, u.nation, owner) and _military(rules, u) and s.city_at(t) is None
        ]
        if not targets:
            continue
        victim = min(targets, key=lambda u: (u.hp, u.id))
        d = strength(s, rules, victim, defending=True, against=owner)
        damage = min(victim.hp, max(8, min(65, 30 * b.bombard // d)))
        victim.hp -= damage
        victim.damaged_turn = s.turn
        ctx.log.add(
            "base_fire",
            tile=tile,
            facility=b.id,
            nation=ref("nation", owner),
            target=ref("unit", victim.id),
            target_tile=victim.tile,
            damage=damage,
            target_hp=victim.hp,
        )
        if victim.hp <= 0:
            _destroy(ctx, victim, victim.tile, f"{b.name}炮击")


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
    """交战格里的部队自动接着打：每支（按本回合的先后名次、ID 顺序，轮流先手）打一次格子里最强的敌军，
    双方都受伤。"""
    from engine.systems.turn_order import rotation

    s, rules = ctx.state, ctx.rules
    rank = rotation(s)
    for tile in _contested(s, rules):
        for uid in sorted(
            (u.id for u in s.units_at(tile, "surface") if _military(rules, u)),
            key=lambda i: (rank.get(s.units[i].nation, 0), i),
        ):
            u = s.units.get(uid)
            if u is None or u.attacked_turn == s.turn or u.formation is not None:
                continue
            foes = [
                v
                for v in s.units_at(tile, "surface")
                if _war(s, v.nation, u.nation) and _military(rules, v)
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
                _destroy(ctx, target, tile, by=u)
            if u.hp <= 0:
                _destroy(ctx, u, tile, by=target)
        _settle(ctx, tile)


def _advance(ctx: StepContext, actor, start: int, tile: int) -> None:
    """近战打完：目标格里已经没有敌方军事单位、本方还放得下，就开进去（剩下的敌方平民被消灭）。"""
    s, rules = ctx.state, ctx.rules
    if s.units.get(actor.id) is not actor or actor.tile == tile or s.city_at(tile) is not None:
        return
    here = s.units_at(tile, actor.layer)
    if any(u.nation != actor.nation and _military(rules, u) for u in here):
        return
    if any(u.nation != actor.nation and not _war(s, u.nation, actor.nation) for u in here):
        return
    from engine.systems.stacking import has_room

    if not has_room(s, rules, tile, rules.units[actor.type].stack_class, actor.nation):
        return
    for stranded in list(here):
        if stranded.nation != actor.nation:
            _destroy(ctx, stranded, tile, "被消灭", by=actor)
    s.move_unit(actor, tile)
    ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])


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
            _destroy(ctx, stranded, tile, "被消灭", by=winner)  # 留在格里的敌方平民（含将领）被消灭


def _intercept(state, rules, order) -> bool:
    """战斗机打空中目标的攻击（拦截）：目标格上有交战国的飞机。"""
    if isinstance(order, CityAttack):
        return False
    unit = state.units.get(parse_ref(order.unit, "unit"))
    if unit is None or "air_superiority" not in rules.units[unit.type].tags:
        return False
    tile = state.grid.idx(*order.to)
    return any(_war(state, v.nation, unit.nation) for v in state.units_at(tile, "air"))


def _resolve(ctx: StepContext) -> None:
    s, rules = ctx.state, ctx.rules
    _skirmish(ctx)
    # 战斗机打空中目标（敌方飞机）先结算，其余攻击按提交顺序（远程先于近战）；轰炸机起飞当回合投弹，
    # 投弹前先挨敌方战斗机拦截一次（air.intercept）
    # 轮流先手（engine/systems/turn_order.py）：各国的攻击按本回合的先后名次结算，同一国家内保持提交顺序
    from engine.systems.turn_order import rotation

    rank = rotation(s)
    attacks = sorted(ctx.attacks, key=lambda o: rank.get(parse_ref(o.issuer, "nation"), 0))
    intercepts = [o for o in attacks if _intercept(s, rules, o)]
    rest = [o for o in attacks if not _intercept(s, rules, o)]
    for order in [*intercepts, *rest]:
        _execute(ctx, order)


def _execute(ctx: StepContext, order) -> None:
    """结算一条攻击指令（_resolve 按顺序调用；拦截来袭轰炸机的战斗机也走这里，air.intercept）。"""
    s, rules = ctx.state, ctx.rules
    from engine.systems import air

    is_city = isinstance(order, CityAttack)
    collection, key = (s.cities, "city") if is_city else (s.units, "unit")
    automatic_order = (order.type, getattr(order, key), order.to) in ctx.automatic_attacks
    actor = collection.get(parse_ref(getattr(order, key), key))
    if actor is None:
        if automatic_order:
            return
        ctx.log.reject(order.model_dump(mode="json"), "not_found", "攻击者已不存在")
        return
    if actor.nation != parse_ref(order.issuer, "nation"):
        if automatic_order:
            return
        ctx.log.reject(order.model_dump(mode="json"), "not_owner", "攻击者已不属于发令国")
        return
    tile = s.grid.idx(*order.to)
    # 先移动后攻击（文明 6 式）：这时部队已经走完，按走到的位置和剩下的移动力结算（lock_reason）
    info, problem = preview(s, rules, actor, tile)
    if problem:
        if automatic_order:
            ctx.log.add(
                "directive_report",
                nation=order.issuer,
                goal="unit_ai",
                status="blocked",
                reason=problem,
            )
            return
        ctx.log.reject(order.model_dump(mode="json"), "cannot_attack", problem)
        return
    start = actor.tile
    if isinstance(actor, Unit) and rules.units[actor.type].domain == "air":
        if actor.sortie is None:
            # 起飞：飞到目标上空当回合就打；轰炸机投弹前先挨敌方战斗机拦截一次
            if not air.launch(ctx, actor, tile):
                return
            if not air.is_fighter(rules, actor) and not air.intercept(ctx, actor):
                return
        elif actor.tile != tile and not air.fly(ctx, actor, tile):
            return  # 在空中的战斗机转去打别的目标，途中被击落
        info, problem = preview(s, rules, actor, tile)  # 飞到了：按带伤后的状态重新结算
        if problem:
            actor.attacked_turn = s.turn
            return
        if info.get("facility_strike"):
            bomb_facility(ctx, actor, tile)
            actor.attacked_turn = s.turn
            gain(ctx, actor, 4, "unit")
            return
    if info.get("storm_city"):
        # 城防已空：近战开进去占领，城里剩下的敌方单位全部消灭（_capture）
        city = s.city_at(tile)
        _capture(ctx, city, actor)
        s.move_unit(actor, tile)
        actor.attacked_turn = s.turn
        actor.moves_left, actor.road_credit, actor.destination, actor.fortified = (
            0,
            0,
            None,
            False,
        )
        ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])
        return
    if info.get("seize_city"):
        # 战舰开进港口占领城防已空的城市
        city = s.city_at(tile)
        _capture(ctx, city, actor)
        s.move_unit(actor, tile)
        actor.attacked_turn = s.turn
        actor.moves_left, actor.road_credit, actor.destination, actor.fortified = (
            0,
            0,
            None,
            False,
        )
        ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])
        return
    target = target_at(s, rules, actor, tile)
    if isinstance(target, City):
        damage_city(ctx, target, info["damage"], by=actor)  # 先打城市本体，再打守城部队
    else:
        target.hp = max(0, target.hp - info["damage"])
    if isinstance(actor, Unit) and actor.tile != tile:
        from engine.systems.zoc import bearing

        actor.facing = bearing(s.grid, actor.tile, tile)  # 攻击时朝向目标
    splashed = []
    for hit in info["area"]:
        kind_, _, oid = hit["target"].partition(":")
        other = s.cities.get(int(oid)) if kind_ == "city" else s.units.get(int(oid))
        if other is None:
            continue
        if isinstance(other, City):
            damage_city(ctx, other, hit["damage"], by=actor)
        else:
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
    if isinstance(actor, Unit) and actor.sortie is not None:
        actor.sortie["strikes"] = actor.sortie.get("strikes", 0) + 1  # 战斗机一次出击最多打两次
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
    if info["kind"] == "air" and bombable(s, rules, actor, tile):
        bomb_facility(ctx, actor, tile)  # 轰炸格上的敌方设施一并炸坏
    gain(ctx, target, 3, "city" if isinstance(target, City) else "unit")
    for other in splashed:
        if isinstance(other, Unit) and other.hp <= 0 and other.id in s.units:
            _destroy(ctx, other, other.tile, "炮击", by=actor)
    dead_target = (city_defense(s, rules, target) if isinstance(target, City) else target.hp) <= 0
    if dead_target and isinstance(target, Unit):
        credit(actor, target, s)
        ctx.log.add(
            "unit_destroyed",
            unit=ref("unit", target.id),
            tile=target.tile,
            unit_type=target.type,
            nation=ref("nation", target.nation),
            **({"name": target.name} if target.name else {}),
        )
        _lost_aboard(ctx, target)
        s.remove_unit(target.id)
    if actor.hp <= 0 and isinstance(actor, Unit):
        credit(target, actor, s)  # 反击打死了攻击方
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
        # 近战从相邻格打，敌我不同格：目标格的敌人打光了才开进去（留下的敌方平民被消灭）
        _advance(ctx, actor, start, tile)
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
                        _destroy(ctx, stranded, tile, "被消灭", by=actor)
            s.move_unit(actor, tile)
            ctx.log.add("unit_moved", unit=ref("unit", actor.id), path=[start, tile])
