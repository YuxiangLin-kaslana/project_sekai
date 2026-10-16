"""城市辖区整理：打仗时土地易手（占领区域、攻下城市）以后，把一个国家的领土在它自己的城市之间重新分一次。

规则：本国的每一格归离它最近的本国城市；当前所属的城市比最近的城市远不超过 SLACK 格的不动
（住宅区往外扩的土地一般还留在建它的城），所属城市已经不是本国的、或者远得多的才划过去。
这样占领、攻城之后不会留下飞地、长条和一格的空城。
"""

from __future__ import annotations

import numpy as np

SLACK = 1


def rebalance(state, nations) -> int:
    """整理这些国家的城市辖区，返回改划的格子数。"""
    g = state.grid
    moved = 0
    for nid in sorted(set(nations)):
        cities = state.nation_cities(nid)
        if not cities:
            continue
        tiles = np.flatnonzero(g.owner == nid)
        if not len(tiles):
            continue
        ids = np.array([c.id for c in cities])
        dist = np.stack([g.distance_from(c.tile)[tiles] for c in cities])
        best = dist.min(axis=0)
        nearest = ids[dist.argmin(axis=0)]
        index = {cid: k for k, cid in enumerate(ids)}
        centers = {c.tile for c in cities}
        for i, t in enumerate(tiles.tolist()):
            if t in centers:
                continue
            cur = int(g.city_of[t])
            k = index.get(cur)
            if k is not None and dist[k, i] <= best[i] + SLACK:
                continue
            g.city_of[t] = int(nearest[i])
            moved += 1
    if moved:
        state._territory_cache = None
        state._worked_cache.clear()
        state._yield_cache.clear()
        state._settlement_usage = None
        state._network_cache = None
    return moved


# ---------------------------------------------------------------- 国土连贯

ENCLAVE_SHARE = 0.5  # 一块国土外围的陆地邻格，归某个外国的占这么大比例以上：算被它包围


def owner_blocks(state) -> np.ndarray:
    """国土连成的块：label[t] = 这一格所在块的编号（块里编号最小的格子），同一国家、六邻相连的陆地算一块；
    水面、无主的格子为 -1。标签传播加指针跳跃，整图向量化。"""
    g = state.grid
    owner = g.owner
    ok = ~g.is_water & (owner >= 0)
    label = np.where(ok, np.arange(g.size), -1)
    links = []
    for d in range(6):
        j = g.neighbors[:, d]
        src = np.flatnonzero(ok & (j >= 0))
        dst = j[src]
        same = ok[dst] & (owner[dst] == owner[src])
        links.append((src[same], dst[same]))
    while True:
        before = label.copy()
        for src, dst in links:
            np.minimum.at(label, src, label[dst])
        label[ok] = label[label[ok]]
        if np.array_equal(label, before):
            return label


def _blocks(state, label) -> list[np.ndarray]:
    tiles = np.flatnonzero(label >= 0)
    order = np.argsort(label[tiles], kind="stable")
    tiles = tiles[order]
    return np.split(tiles, np.flatnonzero(np.diff(label[tiles])) + 1) if len(tiles) else []


def _surround(state, label, block) -> tuple[int, float]:
    """一块国土外围的陆地邻格里占得最多的外国和它的比例（没有外国邻格时返回 (-1, 0)）。"""
    g = state.grid
    nb = g.neighbors[block].ravel()
    nb = nb[nb >= 0]
    nb = nb[~g.is_water[nb] & (label[nb] != label[block[0]])]
    if not len(nb):
        return -1, 0.0
    owners = g.owner[nb]
    me = int(g.owner[block[0]])
    foreign = owners[(owners >= 0) & (owners != me)]
    if not len(foreign):
        return -1, 0.0
    counts = np.bincount(foreign)
    best = int(counts.argmax())  # 并列取编号小的
    return best, float(counts[best]) / len(nb)


def _hand_over(state, tiles, to) -> None:
    """一片没有城市的国土划给 to：归它离得最近的城（原来就是从它那座城划出去的，回原城）。"""
    g = state.grid
    for t in tiles.tolist():
        g.owner[t] = to
        home = state.land_home.get(t)
        city = state.cities.get(home) if home is not None else None
        if city is not None and city.nation == to:
            g.city_of[t] = home
            del state.land_home[t]
        else:
            state.land_home.setdefault(t, int(g.city_of[t]))
            g.city_of[t] = -1  # rebalance 会归到 to 最近的城
    _touch(state)


def _touch(state) -> None:
    state._territory_cache = None
    state._settlement_usage = None
    state._network_cache = None
    state._closed_cache.clear()
    state._worked_cache.clear()
    state._yield_cache.clear()
    state._support = None
    state.invalidate_paths()


def tidy(ctx) -> int:
    """零碎国土归还：没有城市、和本国别的国土不连着、本国陆军也没有据守（块里和紧挨着没有本国陆军）的小块，
    外围大半是某个外国的（ENCLAVE_SHARE），就归还给它——占领区域留下的碎块、被切断的边角不再满地都是。
    打仗时有部队站着的占领区照常保留。返回归还的格子数。"""
    from engine.ids import ref
    from engine.systems.naval import afloat

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    label = owner_blocks(s)
    centers = np.zeros(g.size, dtype=bool)
    for c in s.cities.values():
        centers[c.tile] = True
    held = {}
    for u in s.units.values():
        ut = rules.units[u.type]
        if ut.domain == "land" and ut.category == "military" and u.layer == "surface":
            if not afloat(s, rules, u):
                held.setdefault(u.nation, set()).add(u.tile)
    moved, touched = 0, set()
    for block in _blocks(s, label):
        if centers[block].any():
            continue
        nid = int(g.owner[block[0]])
        mine = held.get(nid, ())
        if mine and any(
            t in mine or any(n in mine for n in g.neighbors_of(t)) for t in block.tolist()
        ):
            continue
        to, share = _surround(s, label, block)
        if to < 0 or share < ENCLAVE_SHARE:
            continue
        _hand_over(s, block, to)
        touched.update((nid, to))
        moved += len(block)
        ctx.log.add(
            "territory_returned",
            nation=ref("nation", nid),
            to=ref("nation", to),
            tiles=len(block),
            tile=int(block[0]),
        )
    if touched:
        rebalance(s, touched)
    return moved


def enclaves(state, rules, nid, label=None) -> list[tuple[np.ndarray, int, float]]:
    """nid 的飞地：带城市、不是首都那块（没有首都时是城最多的那块）、和主块在同一片陆地上的国土块，
    连同包围它的外国和比例。隔海的海外领土不算飞地。"""
    from engine.systems.settlement import geography

    label = owner_blocks(state) if label is None else label
    cities = state.nation_cities(nid)
    if len(cities) < 2:
        return []
    by_block = {}
    for c in cities:
        by_block.setdefault(int(label[c.tile]), []).append(c)
    cap = state.cities.get(state.nations[nid].capital or -1)
    main = (
        int(label[cap.tile])
        if cap is not None and cap.nation == nid
        else max(by_block, key=lambda k: (len(by_block[k]), -k))
    )
    comps = geography(state, rules)["land_components"]
    home = int(comps[by_block[main][0].tile])
    out = []
    for k in sorted(by_block):
        if k == main or int(comps[by_block[k][0].tile]) != home:
            continue
        block = np.flatnonzero(label == k)
        to, share = _surround(state, label, block)
        out.append((block, to, share))
    return out


def cede_enclaves(ctx, a: int, b: int) -> None:
    """和约划界：a、b 签和约时，一方陷在对方国土里的飞地（外围大半是对方的，ENCLAVE_SHARE）连城带地
    割让给对方，双方国土各自连成一片。飞地上的部队随后按停战遣返送回本国（engine/systems/borders.py）。"""
    from engine.ids import ref
    from engine.systems.combat import city_max_hp

    s, rules, g = ctx.state, ctx.rules, ctx.state.grid
    label = owner_blocks(s)
    deals = []
    for giver, taker in ((a, b), (b, a)):
        for block, to, share in enclaves(s, rules, giver, label):
            if to == taker and share >= ENCLAVE_SHARE:
                deals.append((giver, taker, block))
    for giver, taker, block in deals:
        tiles = set(block.tolist())
        ceded = sorted((c for c in s.nation_cities(giver) if c.tile in tiles), key=lambda c: c.id)
        for c in ceded:
            c.nation, c.capital, c.queue, c.production = taker, False, [], 0
            c.captured_turn, c.captured_from, c.drafted = s.turn, giver, 0
            c.hp = max(c.hp, city_max_hp(rules, c) // 2)
            c.managed = True
            c.management = {"status": "pending", "reason": "和约割让，按新国家方针重新安排"}
            if c.construction is not None:
                c.construction.nation = taker
        g.owner[block] = taker
        _touch(s)
        ctx.log.add(
            "cities_ceded",
            nation=ref("nation", giver),
            to=ref("nation", taker),
            cities=[ref("city", c.id) for c in ceded],
            tiles=len(block),
        )
        from engine.systems.upheaval import record

        names = "、".join(c.name for c in ceded)
        record(
            s,
            "ceded",
            f"{s.nations[giver].name}依和约把陷在{s.nations[taker].name}境内的{names}割让给对方",
            (giver, taker),
            ceded[0].tile if ceded else None,
        )
    if deals:
        rebalance(s, (a, b))
