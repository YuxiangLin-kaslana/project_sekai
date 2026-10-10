"""一回合一次并查集：本国首都连接、跨国贸易；循环道路不重复产生收益。"""


def index(state, rules):
    if state._network_cache is not None:
        return state._network_cache
    parent, domestic = {}, {}

    def root(table, key):
        table.setdefault(key, key)
        while table[key] != key:
            table[key] = table[table[key]]
            key = table[key]
        return key

    def union(table, a, b):
        a, b = root(table, a), root(table, b)
        if a != b:
            table[max(a, b)] = min(a, b)

    for (a, b), road in state.roads.items():
        union(parent, a, b)
        union(domestic, (road["nation"], a), (road["nation"], b))
    linked = {}
    clusters = {}
    for c in state.cities.values():
        cap = state.cities.get(state.nations[c.nation].capital)
        connected = cap is not None and root(domestic, (c.nation, c.tile)) == root(
            domestic, (c.nation, cap.tile)
        )
        linked[c.id] = connected
        if connected:
            clusters.setdefault(c.nation, []).append(c.id)
    foreign = {}
    for c in state.cities.values():
        if linked[c.id]:
            foreign.setdefault(root(parent, c.tile), set()).add(c.nation)
    result = {}
    for c in state.cities.values():
        partners = (
            sorted(foreign.get(root(parent, c.tile), set()) - {c.nation}) if linked[c.id] else []
        )
        partners = partners[: rules.cities.max_trade_partners]
        result[c.id] = {
            "capital_connected": linked[c.id],
            "production_percent": rules.cities.capital_road_bonus
            if linked[c.id] and len(clusters.get(c.nation, [])) > 1
            else 0,
            "trade_partners": partners,
            "trade_gold": len(partners) * rules.cities.foreign_trade_gold,
        }
    state._network_cache = result
    return result


def city_network(state, rules, city):
    return index(state, rules).get(
        city.id,
        {
            "capital_connected": False,
            "production_percent": 0,
            "trade_partners": [],
            "trade_gold": 0,
        },
    )
