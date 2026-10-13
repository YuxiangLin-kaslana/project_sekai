import numpy as np
from fastapi.testclient import TestClient

from server.app import app
from server.game import unpack
from server.mapdata import CHUNK

client = TestClient(app)
SMALL = {"seed": 5, "width": 40, "height": 25, "nations": 4}


def test_index_and_health():
    page = client.get("/")
    assert page.status_code == 200
    assert '"/static/viewer.js?v=' in page.text
    assert "&d=" in page.text  # 数据版本，改了生成代码后让分块缓存失效
    assert page.headers["cache-control"] == "no-cache"
    assert client.get("/static/viewer.js").status_code == 200
    assert client.get("/healthz").json()["ok"] is True


def test_world_meta():
    meta = client.get("/api/world", params=SMALL).json()
    assert (meta["width"], meta["height"], meta["chunk"]) == (40, 25, CHUNK)
    assert len(meta["starts"]) == 4
    assert meta["land_areas"] > 0 and meta["sea_areas"] > 0
    assert all(len(r["p"]) >= 4 and len(r["p"]) % 2 == 0 and r["w"] > 0 for r in meta["rivers"])
    assert {"terrain", "resources", "biomes"} <= meta["ruleset"].keys()
    assert meta["plates"] and meta["wind_speed_scale"] > 0


def test_land_ratio_param():
    a = client.get("/api/world", params={**SMALL, "land": 0.2}).json()["stats"]["land_ratio"]
    b = client.get("/api/world", params={**SMALL, "land": 0.7}).json()["stats"]["land_ratio"]
    assert abs(a - 0.2) < 0.02 and abs(b - 0.7) < 0.02


def test_chunks_cover_map_and_decode():
    meta = client.get("/api/world", params=SMALL).json()
    w, h = meta["width"], meta["height"]
    total = 0
    for cy in range((h + CHUNK - 1) // CHUNK):
        for cx in range((w + CHUNK - 1) // CHUNK):
            r = client.get("/api/world/chunk", params={**SMALL, "cx": cx, "cy": cy})
            assert r.status_code == 200
            n = min(CHUNK, w - cx * CHUNK) * min(CHUNK, h - cy * CHUNK)
            assert len(r.content) == 19 * n
            area = np.frombuffer(r.content[: 2 * n], dtype="<u2")
            terrain = np.frombuffer(r.content[6 * n : 7 * n], dtype=np.uint8)
            temp = np.frombuffer(r.content[11 * n : 12 * n], dtype=np.int8)
            plate = np.frombuffer(r.content[12 * n : 13 * n], dtype=np.uint8)
            wspd = np.frombuffer(r.content[15 * n : 16 * n], dtype=np.uint8)
            canom = np.frombuffer(r.content[18 * n : 19 * n], dtype=np.int8)
            is_water = np.array([t["water"] for t in meta["ruleset"]["terrain"]])[terrain]
            assert (canom[~is_water] == 0).all()  # 陆地上没有洋流
            assert plate.max() < len(meta["plates"])
            assert wspd.max() > 0
            assert -60 <= temp.min() and temp.max() <= 60
            assert (area < meta["land_areas"] + meta["sea_areas"]).all()
            assert terrain.max() < len(meta["ruleset"]["terrain"])
            total += n
    assert total == w * h
    out = client.get("/api/world/chunk", params={**SMALL, "cx": 99, "cy": 0})
    assert out.status_code == 404


def test_climate_params_shift_climate():
    def stats(**kw):
        return client.get("/api/world", params={**SMALL, **kw}).json()["stats"]

    assert (
        stats(temp=-1)["avg_temperature_c"]
        < stats()["avg_temperature_c"]
        < stats(temp=1)["avg_temperature_c"]
    )
    assert (
        stats(rain=-1)["avg_rainfall_mm"]
        < stats()["avg_rainfall_mm"]
        < stats(rain=1)["avg_rainfall_mm"]
    )
    smooth, rugged = stats(rough=0)["terrain"], stats(rough=1)["terrain"]
    assert smooth.get("mountains", 0) < rugged.get("mountains", 0)


def test_params_are_bounded():
    assert client.get("/api/world", params={"width": 5000}).status_code == 422
    assert client.get("/api/world", params={"seed": -1}).status_code == 422
    assert client.get("/api/world", params={"land": 0.95}).status_code == 422
    assert client.get("/api/world", params={"temp": 2}).status_code == 422
    assert client.get("/api/world", params={"rough": -0.1}).status_code == 422


# ---------------------------------------------------------------- 对局接口


def test_game_create_get_and_turn():
    created = client.post("/api/games", json=SMALL).json()
    assert {"units", "buildings", "facilities", "stacking"} <= created["rules"].keys()
    assert created["turn"] == 1 and len(created["nations"]) == 4
    assert created["world"]["seed"] == SMALL["seed"]
    city = created["cities"][0]
    city_of = unpack(created["city_of"])  # 每格归属压缩发送，前端还原
    assert city_of.size == SMALL["width"] * SMALL["height"]
    assert city_of[city["tile"]] == int(city["id"].split(":")[1])
    assert {"col", "row", "level"}.isdisjoint(city)  # 坐标、等级由前端从格子编号和经验算
    gid = created["id"]

    got = client.get(f"/api/games/{gid}").json()
    assert got["state_hash"] == created["state_hash"]

    queue = {
        "issuer": city["nation"],
        "type": "city_order",
        "city": city["id"],
        "queue": ["unit:infantry"],
    }
    bad = {"issuer": city["nation"], "type": "fortify", "unit": "unit:999999"}
    res = client.post(f"/api/games/{gid}/turn", json={"orders": [queue, bad]}).json()
    assert res["turn"] == 2
    assert [r["code"] for r in res["rejected"]] == ["not_found"]
    assert next(c for c in res["cities"] if c["id"] == city["id"])["queue"] == ["unit:infantry"]

    # 不带指令：没收到指令的单位随机游走
    before = {u["id"]: u["tile"] for u in res["units"]}
    res = client.post(f"/api/games/{gid}/turn", json={}).json()
    assert res["turn"] == 3 and not res["rejected"]
    assert sum(before[u["id"]] != u["tile"] for u in res["units"] if u["id"] in before) > 0


def test_game_errors():
    assert client.get("/api/games/nope").status_code == 404
    gid = client.post("/api/games", json=SMALL).json()["id"]
    r = client.post(
        f"/api/games/{gid}/turn", json={"orders": [{"issuer": "nation:1", "type": "warp"}]}
    )
    assert r.status_code == 422
    assert client.post("/api/games", json={**SMALL, "width": 5000}).status_code == 422


def test_tactical_actions_range_progression_and_manual_turn():
    from engine.ids import parse_ref
    from engine.systems.movement import movement_map
    from server.app import GAMES, RULES

    snap = client.post("/api/games", json=SMALL).json()
    gid = snap["id"]
    unit = next(u for u in snap["units"] if u["type"] == "infantry")
    assert {"xp", "formation", "attack_blocked"} <= unit.keys()
    assert {"hp", "xp"} <= snap["cities"][0].keys()
    preview = client.get(f"/api/games/{gid}/tactics", params={"actor": unit["id"]}).json()
    # 附近没有敌人；同格最多 2 个单位，只要不在行军 / 编队里就能交战
    assert preview["move_tiles"] and "attack_blocked" in preview
    assert set(map(int, preview["move_paths"])) == set(preview["move_tiles"])
    # Bad actor refs and coordinates fail cleanly.
    assert client.get(f"/api/games/{gid}/tactics", params={"actor": "unit:nope"}).status_code == 404
    assert (
        client.get(
            f"/api/games/{gid}/tactics", params={"actor": unit["id"], "col": 0, "row": 999}
        ).status_code
        == 422
    )
    dest = preview["move_tiles"][0]
    raw = {
        "issuer": unit["nation"],
        "type": "move",
        "unit": unit["id"],
        "to": [dest % SMALL["width"], dest // SMALL["width"]],
    }
    res = client.post(
        f"/api/games/{gid}/actions", json={"orders": [raw], "expected_hash": snap["state_hash"]}
    ).json()
    assert not res["rejected"] and res["turn"] == snap["turn"]
    assert next(u for u in res["units"] if u["id"] == unit["id"])["tile"] == dest
    assert any(e["type"] == "unit_moved" for e in res["events"])
    assert (
        client.post(
            f"/api/games/{gid}/actions", json={"orders": [raw], "expected_hash": snap["state_hash"]}
        ).status_code
        == 409
    )
    s = GAMES.get(gid).state
    remaining, _ = movement_map(s, RULES, s.units[parse_ref(unit["id"], "unit")])
    after = client.get(f"/api/games/{gid}/tactics", params={"actor": unit["id"]}).json()
    assert set(after["move_tiles"]) == set(remaining) - {dest}
    positions = {u["id"]: u["tile"] for u in res["units"]}
    turn = client.post(f"/api/games/{gid}/turn", json={"autoplay": False}).json()
    assert turn["turn"] == 2 and positions == {u["id"]: u["tile"] for u in turn["units"]}
    assert turn["cities"][0]["xp"] > 0 and turn["regions"]


def test_road_preview_and_construction_endpoint():
    from server.app import GAMES

    snap = client.post("/api/games", json=SMALL).json()
    gid = snap["id"]
    s = GAMES.get(gid).state
    mine = [c for c in snap["cities"] if c["nation"] == snap["cities"][0]["nation"]]
    city = mine[0]
    # 本国另一座城还没有：在辖区边上放一个工兵当目标（城市可以把路修到工兵所在位置）
    from engine.state import Unit

    nid = int(city["nation"].split(":")[1])
    spot = next(
        t
        for t in s.grid.within(city["tile"], 2)
        if s.grid.distance(t, city["tile"]) == 2
        and not s.grid.is_water[t]
        and int(s.grid.owner[t]) == nid
        and not s.units_at(t)
        and s.grid.terrain[t] != list(GAMES.get(gid).rules.terrain).index("mountains")
    )
    s.add_unit(Unit(s.new_id(), "engineer", nid, spot, "surface"))
    request = {"city": city["id"], "to": list(s.grid.col_row(spot))}
    plan = client.post(f"/api/games/{gid}/road-plan", json=request).json()
    assert plan["segments"] and plan["cost"] > 0, plan
    built = client.post(
        f"/api/games/{gid}/actions",
        json={"orders": [{"type": "build_road", "issuer": city["nation"], **request}]},
    ).json()
    assert not built["rejected"] and any(e["type"] == "road_planned" for e in built["events"])
    queued = next(c for c in built["cities"] if c["id"] == city["id"])
    assert queued["queue"][-1] == f"road:{spot}"


def test_compact_encoding_round_trip():
    """快照里的整格数组、道路编码可以无损还原（前端 game.js 用同样的规则解码）。"""
    import numpy as np

    from engine.hexmap.worldgen import WorldGenParams, generate_world
    from engine.stress import populate
    from server.app import RULES
    from server.game import pack

    rng = np.random.default_rng(1)
    for arr in (
        np.full(500, -1, dtype=np.int32),  # 全是同一个值：游程编码
        rng.integers(0, 9, 500).astype(np.uint16),  # 杂乱：base64
        rng.integers(-5, 70000, 300).astype(np.int32),
        np.zeros(0, dtype=np.int32),
    ):
        assert (unpack(pack(arr)) == arr).all()

    world = generate_world(WorldGenParams(width=40, height=25, seed=5, num_starts=4), RULES)
    state, _ = populate(world, RULES)
    from server.game import pack_roads

    roads = pack_roads(state)
    a = np.cumsum(roads["a"])
    b = state.grid.neighbors[a, roads["dir"]] if "dir" in roads else np.asarray(roads["b"])
    assert list(zip(a.tolist(), b.tolist(), strict=True)) == sorted(state.roads)
    assert [roads["kinds"][k] for k in roads["kind"]] == [
        state.roads[k]["kind"] for k in sorted(state.roads)
    ]


def test_fighter_patrol_tiles_and_order():
    snap = client.post("/api/games", json=SMALL).json()
    gid = snap["id"]
    fighter = next(u for u in snap["units"] if u["type"] == "fighter")
    tactics = client.get(f"/api/games/{gid}/tactics", params={"actor": fighter["id"]}).json()
    # 停在机场上的战斗机：作战半径内的格子都能去巡逻，带防空圈半径
    assert tactics["patrol_blocked"] is None and tactics["patrol_radius"] >= 1
    assert fighter["tile"] in tactics["patrol_tiles"] and len(tactics["patrol_tiles"]) > 7
    far = max(tactics["patrol_tiles"], key=lambda t: abs(t - fighter["tile"]))
    raw = {
        "issuer": fighter["nation"],
        "type": "patrol",
        "unit": fighter["id"],
        "to": [far % SMALL["width"], far // SMALL["width"]],
    }
    res = client.post(
        f"/api/games/{gid}/actions", json={"orders": [raw], "expected_hash": snap["state_hash"]}
    ).json()
    assert not res["rejected"]
    plane = next(u for u in res["units"] if u["id"] == fighter["id"])
    assert plane["sortie"] == "patrol" and plane["tile"] == far
    again = client.get(f"/api/games/{gid}/tactics", params={"actor": fighter["id"]}).json()
    assert again["patrol_blocked"] and "patrol_tiles" not in again  # 已经在空中


def test_snapshot_shows_the_tide_up_to_the_current_turn():
    snap = client.post("/api/games", json=SMALL).json()
    tide = snap["tide"]
    # 天下大势：当前值、阶段名、合力，曲线只画到当前回合（不剧透以后的走向），还有对各系统的倍率
    assert -1 <= tide["value"] <= 1 and tide["phase"] in {name for _, name in tide["phases"]}
    assert tide["curve"][-1][0] == snap["turn"] and tide["curve"][0][0] == 0
    assert {e[0] for e in tide["effects"]} >= {"宣战", "内战、将领叛变"}


def test_unit_war_history_names_the_units_it_killed():
    from server.app import GAMES

    snap = client.post("/api/games", json=SMALL).json()
    gid = snap["id"]
    unit = next(u for u in snap["units"] if u["type"] == "infantry")
    state = GAMES.get(gid).state
    u = state.units[int(unit["id"].split(":")[1])]
    foe = next(n for n in state.nations if n != u.nation)
    u.kills = 1
    u.history = (*u.history, (state.turn, "kill", ("armor", foe, "", 999999), 0))
    data = client.get(f"/api/games/{gid}/history", params={"unit": unit["id"]}).json()
    assert data["kills"] == 1 and data["entries"][-1]["kind"] == "born"
    kill = data["entries"][0]
    # 被击毁的单位已经不在了，战史里照样写明是哪国的什么兵种
    assert kill["kind"] == "kill" and kill["foe"]["type_name"] == "装甲师"
    assert kill["foe"]["nation_name"] == state.nations[foe].name and not kill["foe"]["alive"]
    missing = client.get(f"/api/games/{gid}/history", params={"unit": "unit:999999"})
    assert missing.status_code == 404


def test_snapshot_lists_wars_with_names_and_losses():
    from engine.systems import diplomacy
    from server.app import GAMES

    snap = client.post("/api/games", json=SMALL).json()
    gid = snap["id"]
    assert snap["wars"] == [] and snap["war_count"] == 0  # 开局各国和平
    state = GAMES.get(gid).state
    a, b = sorted(state.nations)[:2]
    diplomacy._set(state, a, b, "war")
    wars = client.get(f"/api/games/{gid}").json()["wars"]
    # 每场战争有名字、开战回合、双方（损失支数、折算兵力、攻占的城市）和重大战损
    assert len(wars) == 1 and wars[0]["name"].endswith("战争") and wars[0]["end"] is None
    assert [x["nation"] for x in wars[0]["sides"]] == [f"nation:{a}", f"nation:{b}"]
    assert wars[0]["sides"][0]["lost"] == 0 and wars[0]["events"] == []
