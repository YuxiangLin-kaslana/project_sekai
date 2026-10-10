"""剧本：服务器列出剧本、按剧本开局、提供剧本素材（只限剧本的 assets 目录）。"""

from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_scenarios_are_listed_and_games_remember_theirs():
    listed = client.get("/api/scenarios").json()
    assert any(x["id"] == "alt_ww2" and x["default"] for x in listed)
    game = client.post(
        "/api/games",
        json={"scenario": "alt_ww2", "seed": 2, "width": 40, "height": 25, "nations": 3},
    ).json()
    assert game["world"]["scenario"] == "alt_ww2" and game["rules"]["scenario"]["id"] == "alt_ww2"
    assert "icons" in game["rules"]["assets"]
    assert client.post("/api/games", json={"scenario": "nope"}).status_code == 404
    world = client.get("/api/world", params={"scenario": "alt_ww2", "width": 40, "height": 25})
    assert world.status_code == 200


def test_scenario_assets_stay_inside_the_assets_folder():
    assert client.get("/api/scenarios/alt_ww2/assets/README.md").status_code == 200
    assert client.get("/api/scenarios/alt_ww2/assets/../ruleset.yaml").status_code == 404
    assert client.get("/api/scenarios/nope/assets/README.md").status_code == 404


def test_rules_doc_and_new_snapshot_fields():
    doc = client.get("/api/scenarios/alt_ww2/rules-doc").json()
    assert doc["markdown"].startswith("# 二战剧本") and "控制区" in doc["markdown"]
    assert client.get("/api/scenarios/nope/rules-doc").status_code == 404
    game = client.post(
        "/api/games",
        json={"scenario": "alt_ww2", "seed": 2, "width": 40, "height": 25, "nations": 3},
    ).json()
    capital = next(c for c in game["cities"] if c["capital"])
    # 城市卡片：区域列表 [格子, 类型, 劳作人数, 连路, 损坏]；首都自带机场
    assert any(d[1] == "airfield" for d in capital["districts"])
    assert {"housing", "pop_cap", "drafted", "defense_value", "max_hp"} <= capital.keys()
    assert all(0 <= u["facing"] <= 5 for u in game["units"])  # 朝向
    assert all("value" in r for r in game["diplomacy"]["relations"])  # 关系值
    soldier = next(u for u in game["units"] if u["type"] == "infantry")
    tactics = client.get(f"/api/games/{game['id']}/tactics", params={"actor": soldier["id"]}).json()
    assert tactics["zoc"] == []  # 开局互相和平：没有敌方控制区
