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
