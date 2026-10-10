from fastapi.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_ai_controls_settler_tactics_and_region_snapshot():
    snap = client.post(
        "/api/games", json={"seed": 5, "width": 40, "height": 25, "nations": 4}
    ).json()
    path = "/api/games/" + snap["id"]
    assert all(not n["ai_enabled"] for n in snap["nations"])
    assert snap["settlement_regions"]["capacities"]
    settler = next(u for u in snap["units"] if u["type"] == "settler")
    tactics = client.get(path + "/tactics", params={"actor": settler["id"]}).json()
    assert tactics["found_city_blocked"]
    enabled = client.post(
        path + "/actions",
        json={
            "expected_hash": snap["state_hash"],
            "orders": [
                {"type": "set_ai", "issuer": n["id"], "enabled": True} for n in snap["nations"]
            ],
        },
    ).json()
    assert enabled["turn"] == 1 and not enabled["rejected"]
    assert all(n["ai_enabled"] and n["city_management"] for n in enabled["nations"])
    advanced = client.post(path + "/turn", json={"autoplay": False}).json()
    assert advanced["turn"] == 2 and not advanced["rejected"]
    assert any(u["ai_goal"] for u in advanced["units"])
    unit = next(u for u in advanced["units"] if u["type"] == "infantry")
    manual = client.post(
        path + "/actions",
        json={"orders": [{"type": "fortify", "issuer": unit["nation"], "unit": unit["id"]}]},
    ).json()
    assert not next(u for u in manual["units"] if u["id"] == unit["id"])["managed"]
    restored = client.post(
        path + "/actions",
        json={
            "orders": [
                {
                    "type": "unit_management",
                    "issuer": unit["nation"],
                    "unit": unit["id"],
                    "enabled": True,
                }
            ]
        },
    ).json()
    assert next(u for u in restored["units"] if u["id"] == unit["id"])["managed"]
    stopped = client.post(
        path + "/actions",
        json={
            "orders": [
                {"type": "set_ai", "issuer": n["id"], "enabled": False} for n in restored["nations"]
            ]
        },
    ).json()
    assert stopped["turn"] == 2 and all(
        not n["ai_enabled"] and not n["city_management"] for n in stopped["nations"]
    )


def test_random_generation_obeys_shared_city_slots():
    snap = client.post(
        "/api/games",
        json={"seed": 42, "width": 80, "height": 50, "nations": 8, "stress_mode": True},
    ).json()
    for c in snap["cities"]:
        assert c["environment"]["regional_cities"] <= c["environment"]["regional_capacity"]
    assert all(not n["ai_enabled"] for n in snap["nations"])
